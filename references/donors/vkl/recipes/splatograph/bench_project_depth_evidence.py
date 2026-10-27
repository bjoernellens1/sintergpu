"""Measure the Splatograph-shaped batched projection/depth evidence transaction."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from vkl.backend import get_backend_info
from vkl.bench.input_isolation import isolated_call
from vkl.kernels import project_depth_evidence


_SUPPORTS_VALIDATE = "validate" in inspect.signature(project_depth_evidence).parameters


def _inputs(batch: int, count: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    xyz = torch.randn(batch, count, 3, device=device)
    xyz[..., 2].abs_().add_(0.25)
    visible = torch.rand(batch, count, device=device) > 0.2
    provisional = torch.rand(batch, count, device=device) > 0.5
    candidate = torch.rand(batch, count, device=device) > 0.5
    w2c = torch.eye(4, device=device).expand(batch, -1, -1).clone()
    intrinsics = torch.tensor(
        [[width * 0.9, 0.0, (width - 1) * 0.5], [0.0, height * 0.9, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch, -1, -1).clone()
    sensor_depth = torch.rand(batch, 1, height, width, device=device).add_(0.25)
    return xyz, visible, provisional, candidate, w2c, intrinsics, sensor_depth


def _invoke(fn, values: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    kwargs = {"support_threshold": 0.03, "conflict_threshold": 0.08}
    if _SUPPORTS_VALIDATE:
        # Benchmark the prevalidated hot transaction. Value-level camera checks
        # belong at the transaction boundary and must not break fullgraph compile.
        kwargs["validate"] = False
    return fn(*values, **kwargs)


def _samples(fn, values: tuple[torch.Tensor, ...], *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        _invoke(fn, values)
    if values[0].device.type == "cuda":
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _invoke(fn, values)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(start.elapsed_time(end))
        return elapsed
    elapsed = []
    for _ in range(iterations):
        start = perf_counter()
        _invoke(fn, values)
        elapsed.append((perf_counter() - start) * 1000)
    return elapsed


def _summary(samples: list[float], points: int) -> dict[str, float]:
    values = torch.tensor(samples)
    mean = values.mean().item()
    return {
        "mean_ms": mean,
        "p50_ms": values.quantile(0.5).item(),
        "p95_ms": values.quantile(0.95).item(),
        "serialized_points_per_s": points * 1000.0 / mean,
    }


def _workload(output: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {
        f"{name}_fraction": value.float().mean().item()
        for name, value in output.items()
        if value.dtype == torch.bool
    }
    sensor_valid = output["sensor_valid"]
    residual = output["depth_residual"][sensor_valid]
    result["sensor_residual_mean"] = residual.mean().item() if residual.numel() else 0.0
    result["sensor_residual_p95"] = torch.quantile(residual, 0.95).item() if residual.numel() else 0.0
    return result


def _memory(fn, values: tuple[torch.Tensor, ...]) -> dict[str, int] | None:
    if values[0].device.type != "cuda":
        return None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _invoke(fn, values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "baseline_allocated_bytes": int(baseline),
        "peak_allocated_bytes": int(peak),
        "incremental_peak_bytes": int(max(0, peak - baseline)),
    }


def _parity(eager: dict[str, torch.Tensor], compiled: dict[str, torch.Tensor]) -> dict[str, float | int | bool]:
    boolean = [name for name, value in eager.items() if value.dtype == torch.bool]
    mismatch = sum(int((eager[name] != compiled[name]).sum()) for name in boolean)
    eager_residual = eager["depth_residual"]
    compiled_residual = compiled["depth_residual"]
    residual_close = torch.isclose(eager_residual, compiled_residual, rtol=1e-5, atol=1e-6)
    residual_error = (eager_residual - compiled_residual).abs()
    residual_mismatch = ~residual_close
    return {
        "boolean_exact": mismatch == 0,
        "boolean_mismatch_count": mismatch,
        "residual_allclose": bool(residual_close.all()),
        "residual_mismatch_count": int(residual_mismatch.sum()),
        "residual_mismatch_fraction": residual_mismatch.float().mean().item(),
        "residual_max_abs_error": float(residual_error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--counts", default="50000,250000")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.height, args.width, args.warmup, args.iterations) < 1 or args.seed < 0:
        raise ValueError("positive resolution/timing counts and a non-negative seed are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    records = []
    for batch in (int(value) for value in args.batches.split(",")):
        for count in (int(value) for value in args.counts.split(",")):
            if min(batch, count) < 1:
                raise ValueError("batches and point counts must be positive")
            values = _inputs(batch, count, args.height, args.width, device)
            eager_out = isolated_call(lambda *call_values: _invoke(project_depth_evidence, call_values), values)
            record = {
                "batch": batch,
                "points_per_batch": count,
                "points_total": batch * count,
                "workload": _workload(eager_out),
                "eager": _summary(_samples(project_depth_evidence, values, warmup=args.warmup, iterations=args.iterations), batch * count),
                "eager_memory": _memory(project_depth_evidence, values),
                "compile_requested": True,
                "compile_actual": "unsupported",
            }
            try:
                compiled = torch.compile(project_depth_evidence, fullgraph=True)
                compiled_out = isolated_call(lambda *call_values: _invoke(compiled, call_values), values)
                parity = _parity(eager_out, compiled_out)
                record["output_parity"] = parity
                record["compile_actual"] = "compiled" if parity["boolean_exact"] and parity["residual_allclose"] else "compiled-boundary-difference"
                record["compiled"] = _summary(_samples(compiled, values, warmup=args.warmup, iterations=args.iterations), batch * count)
                record["compiled_memory"] = _memory(compiled, values)
            except Exception as exc:
                record["compile_error"] = f"{type(exc).__name__}: {exc}"[:500]
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "backend": get_backend_info(device).as_dict(),
        "resolution": [args.height, args.width],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "timing_mode": "serialized_latency",
        "validation_mode": "prevalidated" if _SUPPORTS_VALIDATE else "kernel-default",
        "source_revision": os.getenv("VKL_SOURCE_REVISION", "unknown"),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
