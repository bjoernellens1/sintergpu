"""Measure fresh versus caller-owned output transaction cost."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch

from vkl.backend import get_backend_info
from vkl.kernels.triton import project_depth_evidence_triton


def make_output_buffers(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.empty_like(value) for name, value in outputs.items()}


def _timing(call, device: torch.device, iterations: int, calls_per_sample: int) -> dict[str, float]:
    events = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            call()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(device)
    values = sorted(start.elapsed_time(end) / calls_per_sample for start, end in events)
    return {"p50_ms": values[len(values) // 2], "p95_ms": values[int(len(values) * 0.95) - 1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--calls-per-sample", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    factory = runpy.run_path(Path(__file__).with_name("bench_project_depth_evidence.py"))["_factory"](480, 640, args.points)
    report = {"backend": get_backend_info(device).as_dict(), "timing_contract": "python_submitted_gpu_event_amortized", "batches": {}}
    for batch in (int(value) for value in args.batches.split(",")):
        inputs, kwargs = factory(device, batch)
        seed = project_depth_evidence_triton(*inputs, **kwargs)
        reused = make_output_buffers(seed)
        for _ in range(args.warmup):
            project_depth_evidence_triton(*inputs, **kwargs)
            project_depth_evidence_triton(*inputs, **kwargs, out=reused)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        fresh_output = project_depth_evidence_triton(*inputs, **kwargs)
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        allocation_count = after["allocation.all.allocated"] - before["allocation.all.allocated"]
        allocation_bytes = after["allocated_bytes.all.allocated"] - before["allocated_bytes.all.allocated"]
        del fresh_output
        report["batches"][str(batch)] = {
            "fresh": _timing(lambda inputs=inputs, kwargs=kwargs: project_depth_evidence_triton(*inputs, **kwargs), device, args.iterations, args.calls_per_sample),
            "reused": _timing(lambda inputs=inputs, kwargs=kwargs, reused=reused: project_depth_evidence_triton(*inputs, **kwargs, out=reused), device, args.iterations, args.calls_per_sample),
            "fresh_output_allocations": allocation_count,
            "fresh_output_allocated_bytes": allocation_bytes,
        }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
