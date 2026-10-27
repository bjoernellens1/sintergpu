from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from .compiler import compile_slang, find_slangc
from .demo import demo_kernel
from .emitter import emit_slang


def _doctor() -> int:
    checks = {
        "python": sys.version.split()[0],
        "symforce": bool(importlib.util.find_spec("symforce")),
        "slangc": find_slangc() or False,
        "slangpy": bool(importlib.util.find_spec("slangpy")),
        "vulkaninfo": shutil.which("vulkaninfo") or False,
    }
    width = max(map(len, checks))
    for name, value in checks.items():
        print(f"{name:<{width}} : {value}")
    return 0


def _demo(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(emit_slang(demo_kernel()))
    print(output)
    return 0


def _compile(source: Path, output: Path | None, target: str) -> int:
    if output is None:
        suffix = {"spirv": ".spv", "cuda": ".cu", "ptx": ".ptx", "metal": ".metal", "cpp": ".cpp"}[
            target
        ]
        output = source.with_suffix(suffix)
    compile_slang(source, output, target=target)
    print(output)
    return 0


def _validate_gpu(size: int, reduction: str) -> int:
    import numpy as np

    from .gn import GaussNewton, reference, synthetic_problem

    solver = GaussNewton(reduction=reduction)
    inputs = synthetic_problem(size)
    result = solver.step(*inputs, damping=1e-3)
    expected = reference(*inputs, damping=1e-3)
    errors = {}
    for name in ("residual", "jacobian", "hessian", "gradient", "delta"):
        actual = getattr(result, name)
        np.testing.assert_allclose(actual, expected[name], atol=3e-5, rtol=3e-4)
        errors[name] = float(np.max(np.abs(actual - expected[name])))
    if result.status != expected["status"]:
        raise RuntimeError("GPU solver status disagrees with reference")
    print(
        json.dumps(
            {
                "validation": "PASS",
                "backend": "vulkan",
                "device": str(solver.runtime.device.info.adapter_name),
                "size": size,
                "reduction": reduction,
                "max_abs_error": errors,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="caspar-slang")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor")

    demo = sub.add_parser("demo")
    demo.add_argument("--output", type=Path, default=Path("build/demo.slang"))

    comp = sub.add_parser("compile")
    comp.add_argument("source", type=Path)
    comp.add_argument("--output", type=Path)
    comp.add_argument("--target", choices=["spirv", "cuda", "ptx", "metal", "cpp"], default="spirv")

    validate = sub.add_parser("validate", help="run real Caspar/Vulkan GN against float64 CPU")
    validate.add_argument("--size", type=int, default=4099)
    validate.add_argument("--reduction", choices=["workgroup", "subgroup"], default="workgroup")
    benchmark = sub.add_parser("benchmark", help="time prepared GPU GN dispatches")
    benchmark.add_argument("--size", type=int, default=4099)
    benchmark.add_argument("--iterations", type=int, default=20)
    benchmark.add_argument("--reduction", choices=["workgroup", "subgroup"], default="workgroup")

    args = parser.parse_args()
    if args.command == "doctor":
        return _doctor()
    if args.command == "demo":
        return _demo(args.output)
    if args.command == "compile":
        return _compile(args.source, args.output, args.target)
    if args.command == "validate":
        return _validate_gpu(args.size, args.reduction)
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        print(json.dumps(run_benchmark(args.size, args.iterations, args.reduction), indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
