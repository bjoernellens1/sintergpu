"""Measure eager, compiled, and Triton project/depth evidence on GPU."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

from vkl.backend import get_backend_info
from vkl.bench.triton_qualification import (
    KernelCase,
    compare_outputs,
    measure_case,
    serialize_report,
)
from vkl.kernels.reference import project_depth_evidence
from vkl.kernels.triton import project_depth_evidence_triton


def _factory(height: int, width: int, points: int):
    def build(device: torch.device, batch: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
        generator = torch.Generator(device=device).manual_seed(17)
        xyz = torch.rand((batch, points, 3), device=device, generator=generator)
        xyz[..., :2] = (xyz[..., :2] - 0.5) * 4.0
        xyz[..., 2] = xyz[..., 2] * 3.0 + 0.5
        visible = torch.rand((batch, points), device=device, generator=generator) > 0.1
        provisional = torch.rand((batch, points), device=device, generator=generator) > 0.4
        candidate = torch.rand((batch, points), device=device, generator=generator) > 0.6
        w2c = torch.eye(4, device=device)[None].expand(batch, -1, -1).clone()
        intrinsics = torch.eye(3, device=device)[None].expand(batch, -1, -1).clone()
        intrinsics[:, 0, 0] = width * 0.9
        intrinsics[:, 1, 1] = height * 0.9
        intrinsics[:, 0, 2] = (width - 1) / 2
        intrinsics[:, 1, 2] = (height - 1) / 2
        sensor_depth = torch.full((batch, 1, height, width), 2.0, device=device)
        return (xyz, visible, provisional, candidate, w2c, intrinsics, sensor_depth), {
            "support_threshold": 0.05,
            "conflict_threshold": 0.10,
            "validate": False,
        }

    return build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--variants", default="eager,compiled,triton")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.height, args.width, args.points, args.warmup, args.iterations, args.calls_per_sample) < 1:
        raise ValueError("dimensions, warmup, iterations, and calls_per_sample must be positive")
    batches = tuple(int(value) for value in args.batches.split(",") if value)
    variants = tuple(value for value in args.variants.split(",") if value)
    if not variants or any(value not in {"eager", "compiled", "triton"} for value in variants):
        raise ValueError("variants must be a non-empty comma-separated subset of eager,compiled,triton")
    case = KernelCase(
        name="project_depth_evidence",
        family="projection_depth_evidence",
        factory=_factory(args.height, args.width, args.points),
        reference=project_depth_evidence,
        triton=project_depth_evidence_triton,
        parity=compare_outputs,
        workload={
            "batches": batches,
            "algorithm_dimensions": {
                "points": args.points,
                "sensor_depth_hw": [args.height, args.width],
                "support_threshold": 0.05,
                "conflict_threshold": 0.10,
                "validate": False,
            },
        },
    )
    device = torch.device(args.device)
    backend = get_backend_info(device).as_dict()
    serialize_report(
        {
            "metadata": {
                "source_revision": os.getenv("VKL_SOURCE_REVISION", "unknown"),
                "image_revision": os.getenv("VKL_IMAGE_REVISION", "unknown"),
                "image_reference": os.getenv("VKL_IMAGE_REFERENCE", "unknown"),
                "backend": backend,
                "command": " ".join(sys.argv),
            },
            "case": measure_case(
                case,
                variants,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                calls_per_sample=args.calls_per_sample,
            ),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
