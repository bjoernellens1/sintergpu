"""Triton fast path for prevalidated projected sensor-depth evidence."""

# ruff: noqa: PLR0124 -- Triton uses value == value as its portable NaN predicate.

from __future__ import annotations

import math

import torch

from ..reference.project_depth_evidence import project_depth_evidence

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU fallback.
    triton = tl = None


if triton is not None:
    @triton.jit
    def _round_even(value):
        lower = tl.floor(value)
        lower_int = lower.to(tl.int32)
        fraction = value - lower
        increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower_int % 2) != 0))
        return lower_int + increment.to(tl.int32)


    @triton.jit
    def _project_depth_evidence(
        xyz,
        visible,
        provisional,
        depth_candidate,
        world_to_camera,
        intrinsics,
        sensor_depth,
        eligible_out,
        projected_out,
        sensor_out,
        support_out,
        conflict_out,
        residual_out,
        total,
        points,
        support_threshold,
        conflict_threshold,
        epsilon,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offset < total
        batch = offset // points
        safe_batch = tl.where(mask, batch, 0)
        x = tl.load(xyz + offset * 3, mask=mask, other=0.0)
        y = tl.load(xyz + offset * 3 + 1, mask=mask, other=0.0)
        z_world = tl.load(xyz + offset * 3 + 2, mask=mask, other=0.0)
        visible_value = tl.load(visible + offset, mask=mask, other=0).to(tl.int1)
        provisional_value = tl.load(provisional + offset, mask=mask, other=0).to(tl.int1)
        candidate_value = tl.load(depth_candidate + offset, mask=mask, other=0).to(tl.int1)
        finite_xyz = (x == x) & (y == y) & (z_world == z_world) & (x < float("inf")) & (x > -float("inf")) & (y < float("inf")) & (y > -float("inf")) & (z_world < float("inf")) & (z_world > -float("inf"))
        eligible = mask & visible_value & (provisional_value | candidate_value) & finite_xyz
        matrix = world_to_camera + safe_batch * 16
        r00, r01, r02, t0 = tl.load(matrix), tl.load(matrix + 1), tl.load(matrix + 2), tl.load(matrix + 3)
        r10, r11, r12, t1 = tl.load(matrix + 4), tl.load(matrix + 5), tl.load(matrix + 6), tl.load(matrix + 7)
        r20, r21, r22, t2 = tl.load(matrix + 8), tl.load(matrix + 9), tl.load(matrix + 10), tl.load(matrix + 11)
        x_camera = x * r00 + y * r01 + z_world * r02 + t0
        y_camera = x * r10 + y * r11 + z_world * r12 + t1
        z_camera = x * r20 + y * r21 + z_world * r22 + t2
        finite_z = (z_camera == z_camera) & (z_camera < float("inf")) & (z_camera > -float("inf"))
        positive_z = finite_z & (z_camera > epsilon)
        safe_z = tl.where(positive_z, z_camera, 1.0)
        camera = intrinsics + safe_batch * 9
        fx, cx = tl.load(camera), tl.load(camera + 2)
        fy, cy = tl.load(camera + 4), tl.load(camera + 5)
        u = x_camera / safe_z * fx + cx
        v = y_camera / safe_z * fy + cy
        projected_finite = (u == u) & (v == v) & (u < float("inf")) & (u > -float("inf")) & (v < float("inf")) & (v > -float("inf")) & positive_z
        safe_u = tl.where(projected_finite, u, 0.0)
        safe_v = tl.where(projected_finite, v, 0.0)
        ui, vi = _round_even(safe_u), _round_even(safe_v)
        projected = eligible & projected_finite & (ui >= 0) & (ui < WIDTH) & (vi >= 0) & (vi < HEIGHT)
        lookup_index = safe_batch * (HEIGHT * WIDTH) + tl.maximum(0, tl.minimum(HEIGHT - 1, vi)) * WIDTH + tl.maximum(0, tl.minimum(WIDTH - 1, ui))
        lookup = tl.load(sensor_depth + lookup_index, mask=mask, other=0.0)
        sensor_valid = projected & (lookup == lookup) & (lookup < float("inf")) & (lookup > 0.0)
        residual = tl.where(sensor_valid, tl.abs(z_camera - lookup), 0.0)
        support = sensor_valid & (residual < support_threshold)
        conflict = sensor_valid & candidate_value & (z_camera < lookup - conflict_threshold)
        tl.store(eligible_out + offset, eligible, mask=mask)
        tl.store(projected_out + offset, projected, mask=mask)
        tl.store(sensor_out + offset, sensor_valid, mask=mask)
        tl.store(support_out + offset, support, mask=mask)
        tl.store(conflict_out + offset, conflict, mask=mask)
        tl.store(residual_out + offset, residual, mask=mask)


def project_depth_evidence_triton(
    xyz: torch.Tensor,
    visible: torch.Tensor,
    provisional: torch.Tensor,
    depth_candidate: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    sensor_depth: torch.Tensor,
    *,
    support_threshold: float,
    conflict_threshold: float,
    validate: bool = True,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Run one Triton program per prevalidated point, or preserve the reference path.

    ``validate=True`` keeps the defensive reference contract. The fast path is
    explicitly for a caller that already validated camera tensors at its
    transaction boundary. Triton dispatch also requires contiguous inputs; a
    strided input falls back rather than performing a hidden hot-path copy.
    """
    if (
        xyz.ndim != 3
        or xyz.shape[-1] != 3
        or visible.shape != xyz.shape[:2]
        or provisional.shape != visible.shape
        or depth_candidate.shape != visible.shape
        or visible.dtype is not torch.bool
        or provisional.dtype is not torch.bool
        or depth_candidate.dtype is not torch.bool
    ):
        raise ValueError("xyz [B,N,3] and matching boolean masks are required")
    batch, points, _ = xyz.shape
    if (
        world_to_camera.shape != (batch, 4, 4)
        or intrinsics.shape != (batch, 3, 3)
        or sensor_depth.ndim != 4
        or sensor_depth.shape[:2] != (batch, 1)
        or sensor_depth.shape[-2] < 1
        or sensor_depth.shape[-1] < 1
    ):
        raise ValueError("camera matrices and non-empty sensor_depth [B,1,H,W] must match xyz batch")
    tensors = (visible, provisional, depth_candidate, world_to_camera, intrinsics, sensor_depth)
    if (
        any(value.device != xyz.device for value in tensors)
        or not xyz.is_floating_point()
        or any(value.dtype != xyz.dtype for value in (world_to_camera, intrinsics, sensor_depth))
    ):
        raise ValueError("all tensors must share xyz device and floating dtype")
    if (
        not math.isfinite(support_threshold)
        or not math.isfinite(conflict_threshold)
        or support_threshold <= 0
        or conflict_threshold <= 0
    ):
        raise ValueError("thresholds must be finite and positive")
    if not isinstance(validate, bool):
        raise ValueError("validate must be bool")  # noqa: TRY004 - preserve reference contract.

    output_shapes = {
        "eligible": visible.shape,
        "projected_valid": visible.shape,
        "sensor_valid": visible.shape,
        "support": visible.shape,
        "conflict": visible.shape,
        "depth_residual": xyz.shape[:2],
    }
    if out is not None:
        if set(out) != set(output_shapes):
            raise ValueError("out must contain every project-depth evidence output")
        for name, shape in output_shapes.items():
            value = out[name]
            dtype = xyz.dtype if name == "depth_residual" else torch.bool
            if value.shape != shape or value.dtype != dtype or value.device != xyz.device or not value.is_contiguous():
                raise ValueError("out tensors must be contiguous, typed, and shaped for this transaction")

    total = batch * points
    fallback_reason = None
    if validate:
        fallback_reason = "defensive_validation"
    elif xyz.device.type != "cuda":
        fallback_reason = "non_gpu_device"
    elif triton is None:
        fallback_reason = "triton_unavailable"
    elif xyz.dtype not in (torch.float16, torch.float32):
        fallback_reason = "unsupported_dtype"
    elif total == 0:
        fallback_reason = "empty_workload"
    elif not all(value.is_contiguous() for value in (xyz, *tensors)):
        fallback_reason = "noncontiguous_input"

    if fallback_reason is not None:
        project_depth_evidence_triton.dispatch_metadata = {
            "requested": "triton",
            "actual": "eager_fallback",
            "reason": fallback_reason,
        }
        result = project_depth_evidence(
            xyz,
            visible,
            provisional,
            depth_candidate,
            world_to_camera,
            intrinsics,
            sensor_depth,
            support_threshold=support_threshold,
            conflict_threshold=conflict_threshold,
            validate=validate,
        )
        if out is None:
            return result
        for name, value in result.items():
            out[name].copy_(value)
        return out

    if out is None:
        eligible = torch.empty_like(visible)
        projected = torch.empty_like(visible)
        sensor_valid = torch.empty_like(visible)
        support = torch.empty_like(visible)
        conflict = torch.empty_like(visible)
        residual = torch.empty_like(xyz[..., 0])
    else:
        eligible = out["eligible"]
        projected = out["projected_valid"]
        sensor_valid = out["sensor_valid"]
        support = out["support"]
        conflict = out["conflict"]
        residual = out["depth_residual"]
    height, width = sensor_depth.shape[-2:]
    block = 256
    _project_depth_evidence[(triton.cdiv(total, block),)](
        xyz,
        visible,
        provisional,
        depth_candidate,
        world_to_camera,
        intrinsics,
        sensor_depth,
        eligible,
        projected,
        sensor_valid,
        support,
        conflict,
        residual,
        total,
        points,
        support_threshold,
        conflict_threshold,
        torch.finfo(xyz.dtype).eps,
        HEIGHT=height,
        WIDTH=width,
        BLOCK=block,
    )
    project_depth_evidence_triton.dispatch_metadata = {"requested": "triton", "actual": "triton"}
    return {
        "eligible": eligible,
        "projected_valid": projected,
        "sensor_valid": sensor_valid,
        "support": support,
        "conflict": conflict,
        "depth_residual": residual,
    }


project_depth_evidence_triton.dispatch_metadata = {"requested": "triton", "actual": "not_dispatched"}
