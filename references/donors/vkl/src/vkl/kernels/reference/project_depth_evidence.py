"""Batched projection and sensor-depth classification for Gaussian evidence."""

from __future__ import annotations

import math

import torch

from .sparse_geometry import _intrinsics, _pose


def project_depth_evidence(
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
) -> dict[str, torch.Tensor]:
    """Classify visible world points against one metric sensor-depth frame.

    All point-state mutation and per-view de-duplication remain caller policy.
    Inputs are batched ``[B,N]`` / ``[B,N,3]`` tensors; poses are world-to-camera.
    Non-finite point samples are masked from geometric evidence rather than
    becoming projected border samples.

    ``validate=False`` skips value-level camera validation for callers that have
    already validated camera state at a transaction boundary. This keeps the
    hot reference eligible for ``torch.compile(fullgraph=True)`` while the
    public default remains defensive.
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
    batch, _, _ = xyz.shape
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
        raise ValueError("validate must be bool")

    if validate:
        _intrinsics(intrinsics, batch, xyz.device, xyz.dtype)
        # The rigid-transform check is convention agnostic even though the shared
        # helper is named for the c2w geometry path.
        _pose(world_to_camera, batch, xyz.device, xyz.dtype)

    point_finite = torch.isfinite(xyz).all(-1)
    eligible = visible & (provisional | depth_candidate) & point_finite
    camera = xyz @ world_to_camera[:, :3, :3].transpose(-1, -2) + world_to_camera[:, None, :3, 3]
    z = camera[..., 2]
    positive_z = torch.isfinite(z) & (z > torch.finfo(xyz.dtype).eps)
    safe_z = torch.where(positive_z, z, torch.ones_like(z))
    u = camera[..., 0] / safe_z * intrinsics[:, None, 0, 0] + intrinsics[:, None, 0, 2]
    v = camera[..., 1] / safe_z * intrinsics[:, None, 1, 1] + intrinsics[:, None, 1, 2]
    projected_finite = torch.isfinite(u) & torch.isfinite(v) & positive_z
    safe_u = torch.where(projected_finite, u, torch.zeros_like(u))
    safe_v = torch.where(projected_finite, v, torch.zeros_like(v))

    height, width = sensor_depth.shape[-2:]
    ui, vi = safe_u.round().long(), safe_v.round().long()
    projected_valid = (
        eligible
        & projected_finite
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )
    lookup_index = vi.clamp(0, height - 1) * width + ui.clamp(0, width - 1)
    lookup = sensor_depth[:, 0].flatten(1).gather(1, lookup_index)
    sensor_valid = projected_valid & torch.isfinite(lookup) & (lookup > 0)
    residual = torch.where(sensor_valid, (z - lookup).abs(), torch.zeros_like(z))
    support = sensor_valid & (residual < support_threshold)
    conflict = sensor_valid & depth_candidate & (z < lookup - conflict_threshold)
    return {
        "eligible": eligible,
        "projected_valid": projected_valid,
        "sensor_valid": sensor_valid,
        "support": support,
        "conflict": conflict,
        "depth_residual": residual,
    }
