# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Stateless multiview validation for proposed persistent landmarks."""

from __future__ import annotations

import math

import torch

from .sparse_geometry import _intrinsics, _pose


def landmark_multiview_validate(
    xyz_world: torch.Tensor,
    observed_xy: torch.Tensor,
    observed_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    poses_c2w: torch.Tensor,
    *,
    landmark_valid: torch.Tensor | None = None,
    observed_valid: torch.Tensor | None = None,
    min_support: int = 2,
    max_reprojection_px: float = 1.0,
    max_relative_depth_error: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Accept landmarks supported by multiple consistent posed RGB-D views."""
    if xyz_world.ndim != 3 or xyz_world.shape[-1] != 3 or not xyz_world.is_floating_point():
        raise ValueError("xyz_world must be floating [B,M,3]")
    batch, landmarks, _ = xyz_world.shape
    if batch < 1 or landmarks < 1:
        raise ValueError("xyz_world batch and landmark axes must be non-empty")
    if (
        observed_xy.ndim != 4
        or observed_xy.shape[0] != batch
        or observed_xy.shape[2:] != (landmarks, 2)
        or observed_depth.shape != observed_xy.shape[:3]
    ):
        raise ValueError("observations must be [B,N,M,2] XY and [B,N,M] depth")
    views = observed_xy.shape[1]
    if (
        intrinsics.shape != (batch, views, 3, 3)
        or poses_c2w.shape != (batch, views, 4, 4)
        or any(
            item.device != xyz_world.device or item.dtype != xyz_world.dtype
            for item in (observed_xy, observed_depth, intrinsics, poses_c2w)
        )
    ):
        raise ValueError("observations, intrinsics, and poses must share batch/device/dtype")
    if (
        isinstance(min_support, bool)
        or not isinstance(min_support, int)
        or not 1 <= min_support <= views
        or not math.isfinite(max_reprojection_px)
        or not math.isfinite(max_relative_depth_error)
        or max_reprojection_px < 0
        or max_relative_depth_error < 0
    ):
        raise ValueError("multiview thresholds must be finite and valid")

    _intrinsics(
        intrinsics.reshape(batch * views, 3, 3),
        batch * views,
        xyz_world.device,
        xyz_world.dtype,
    )
    _pose(
        poses_c2w.reshape(batch * views, 4, 4),
        batch * views,
        xyz_world.device,
        xyz_world.dtype,
    )
    if landmark_valid is None:
        landmark_valid = torch.ones(
            (batch, landmarks), dtype=torch.bool, device=xyz_world.device
        )
    elif (
        landmark_valid.shape != (batch, landmarks)
        or landmark_valid.dtype != torch.bool
        or landmark_valid.device != xyz_world.device
    ):
        raise ValueError("landmark_valid must be bool [B,M] on landmark device")
    if observed_valid is None:
        observed_valid = torch.ones(
            (batch, views, landmarks), dtype=torch.bool, device=xyz_world.device
        )
    elif (
        observed_valid.shape != (batch, views, landmarks)
        or observed_valid.dtype != torch.bool
        or observed_valid.device != xyz_world.device
    ):
        raise ValueError("observed_valid must be bool [B,N,M] on landmark device")

    rotation = poses_c2w[:, :, :3, :3].transpose(-1, -2)
    xyz_camera = torch.einsum(
        "bnij,bnmj->bnmi",
        rotation,
        xyz_world[:, None] - poses_c2w[:, :, None, :3, 3],
    )
    z = xyz_camera[..., 2]
    eps = torch.finfo(xyz_world.dtype).eps
    safe_z = z.clamp_min(eps)
    projected_xy = torch.stack(
        (
            intrinsics[:, :, None, 0, 0] * xyz_camera[..., 0] / safe_z
            + intrinsics[:, :, None, 0, 2],
            intrinsics[:, :, None, 1, 1] * xyz_camera[..., 1] / safe_z
            + intrinsics[:, :, None, 1, 2],
        ),
        -1,
    )
    reprojection = (projected_xy - observed_xy).square().sum(-1).sqrt()
    depth_error = (z - observed_depth).abs() / safe_z
    support = (
        landmark_valid[:, None]
        & observed_valid
        & torch.isfinite(xyz_world).all(-1)[:, None]
        & torch.isfinite(observed_xy).all(-1)
        & torch.isfinite(observed_depth)
        & (observed_depth > 0)
        & torch.isfinite(projected_xy).all(-1)
        & (z > 0)
        & (reprojection <= max_reprojection_px)
        & (depth_error <= max_relative_depth_error)
    )
    count = support.sum(1)
    support_count = support.to(xyz_world.dtype).sum(1).clamp_min(1)
    reprojection_sq = torch.where(
        support, reprojection.square(), torch.zeros_like(reprojection)
    )
    depth_residual = torch.where(
        support, depth_error, torch.zeros_like(depth_error)
    )
    reprojection_rmse = (reprojection_sq.sum(1) / support_count).sqrt()
    depth_mae = depth_residual.sum(1) / support_count
    valid = landmark_valid & (count >= min_support)
    return {
        "support_count": count,
        "reprojection_rmse": torch.where(
            valid, reprojection_rmse, torch.zeros_like(reprojection_rmse)
        ),
        "relative_depth_mae": torch.where(valid, depth_mae, torch.zeros_like(depth_mae)),
        "valid": valid,
    }
