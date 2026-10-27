# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Batched rigid 3D correspondence alignment for relocalization."""

from __future__ import annotations

import torch


def rigid_align_3d(
    source_xyz_camera: torch.Tensor,
    target_xyz_world: torch.Tensor,
    *,
    matched: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    min_points: int = 6,
) -> dict[str, torch.Tensor]:
    """Fit camera-to-world rigid poses from matched 3D points.

    ``source_xyz_camera`` is transformed into ``target_xyz_world``.  Invalid
    or rank-deficient correspondence sets return identity poses and explicit
    ``valid=False`` rather than a numerically plausible arbitrary transform.
    """
    if (
        source_xyz_camera.ndim != 3
        or source_xyz_camera.shape[-1] != 3
        or not source_xyz_camera.is_floating_point()
        or target_xyz_world.shape != source_xyz_camera.shape
        or target_xyz_world.device != source_xyz_camera.device
        or target_xyz_world.dtype != source_xyz_camera.dtype
        or not isinstance(min_points, int)
        or isinstance(min_points, bool)
        or min_points < 3
    ):
        raise ValueError("XYZ inputs must match floating [B,P,3] and min_points must be at least three")
    batch, points, _ = source_xyz_camera.shape
    if points < min_points:
        raise ValueError("point axis must cover min_points")
    if matched is None:
        matched = torch.ones((batch, points), dtype=torch.bool, device=source_xyz_camera.device)
    elif matched.shape != (batch, points) or matched.dtype != torch.bool or matched.device != source_xyz_camera.device:
        raise ValueError("matched must be bool [B,P] on the XYZ device")
    if confidence is None:
        confidence = torch.ones((batch, points), dtype=source_xyz_camera.dtype, device=source_xyz_camera.device)
    elif (
        confidence.shape != (batch, points)
        or confidence.device != source_xyz_camera.device
        or confidence.dtype != source_xyz_camera.dtype
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError("confidence must be finite [0,1] [B,P] on the XYZ device")
    usable = matched & torch.isfinite(source_xyz_camera).all(-1) & torch.isfinite(target_xyz_world).all(-1)
    weight = torch.where(usable, confidence, torch.zeros_like(confidence))
    source = torch.where(usable[..., None], source_xyz_camera, torch.zeros_like(source_xyz_camera))
    target = torch.where(usable[..., None], target_xyz_world, torch.zeros_like(target_xyz_world))
    weight_sum = weight.sum(1)
    source_mean = (source * weight[..., None]).sum(1) / weight_sum[:, None].clamp_min(torch.finfo(weight.dtype).eps)
    target_mean = (target * weight[..., None]).sum(1) / weight_sum[:, None].clamp_min(torch.finfo(weight.dtype).eps)
    source_centered = source - source_mean[:, None]
    target_centered = target - target_mean[:, None]
    covariance = torch.einsum("bpi,bpj,bp->bij", source_centered, target_centered, weight)
    u, singular, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    rotation = v @ u.transpose(-1, -2)
    correction = torch.eye(3, dtype=source_xyz_camera.dtype, device=source_xyz_camera.device).expand(batch, -1, -1).clone()
    correction[:, 2, 2] = torch.where(torch.linalg.det(rotation) < 0, -1.0, 1.0)
    rotation = v @ correction @ u.transpose(-1, -2)
    translation = target_mean - torch.einsum("bij,bj->bi", rotation, source_mean)
    eps = torch.finfo(source_xyz_camera.dtype).eps
    rank = (singular > singular[:, :1] * (eps * 64)).sum(1)
    support_count = usable.sum(1).to(torch.int64)
    valid = (support_count >= min_points) & (weight_sum > eps) & (rank >= 2) & torch.isfinite(rotation).all((1, 2)) & torch.isfinite(translation).all(1)
    transformed = torch.einsum("bij,bpj->bpi", rotation, source) + translation[:, None]
    squared = (transformed - target).square().sum(-1)
    rmse = ((squared * weight).sum(1) / weight_sum.clamp_min(eps)).sqrt()
    pose = torch.eye(4, dtype=source_xyz_camera.dtype, device=source_xyz_camera.device).expand(batch, -1, -1).clone()
    pose[:, :3, :3] = torch.where(valid[:, None, None], rotation, pose[:, :3, :3])
    pose[:, :3, 3] = torch.where(valid[:, None], translation, pose[:, :3, 3])
    return {
        "pose_c2w": pose,
        "valid": valid,
        "support_count": support_count,
        "rank": rank.to(torch.int64),
        "singular_values": singular,
        "rmse_m": torch.where(valid, rmse, torch.zeros_like(rmse)),
    }
