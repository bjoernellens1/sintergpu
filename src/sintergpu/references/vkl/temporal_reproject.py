# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Pose/depth-aligned temporal RGB sampling built on D4 reprojection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .reproject import rasterize_reprojection, reproject_rgbd
from .temporal_depth_fuse import temporal_depth_fuse


def temporal_reproject_rgb(
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor,
    source_rgb: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_pose_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    *,
    source_depth: torch.Tensor | None = None,
    source_confidence: torch.Tensor | None = None,
    consistency_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Sample ``[B,N]`` source RGB frames into target pixels using D4 geometry."""
    if target_rgb.ndim != 4 or target_rgb.shape[1] != 3 or target_depth.shape != (target_rgb.shape[0], 1, *target_rgb.shape[-2:]) or source_rgb.ndim != 5 or source_rgb.shape[:2] != (target_rgb.shape[0], source_intrinsics.shape[1]) or source_rgb.shape[2:] != target_rgb.shape[1:]:
        raise ValueError("target RGB-D and source RGB [B,N,3,H,W] must align")
    b, n, _, h, w = source_rgb.shape
    if target_intrinsics.shape != (b, 3, 3) or target_pose_c2w.shape != (b, 4, 4) or source_intrinsics.shape != (b, n, 3, 3) or source_pose_c2w.shape != (b, n, 4, 4):
        raise ValueError("target and source camera geometry must match RGB batches")
    tensors = (target_depth, source_rgb, target_intrinsics, target_pose_c2w, source_intrinsics, source_pose_c2w)
    if any(t.device != target_rgb.device or t.dtype != target_rgb.dtype for t in tensors) or not target_rgb.is_floating_point() or consistency_threshold <= 0:
        raise ValueError("all tensors must share floating device/dtype")
    expected = (b, n, 1, h, w)
    if source_depth is not None and (source_depth.shape != expected or source_depth.device != target_rgb.device or source_depth.dtype != target_rgb.dtype):
        raise ValueError("source_depth must match [B,N,1,H,W]")
    if source_confidence is not None and (source_confidence.shape != expected or source_confidence.device != target_rgb.device or source_confidence.dtype != target_rgb.dtype or not torch.isfinite(source_confidence).all() or (source_confidence < 0).any() or (source_confidence > 1).any()):
        raise ValueError("source_confidence must be finite [0,1] and match source RGB")
    repeat_depth = target_depth[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, 1, h, w)
    projected = reproject_rgbd(
        repeat_depth,
        target_intrinsics[:, None].expand(-1, n, -1, -1).reshape(b * n, 3, 3),
        target_pose_c2w[:, None].expand(-1, n, -1, -1).reshape(b * n, 4, 4),
        source_intrinsics.reshape(b * n, 3, 3),
        source_pose_c2w.reshape(b * n, 4, 4),
        None if source_depth is None else source_depth.reshape(b * n, 1, h, w),
        target_hw=(h, w),
        consistency_threshold=consistency_threshold,
    )
    xy = projected["projected_xy"].permute(0, 2, 3, 1)
    grid = torch.stack((2 * xy[..., 0] / max(w - 1, 1) - 1, 2 * xy[..., 1] / max(h - 1, 1) - 1), -1)
    aligned_rgb = F.grid_sample(source_rgb.reshape(b * n, 3, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    aligned_valid = projected["visible"]
    confidence = aligned_valid.to(target_rgb.dtype)
    if source_depth is not None:
        consistent = projected["depth_consistency_valid"] & ~projected["occluded"]
        aligned_valid = aligned_valid & consistent
        confidence = confidence * torch.exp(-projected["depth_consistency"] / consistency_threshold) * consistent
    if source_confidence is not None:
        sampled_confidence = F.grid_sample(source_confidence.reshape(b * n, 1, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        confidence = confidence * sampled_confidence
    return {
        "aligned_rgb": aligned_rgb.reshape(b, n, 3, h, w),
        "aligned_valid": aligned_valid.reshape(b, n, 1, h, w),
        "alignment_confidence": confidence.reshape(b, n, 1, h, w),
    }


def temporal_rgbd_reference(
    *,
    target_depth: torch.Tensor,
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_pose_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    source_confidence: torch.Tensor | None = None,
    consistency_threshold: float = 0.05,
    min_support: int = 1,
) -> dict[str, torch.Tensor]:
    """Build a target-camera RGB-D reference from prior RGB-D observations only.

    D4 z-buffers every source depth into the target camera, T0 samples source
    RGB through current depth geometry, and D9 fuses projected prior depths.
    ``target_depth`` never contributes depth to the returned reference.
    """
    if source_rgb.ndim != 5 or source_rgb.shape[2] != 3 or source_depth.shape != (source_rgb.shape[0], source_rgb.shape[1], 1, *source_rgb.shape[-2:]):
        raise ValueError("source RGB-D must be [B,N,3,H,W]/[B,N,1,H,W]")
    b, n, _, h, w = source_rgb.shape
    if n < 1 or target_depth.shape != (b, 1, h, w) or target_intrinsics.shape != (b, 3, 3) or target_pose_c2w.shape != (b, 4, 4) or source_intrinsics.shape != (b, n, 3, 3) or source_pose_c2w.shape != (b, n, 4, 4):
        raise ValueError("target and source RGB-D camera geometry must match")
    if source_confidence is None:
        source_confidence = torch.ones_like(source_depth)
    if source_confidence.shape != source_depth.shape:
        raise ValueError("source_confidence must match source_depth")
    aligned = temporal_reproject_rgb(
        torch.zeros((b, 3, h, w), device=source_rgb.device, dtype=source_rgb.dtype),
        target_depth,
        source_rgb,
        target_intrinsics,
        target_pose_c2w,
        source_intrinsics,
        source_pose_c2w,
        source_depth=source_depth,
        source_confidence=source_confidence,
        consistency_threshold=consistency_threshold,
    )
    raster = rasterize_reprojection(
        source_depth.reshape(b * n, 1, h, w),
        source_confidence.reshape(b * n, 1, h, w),
        source_intrinsics.reshape(b * n, 3, 3),
        source_pose_c2w.reshape(b * n, 4, 4),
        target_intrinsics[:, None].expand(-1, n, -1, -1).reshape(b * n, 3, 3),
        target_pose_c2w[:, None].expand(-1, n, -1, -1).reshape(b * n, 4, 4),
        target_hw=(h, w),
    )
    raster_depth = raster["depth"].reshape(b, n, 1, h, w)
    raster_valid = raster["valid"].reshape(b, n, 1, h, w)
    raster_confidence = raster["confidence"].reshape(b, n, 1, h, w)
    fused = temporal_depth_fuse(
        torch.zeros_like(target_depth),
        list(raster_depth.unbind(1)),
        torch.zeros_like(target_depth, dtype=torch.bool),
        list(raster_valid.unbind(1)),
        torch.zeros_like(target_depth),
        list(raster_confidence.unbind(1)),
        consistency_threshold=consistency_threshold,
        min_support=min_support,
        min_temporal_support=1,
    )
    weight = aligned["alignment_confidence"] * raster_confidence * raster_valid.to(source_rgb.dtype)
    weight = weight * aligned["aligned_valid"].to(source_rgb.dtype)
    weight_sum = weight.sum(1)
    valid = fused["valid"] & (weight_sum > 0)
    rgb = (aligned["aligned_rgb"] * weight).sum(1) / weight_sum.clamp_min(torch.finfo(source_rgb.dtype).eps)
    return {
        "rgb": torch.where(valid, rgb, torch.zeros_like(rgb)),
        "depth": torch.where(valid, fused["depth"], torch.zeros_like(fused["depth"])),
        "valid": valid,
        "confidence": torch.where(valid, fused["confidence"] * weight_sum.clamp_max(1), torch.zeros_like(weight_sum)),
        "support_count": fused["support_count"],
        "temporal_support_count": fused["temporal_support_count"],
        "source_disagreement": fused["source_disagreement"],
    }
