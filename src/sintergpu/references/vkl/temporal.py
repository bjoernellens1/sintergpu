# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Already-aligned temporal RGB restoration."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_restore_rgb(
    target_rgb: torch.Tensor,
    aligned_rgb: torch.Tensor,
    *,
    alignment_confidence: torch.Tensor | None = None,
    aligned_valid: torch.Tensor | None = None,
    depth_consistency: torch.Tensor | None = None,
    photometric_agreement: torch.Tensor | None = None,
    neighbor_sharpness: torch.Tensor | None = None,
    restoration_support: torch.Tensor | None = None,
    temporal_distance: torch.Tensor | None = None,
    keypoint_confidence: torch.Tensor | None = None,
    tile_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Restore target detail from already aligned, equivalent RGB observations.

    Only the target's low-frequency colour is retained; context contributes a
    gated, confidence-weighted high-frequency residual.  Alignment, validity,
    consistency, and optional quality evidence are all per-pixel tensors with
    shape ``[B,N,1,H,W]`` (except ``temporal_distance``, which may be ``[B,N]``).
    """
    if target_rgb.ndim != 4 or target_rgb.shape[1] != 3 or not target_rgb.is_floating_point():
        raise ValueError("target_rgb must be floating [B,3,H,W]")
    if aligned_rgb.ndim != 5 or aligned_rgb.shape[0] != target_rgb.shape[0] or aligned_rgb.shape[2:] != target_rgb.shape[1:]:
        raise ValueError("aligned_rgb must be [B,N,3,H,W] matching target_rgb")
    if aligned_rgb.device != target_rgb.device or aligned_rgb.dtype != target_rgb.dtype or aligned_rgb.shape[1] == 0:
        raise ValueError("RGB tensors must share device/dtype and contain neighbors")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    b, n, _, h, w = aligned_rgb.shape
    shape = (b, n, 1, h, w)

    def evidence(value: torch.Tensor | None, name: str, default: float) -> torch.Tensor:
        if value is None:
            return torch.full(shape, default, dtype=target_rgb.dtype, device=target_rgb.device)
        if value.shape != shape or value.device != target_rgb.device or value.dtype != target_rgb.dtype:
            raise ValueError(f"{name} must be floating [B,N,1,H,W]")
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError(f"{name} must be finite and in [0,1]")
        return value

    confidence = evidence(alignment_confidence, "alignment_confidence", 1.0)
    valid = torch.ones(shape, dtype=torch.bool, device=target_rgb.device) if aligned_valid is None else aligned_valid
    if valid.shape != shape or valid.dtype is not torch.bool or valid.device != target_rgb.device:
        raise ValueError("aligned_valid must be bool [B,N,1,H,W]")
    depth = evidence(depth_consistency, "depth_consistency", 1.0)
    photo = evidence(photometric_agreement, "photometric_agreement", 1.0)
    if neighbor_sharpness is None:
        gray_target = target_rgb[:, 0:1] * 0.299 + target_rgb[:, 1:2] * 0.587 + target_rgb[:, 2:3] * 0.114
        gray_neighbors = aligned_rgb[:, :, 0:1] * 0.299 + aligned_rgb[:, :, 1:2] * 0.587 + aligned_rgb[:, :, 2:3] * 0.114
        target_grad = F.avg_pool2d((gray_target[..., 1:] - gray_target[..., :-1]).abs(), (1, 2), stride=1, padding=(0, 1))
        source_grad = F.avg_pool2d((gray_neighbors.flatten(0, 1)[..., 1:] - gray_neighbors.flatten(0, 1)[..., :-1]).abs(), (1, 2), stride=1, padding=(0, 1)).reshape(b, n, 1, h, w)
        sharpness = ((source_grad - target_grad[:, None]).relu() / (source_grad + target_grad[:, None] + 1e-3)).clamp(0, 1)
    else:
        sharpness = evidence(neighbor_sharpness, "neighbor_sharpness", 0.0)
    support = evidence(restoration_support, "restoration_support", 1.0)
    source_support = support.mean((-1, -2, -3), keepdim=True) >= 0.09
    distance = evidence(temporal_distance, "temporal_distance", 0.0) if temporal_distance is not None and temporal_distance.ndim == 5 else None
    if temporal_distance is not None and distance is None:
        if temporal_distance.shape != (b, n) or temporal_distance.device != target_rgb.device or not temporal_distance.is_floating_point() or (temporal_distance < 0).any():
            raise ValueError("temporal_distance must be non-negative floating [B,N] or [B,N,1,H,W]")
        distance = temporal_distance[:, :, None, None, None].expand(shape)
    if keypoint_confidence is not None:
        keypoint = evidence(keypoint_confidence, "keypoint_confidence", 1.0)
    else:
        keypoint = torch.ones_like(confidence)

    target_low = F.avg_pool2d(target_rgb, 5, stride=1, padding=2, count_include_pad=False)
    neighbor_low = F.avg_pool2d(aligned_rgb.flatten(0, 1), 5, stride=1, padding=2, count_include_pad=False).reshape(b, n, 3, h, w)
    high_frequency = aligned_rgb - neighbor_low
    weight = confidence * depth * photo * sharpness * support * keypoint * valid.to(target_rgb.dtype)
    if distance is not None:
        weight = weight * torch.exp(-distance)
    weight = torch.where((confidence >= 0.25) & (depth >= 0.8) & (photo >= 0.5) & (support >= 0.25) & source_support, weight, torch.zeros_like(weight))
    total = weight.sum(1)
    restored_detail = (high_frequency * weight).sum(1) / total.clamp_min(1e-6)
    restore_valid = total >= 0.25
    restored_rgb = torch.where(restore_valid, target_low + restored_detail, target_rgb)
    restore_confidence = (1 - torch.exp(-total)).clamp(0, 1) * restore_valid.to(target_rgb.dtype)
    restore_gain = restored_detail.abs().mean(1, keepdim=True) * restore_valid.to(target_rgb.dtype)
    tile = F.adaptive_avg_pool2d(torch.cat((restore_confidence, restore_gain, restore_valid.to(target_rgb.dtype), (total > 0).to(target_rgb.dtype)), dim=1), (max(1, h // tile_size), max(1, w // tile_size)))
    frame = torch.stack((restore_confidence.mean((-1, -2, -3)), restore_gain.mean((-1, -2, -3)), restore_valid.to(target_rgb.dtype).mean((-1, -2, -3)), (total > 0).to(target_rgb.dtype).mean((-1, -2, -3))), dim=1)
    return {"restored_rgb": restored_rgb, "restore_confidence": restore_confidence, "restore_gain": restore_gain, "restore_valid": restore_valid, "restore_tile_evidence": tile, "restore_frame_evidence": frame}


def temporal_restore_rgbd(
    rgb: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    context_valid: torch.Tensor | None = None,
    context_confidence: torch.Tensor | None = None,
    alignment: str = "rgb_aligned",
    lookahead: int = 0,
) -> dict[str, torch.Tensor | str]:
    """Confidence-fuse current RGB with caller-aligned temporal context.

    ``alignment`` documents whether context was aligned in image space or by
    externally performed RGB-D+pose reprojection; this kernel performs no warp.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point() or context_rgb.ndim != 5 or context_rgb.shape[0] != rgb.shape[0] or context_rgb.shape[2:] != rgb.shape[1:] or context_rgb.device != rgb.device or context_rgb.dtype != rgb.dtype:
        raise ValueError("rgb [B,3,H,W] and context_rgb [B,N,3,H,W] must match")
    if alignment not in {"rgb_aligned", "rgbd_reprojected"} or lookahead < 0 or lookahead > context_rgb.shape[1]:
        raise ValueError("alignment and lookahead must describe supplied context")
    expected = (rgb.shape[0], context_rgb.shape[1], 1, *rgb.shape[-2:])
    if context_valid is None:
        context_valid = torch.ones(expected, device=rgb.device, dtype=torch.bool)
    if context_valid.shape != expected or context_valid.dtype is not torch.bool or context_valid.device != rgb.device:
        raise ValueError("context_valid must be matching bool [B,N,1,H,W]")
    if context_confidence is None:
        context_confidence = context_valid.to(rgb.dtype)
    if context_confidence.shape != expected or context_confidence.device != rgb.device or context_confidence.dtype != rgb.dtype:
        raise ValueError("context_confidence must be matching [B,N,1,H,W] tensor")
    finite_context = torch.isfinite(context_rgb).all(2, keepdim=True)
    confidence_valid = torch.isfinite(context_confidence) & (context_confidence >= 0) & (context_confidence <= 1)
    effective_valid = context_valid & finite_context & confidence_valid
    weight = torch.where(effective_valid, context_confidence, torch.zeros_like(context_confidence))
    support = effective_valid.sum(1, dtype=torch.int64)
    total_weight = weight.sum(1)
    context_mask = effective_valid.expand(-1, -1, context_rgb.shape[2], -1, -1)
    safe_context = torch.where(context_mask, context_rgb, torch.zeros_like(context_rgb))
    restored = (rgb + (safe_context * weight).sum(1)) / (1 + total_weight)
    alignment_code = 1 if alignment == "rgbd_reprojected" else 0
    return {"rgb": restored, "confidence": (total_weight / (1 + total_weight)).clamp(0, 1), "support_count": support, "alignment_code": torch.full((rgb.shape[0],), alignment_code, device=rgb.device, dtype=torch.int64), "lookahead": torch.full((rgb.shape[0],), lookahead, device=rgb.device, dtype=torch.int64)}
