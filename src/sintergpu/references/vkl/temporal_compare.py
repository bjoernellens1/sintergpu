# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Temporal RGB/depth comparison evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114


def _gradient(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = F.pad(gray[..., 1:] - gray[..., :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1), mode="replicate")
    return dx, dy, torch.sqrt(dx.square() + dy.square() + 1e-12)


def _mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).flatten(-3).sum(-1) / weight.flatten(-3).sum(-1).clamp_min(1e-6)


def temporal_frame_compare(
    target_rgb: torch.Tensor,
    aligned_rgb: torch.Tensor,
    *,
    alignment_confidence: torch.Tensor | None = None,
    aligned_valid: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    aligned_depth: torch.Tensor | None = None,
    target_depth_valid: torch.Tensor | None = None,
    aligned_depth_valid: torch.Tensor | None = None,
    motion_direction: torch.Tensor | None = None,
    tile_size: int = 16,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Compare aligned observations and return explicit temporal evidence.

    ``aligned_rgb`` is already in target image coordinates with shape
    ``[B,N,3,H,W]``. All returned pairwise vectors have shape ``[B,N]``;
    diagnostic maps retain ``[B,N,1,H,W]`` and compact tile evidence is
    ``[B,8,Ty,Tx]``.
    """
    if target_rgb.ndim != 4 or target_rgb.shape[1] != 3 or not target_rgb.is_floating_point():
        raise ValueError("target_rgb must be floating [B,3,H,W]")
    if aligned_rgb.ndim != 5 or aligned_rgb.shape[0] != target_rgb.shape[0] or aligned_rgb.shape[2:] != target_rgb.shape[1:]:
        raise ValueError("aligned_rgb must be [B,N,3,H,W] matching target_rgb")
    b, n, _, h, w = aligned_rgb.shape
    if n == 0 or tile_size < 1:
        raise ValueError("aligned_rgb must contain neighbors and tile_size must be positive")
    if aligned_rgb.device != target_rgb.device or aligned_rgb.dtype != target_rgb.dtype:
        raise ValueError("RGB tensors must share device and dtype")
    shape = (b, n, 1, h, w)
    if alignment_confidence is None:
        confidence = torch.ones(shape, dtype=target_rgb.dtype, device=target_rgb.device)
    else:
        confidence = alignment_confidence
        if confidence.shape != shape or confidence.device != target_rgb.device or confidence.dtype != target_rgb.dtype:
            raise ValueError("alignment_confidence must be floating [B,N,1,H,W]")
        if validate and (
            not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()
        ):
            raise ValueError("alignment_confidence must be finite and in [0,1]")
    valid = confidence > 0 if aligned_valid is None else aligned_valid
    if valid.shape != shape or valid.dtype != torch.bool or valid.device != target_rgb.device:
        raise ValueError("aligned_valid must be bool [B,N,1,H,W]")
    target_gray = _gray(target_rgb)
    aligned_gray = _gray(aligned_rgb.flatten(0, 1)).reshape(b, n, 1, h, w)
    target_dx, target_dy, target_grad = _gradient(target_gray)
    source_dx, source_dy, source_grad = _gradient(aligned_gray.flatten(0, 1))
    source_dx = source_dx.reshape(b, n, 1, h, w)
    source_dy = source_dy.reshape(b, n, 1, h, w)
    source_grad = source_grad.reshape(b, n, 1, h, w)
    target_gray = target_gray[:, None]
    target_grad = target_grad[:, None]
    target_dx = target_dx[:, None]
    target_dy = target_dy[:, None]
    weight = confidence * valid.to(confidence.dtype)
    valid_fraction = weight.flatten(-3).mean(-1)

    rgb_error = (aligned_gray - target_gray).abs()
    rgb_consistency_map = torch.exp(-rgb_error / 0.1)
    rgb_consistency = _mean(rgb_consistency_map, weight)
    target_low = F.avg_pool2d(target_gray[:, 0], 7, stride=1, padding=3, count_include_pad=False)[:, None]
    source_low = F.avg_pool2d(aligned_gray.flatten(0, 1), 7, stride=1, padding=3, count_include_pad=False).reshape(b, n, 1, h, w)
    low_rgb_consistency_map = torch.exp(-(source_low - target_low).abs() / 0.1)
    low_rgb_consistency = _mean(low_rgb_consistency_map, weight)
    target_mean = _mean(target_low.expand(-1, n, -1, -1, -1), weight)
    source_mean = _mean(source_low, weight)
    target_centered = target_low - target_mean[..., None, None, None]
    source_centered = source_low - source_mean[..., None, None, None]
    target_std = torch.sqrt(_mean(target_centered.square(), weight) + 1e-6)
    source_std = torch.sqrt(_mean(source_centered.square(), weight) + 1e-6)
    exposure_quality = torch.exp(-(target_mean - source_mean).abs() / 0.1) * torch.exp(
        -(torch.log(target_std / source_std.clamp_min(1e-3))).abs()
    )

    relative_sharpness_map = source_grad / target_grad.clamp_min(1e-3)
    relative_sharpness = _mean(relative_sharpness_map.clamp_max(4) / 4, weight).clamp(0, 1)
    neighbor_high_frequency = source_grad > 0.02
    motion_blur_map = ((source_grad - target_grad).relu() / source_grad.clamp_min(1e-3)) * neighbor_high_frequency
    motion_blur_evidence = _mean(motion_blur_map, weight).clamp(0, 1)
    if motion_direction is None:
        directional_blur_map = torch.zeros_like(motion_blur_map)
    else:
        if motion_direction.shape != (b, n, 2, h, w) or motion_direction.device != target_rgb.device or motion_direction.dtype != target_rgb.dtype:
            raise ValueError("motion_direction must be floating [B,N,2,H,W]")
        direction_grad = (source_dx * motion_direction[:, :, 0:1] + source_dy * motion_direction[:, :, 1:2]).abs()
        directional_blur_map = ((direction_grad - target_grad).relu() / direction_grad.clamp_min(1e-3)) * (direction_grad > 0.02)
    directional_blur_support = _mean(directional_blur_map, weight).clamp(0, 1)

    depth_consistency_map = torch.ones_like(weight)
    depth_consistency = torch.ones((b, n), dtype=target_rgb.dtype, device=target_rgb.device)
    if (target_depth is None) != (aligned_depth is None):
        raise ValueError("target_depth and aligned_depth must be supplied together")
    if target_depth is not None:
        if target_depth.shape != (b, 1, h, w) or aligned_depth.shape != (b, n, 1, h, w):
            raise ValueError("depth tensors must match RGB spatial and batch shapes")
        if target_depth.device != target_rgb.device or aligned_depth.device != target_rgb.device or target_depth.dtype != target_rgb.dtype or aligned_depth.dtype != target_rgb.dtype:
            raise ValueError("depth tensors must share RGB device and dtype")
        target_depth_valid = torch.isfinite(target_depth) & (target_depth > 0) if target_depth_valid is None else target_depth_valid
        aligned_depth_valid = torch.isfinite(aligned_depth) & (aligned_depth > 0) if aligned_depth_valid is None else aligned_depth_valid
        if target_depth_valid.shape != (b, 1, h, w) or aligned_depth_valid.shape != shape or target_depth_valid.dtype != torch.bool or aligned_depth_valid.dtype != torch.bool:
            raise ValueError("depth valid masks have incorrect shape or dtype")
        depth_weight = weight * target_depth_valid[:, None].to(weight.dtype) * aligned_depth_valid.to(weight.dtype)
        relative_depth_error = (aligned_depth - target_depth[:, None]).abs() / target_depth[:, None].clamp_min(1e-3)
        depth_consistency_map = torch.exp(-relative_depth_error / 0.05) * (depth_weight > 0).to(weight.dtype)
        depth_consistency = _mean(depth_consistency_map, depth_weight)
        weight = depth_weight

    restoration_map = (motion_blur_map * low_rgb_consistency_map).clamp(0, 1)
    restoration_support = _mean(restoration_map, weight).clamp(0, 1)
    pairwise_quality = (low_rgb_consistency * exposure_quality * depth_consistency * (0.5 + 0.5 * relative_sharpness)).clamp(0, 1)
    pairwise_equivalence = (valid_fraction >= 0.5) & (low_rgb_consistency >= 0.5) & (depth_consistency >= 0.8)
    target_quality = (1 - torch.exp(-_mean(target_grad, torch.ones_like(target_grad).expand(-1, n, -1, -1, -1)).mean(1))).clamp(0, 1)
    best_neighbor_quality = pairwise_quality.amax(1)
    quality_advantage = (best_neighbor_quality - target_quality).clamp(0, 1)
    replaceability = pairwise_equivalence.to(target_rgb.dtype) * quality_advantage[:, None]
    target_sharpness = target_grad.mean((-1, -2, -3))
    features = torch.cat(
        (relative_sharpness_map.clamp(0, 1), exposure_quality[..., None, None, None].expand(-1, -1, 1, h, w),
         rgb_consistency[..., None, None, None].expand(-1, -1, 1, h, w), depth_consistency_map,
         motion_blur_map, directional_blur_map, restoration_map, valid.to(target_rgb.dtype)), dim=2
    )
    feature_weight = weight
    pooled = F.adaptive_avg_pool2d((features * feature_weight).sum(1) / feature_weight.sum(1).clamp_min(1e-6), (max(1, h // tile_size), max(1, w // tile_size)))
    frame_features = torch.stack((relative_sharpness.mean(1), exposure_quality.mean(1), rgb_consistency.mean(1), depth_consistency.mean(1), best_neighbor_quality, quality_advantage, replaceability.amax(1), motion_blur_evidence.mean(1), directional_blur_support.mean(1), restoration_support.mean(1)), dim=1)
    return {
        "relative_sharpness": relative_sharpness,
        "relative_sharpness_map": relative_sharpness_map,
        "relative_exposure_quality": exposure_quality,
        "temporal_rgb_consistency": rgb_consistency,
        "low_frequency_rgb_consistency": low_rgb_consistency,
        "temporal_depth_consistency": depth_consistency,
        "best_neighbor_quality": best_neighbor_quality,
        "quality_advantage": quality_advantage,
        "replaceability": replaceability,
        "motion_blur_evidence": motion_blur_evidence,
        "directional_blur_support": directional_blur_support,
        "restoration_support": restoration_support,
        "restoration_support_map": restoration_map,
        "pairwise_quality": pairwise_quality,
        "pairwise_equivalence": pairwise_equivalence,
        "temporal_pixel_features": features,
        "temporal_tile_features": pooled,
        "temporal_frame_features": frame_features,
        "target_quality": target_quality,
        "target_sharpness": target_sharpness,
    }
