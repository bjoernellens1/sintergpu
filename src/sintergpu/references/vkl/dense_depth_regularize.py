# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import torch
import torch.nn.functional as F


def regularize_dense_depth(
    depth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    rgb: torch.Tensor,
    *,
    max_rgb_difference: float = 0.15,
    max_relative_deviation: float = 0.15,
    min_support: int = 4,
) -> dict[str, torch.Tensor]:
    """Remove isolated plane-sweep labels without crossing strong RGB edges."""
    if (
        depth.ndim != 4
        or depth.shape[1] != 1
        or valid.shape != depth.shape
        or confidence.shape != depth.shape
        or rgb.ndim != 4
        or rgb.shape[0] != depth.shape[0]
        or rgb.shape[-2:] != depth.shape[-2:]
        or rgb.shape[1] not in {1, 3}
    ):
        raise ValueError("depth/valid/confidence must be [B,1,H,W] and RGB must align")
    if (
        not depth.is_floating_point()
        or valid.dtype != torch.bool
        or valid.device != depth.device
        or not confidence.is_floating_point()
        or confidence.device != depth.device
        or confidence.dtype != depth.dtype
        or not rgb.is_floating_point()
        or rgb.device != depth.device
        or rgb.dtype != depth.dtype
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
        or max_rgb_difference < 0
        or max_relative_deviation < 0
        or min_support < 1
        or min_support > 9
    ):
        raise ValueError("invalid regularization contract")

    b, _, h, w = depth.shape
    gray = (
        rgb[:, :1]
        if rgb.shape[1] == 1
        else (
            rgb[:, :3]
            * torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)[None, :, None, None]
        ).sum(1, keepdim=True)
    )
    depth_patch = F.unfold(depth, 3, padding=1).reshape(b, 9, h, w)
    valid_patch = F.unfold(valid.to(depth.dtype), 3, padding=1).reshape(b, 9, h, w).bool()
    confidence_patch = F.unfold(confidence, 3, padding=1).reshape(b, 9, h, w)
    gray_patch = F.unfold(gray, 3, padding=1).reshape(b, 9, h, w)
    neighbours = valid_patch & torch.isfinite(depth_patch) & (depth_patch > 0)
    neighbours &= (gray_patch - gray).abs() <= max_rgb_difference
    support = neighbours.sum(1, keepdim=True)

    sorted_depth = depth_patch.masked_fill(~neighbours, torch.inf).sort(1).values
    median = sorted_depth.gather(1, (support.clamp_min(1) - 1) // 2).where(support > 0, depth)
    direct_valid = (
        valid
        & torch.isfinite(depth)
        & (depth > 0)
        & (support >= min_support)
        & ((depth - median).abs() <= max_relative_deviation * depth.clamp_min(torch.finfo(depth.dtype).eps))
    )
    output_valid = (support >= min_support) & torch.isfinite(median) & (median > 0)
    output_depth = torch.where(output_valid, median, torch.zeros_like(depth))

    support_confidence = (
        confidence_patch.masked_fill(~neighbours, 0).sum(1, keepdim=True)
        / support.clamp_min(1)
    )
    support_fraction = support.to(depth.dtype) / 9
    output_confidence = torch.where(
        output_valid,
        support_confidence * support_fraction,
        torch.zeros_like(confidence),
    )
    return {
        "depth": output_depth,
        "valid": output_valid,
        "confidence": output_confidence,
        "direct_valid": direct_valid,
        "support": support,
    }
