# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Conservative validity-aware depth resampling and RGB-guided propagation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .depth_pyramid import depth_pyramid


def _check(depth: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor) -> None:
    if (
        depth.ndim != 4
        or depth.shape[1] != 1
        or not depth.is_floating_point()
        or valid.shape != depth.shape
        or valid.dtype is not torch.bool
        or confidence.shape != depth.shape
        or not confidence.is_floating_point()
        or any(t.device != depth.device for t in (valid, confidence))
        or not torch.isfinite(confidence).all()
        or not ((confidence >= 0) & (confidence <= 1)).all()
    ):
        raise ValueError("depth/confidence must be floating [B,1,H,W], and valid must be matching bool")


def depth_downsample(
    depth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    *,
    scale: int,
) -> dict[str, torch.Tensor]:
    """Downsample by 2, 4, or 8 with the validity-aware depth pyramid rule."""
    _check(depth, valid, confidence)
    if scale not in {2, 4, 8}:
        raise ValueError("scale must be one of 2, 4, or 8")
    level = depth_pyramid(depth=depth, valid=valid, confidence=confidence).get(f"/{scale}")
    if level is None:
        raise ValueError("scale exceeds available depth pyramid levels")
    return {
        **level,
        "direct_valid": level["valid"],
        "propagated_valid": torch.zeros_like(level["valid"]),
        "provenance": level["valid"].to(torch.uint8),
    }


def depth_upsample(
    depth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    *,
    size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Nearest-neighbour depth lift; every lifted value is labelled propagated, never direct."""
    _check(depth, valid, confidence)
    if len(size) != 2 or min(size) < 1:
        raise ValueError("size must contain two positive dimensions")
    usable = valid & torch.isfinite(depth) & (depth > 0)
    output_depth = F.interpolate(torch.where(usable, depth, torch.zeros_like(depth)), size=size, mode="nearest")
    output_valid = F.interpolate(usable.float(), size=size, mode="nearest").bool()
    output_confidence = F.interpolate(torch.where(usable, confidence, torch.zeros_like(confidence)), size=size, mode="nearest")
    output_depth = torch.where(output_valid, output_depth, torch.zeros_like(output_depth))
    output_confidence = torch.where(output_valid, output_confidence, torch.zeros_like(output_confidence))
    propagated = output_valid
    return {
        "depth": output_depth,
        "valid": output_valid,
        "confidence": output_confidence,
        "direct_valid": torch.zeros_like(output_valid),
        "propagated_valid": propagated,
        "provenance": propagated.to(torch.uint8) * 2,
    }


def _propagate_once(
    depth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    direct_valid: torch.Tensor,
    rgb: torch.Tensor,
    *,
    max_rgb_difference: float,
    max_relative_spread: float,
    min_support: int,
) -> dict[str, torch.Tensor]:
    batch, _, height, width = depth.shape
    rgb_patch = F.unfold(rgb, 3, padding=1).reshape(batch, rgb.shape[1], 9, height, width)
    rgb_difference = (rgb_patch - rgb[:, :, None]).abs().amax(1)
    depth_patch = F.unfold(depth, 3, padding=1).reshape(batch, 9, height, width)
    valid_patch = F.unfold(valid.float(), 3, padding=1).reshape(batch, 9, height, width).bool()
    neighbours = valid_patch & (rgb_difference <= max_rgb_difference)
    support = neighbours.sum(1, keepdim=True)
    values = depth_patch.masked_fill(~neighbours, torch.inf)
    minimum = values.min(1, keepdim=True).values
    maximum = depth_patch.masked_fill(~neighbours, -torch.inf).max(1, keepdim=True).values
    median = values.sort(1).values.gather(1, (support.clamp_min(1) - 1) // 2)
    relative_spread = (maximum - minimum) / median.clamp_min(torch.finfo(depth.dtype).eps)
    fill = ~valid & (support >= min_support) & (relative_spread <= max_relative_spread)
    confidence_patch = F.unfold(confidence, 3, padding=1).reshape(batch, 9, height, width)
    fill_confidence = confidence_patch.masked_fill(~neighbours, 0).sum(1, keepdim=True) / support.clamp_min(1)
    fill_confidence = fill_confidence * (support / 9)
    output_depth = torch.where(fill, median, depth)
    output_valid = valid | fill
    output_confidence = torch.where(fill, fill_confidence, confidence)
    propagated = output_valid & ~direct_valid
    return {
        "depth": torch.where(output_valid, output_depth, torch.zeros_like(output_depth)),
        "valid": output_valid,
        "confidence": torch.where(output_valid, output_confidence, torch.zeros_like(output_confidence)),
        "direct_valid": direct_valid,
        "propagated_valid": propagated,
        "provenance": direct_valid.to(torch.uint8) + propagated.to(torch.uint8) * 2,
        "support": support,
    }


def depth_densify(
    depth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    rgb: torch.Tensor,
    *,
    direct_valid: torch.Tensor | None = None,
    max_rgb_difference: float = 0.15,
    max_relative_spread: float = 0.10,
    min_support: int = 2,
    steps: int = 1,
) -> dict[str, torch.Tensor]:
    """Fill only RGB- and local-depth-consistent holes, retaining direct/propgated provenance."""
    _check(depth, valid, confidence)
    if (
        rgb.ndim != 4
        or rgb.shape[0] != depth.shape[0]
        or rgb.shape[1] not in {1, 3}
        or rgb.shape[-2:] != depth.shape[-2:]
        or rgb.device != depth.device
        or not rgb.is_floating_point()
        or max_rgb_difference < 0
        or max_relative_spread < 0
        or min_support < 1
        or steps < 1
    ):
        raise ValueError("RGB must align with depth and propagation parameters must be non-negative")
    usable = valid & torch.isfinite(depth) & (depth > 0)
    if direct_valid is None:
        direct_valid = usable
    if direct_valid.shape != depth.shape or direct_valid.dtype is not torch.bool or direct_valid.device != depth.device:
        raise ValueError("direct_valid must be a matching bool depth mask")
    direct_valid = direct_valid & usable
    state = {
        "depth": torch.where(usable, depth, torch.zeros_like(depth)),
        "valid": usable,
        "confidence": torch.where(usable, confidence, torch.zeros_like(confidence)),
    }
    for _ in range(steps):
        state = _propagate_once(
            depth=state["depth"],
            valid=state["valid"],
            confidence=state["confidence"],
            direct_valid=direct_valid,
            rgb=rgb,
            max_rgb_difference=max_rgb_difference,
            max_relative_spread=max_relative_spread,
            min_support=min_support,
        )
    return state
