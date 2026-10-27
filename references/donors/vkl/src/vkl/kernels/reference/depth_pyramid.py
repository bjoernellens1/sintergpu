"""Validity-aware bounded depth pyramid."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _level(depth: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
    h, w = depth.shape[-2:]
    padded_depth = F.pad(depth, (0, w % 2, 0, h % 2))
    padded_valid = F.pad(valid, (0, w % 2, 0, h % 2), value=False)
    padded_confidence = F.pad(confidence, (0, w % 2, 0, h % 2))
    shape = (*depth.shape[:2], (h + 1) // 2, 2, (w + 1) // 2, 2)
    blocks = padded_depth.reshape(shape).permute(0, 1, 2, 4, 3, 5).flatten(-2)
    usable = padded_valid.reshape(shape).permute(0, 1, 2, 4, 3, 5).flatten(-2) & torch.isfinite(blocks) & (blocks > 0)
    confidences = padded_confidence.reshape(shape).permute(0, 1, 2, 4, 3, 5).flatten(-2)
    count = usable.sum(dim=-1)
    supported = count > 0
    values = blocks.masked_fill(~usable, torch.inf)
    minimum, index = values.min(dim=-1)
    maximum = values.max(dim=-1).values
    mean = blocks.masked_fill(~usable, 0).sum(dim=-1) / count.clamp_min(1)
    mean_confidence = confidences.masked_fill(~usable, 0).sum(dim=-1) / count.clamp_min(1)
    same_surface = (maximum - minimum) <= torch.finfo(depth.dtype).eps * 32
    selected_confidence = confidences.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    use_mean = supported & same_surface
    return {
        "depth": torch.where(supported, torch.where(use_mean, mean, minimum), torch.zeros_like(mean)),
        "valid": supported,
        "confidence": torch.where(supported, torch.where(use_mean, mean_confidence, selected_confidence), torch.zeros_like(mean)),
    }


def depth_pyramid(
    *, depth: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return native and bounded half/quarter/eighth validity-aware depth levels."""
    if not isinstance(depth, torch.Tensor) or depth.ndim != 4 or depth.shape[1] != 1 or not depth.is_floating_point() or min(depth.shape[-2:]) < 1:
        raise ValueError("depth must be a non-empty floating [B, 1, H, W] tensor")
    if not isinstance(valid, torch.Tensor) or valid.shape != depth.shape or valid.dtype is not torch.bool or valid.device != depth.device:
        raise ValueError("valid must be a bool tensor matching depth on the same device")
    if confidence is None:
        confidence = valid.to(depth.dtype)
    if (
        not isinstance(confidence, torch.Tensor)
        or confidence.shape != depth.shape
        or confidence.device != depth.device
        or not confidence.is_floating_point()
        or not torch.isfinite(confidence).all()
        or not ((confidence >= 0) & (confidence <= 1)).all()
    ):
        raise ValueError("confidence must be a finite floating tensor in [0, 1] matching depth")
    base_valid = valid & torch.isfinite(depth) & (depth > 0)
    base_depth = torch.where(base_valid, depth, torch.zeros_like(depth))
    base_confidence = torch.where(base_valid, confidence.to(depth.dtype), torch.zeros_like(depth))
    levels = {"/1": {"depth": base_depth, "valid": base_valid, "confidence": base_confidence}}
    current = levels["/1"]
    for key in ("/2", "/4", "/8"):
        if current["depth"].shape[-2:] == (1, 1):
            break
        current = _level(current["depth"], current["valid"], current["confidence"])
        levels[key] = current
    return levels
