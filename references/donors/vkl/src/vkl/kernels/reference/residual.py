"""Observed-versus-predicted RGB-D residual evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def residual_evidence(
    observed_rgb: torch.Tensor,
    predicted_rgb: torch.Tensor,
    *,
    observed_depth: torch.Tensor | None = None,
    predicted_depth: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    tile_rows: int = 12,
    tile_cols: int = 16,
) -> dict[str, torch.Tensor]:
    """Return RGB-D residual maps plus bounded frame and tile aggregates."""
    if observed_rgb.ndim != 4 or observed_rgb.shape[1] not in {1, 3} or observed_rgb.shape != predicted_rgb.shape or not observed_rgb.is_floating_point() or observed_rgb.device != predicted_rgb.device or observed_rgb.dtype != predicted_rgb.dtype:
        raise ValueError("observed/predicted_rgb must be matching floating [B,1|3,H,W]")
    has_depth = observed_depth is not None or predicted_depth is not None
    if has_depth and (observed_depth is None or predicted_depth is None or observed_depth.shape != predicted_depth.shape or observed_depth.shape != (observed_rgb.shape[0], 1, *observed_rgb.shape[-2:]) or observed_depth.device != observed_rgb.device or observed_depth.dtype != observed_rgb.dtype):
        raise ValueError("depth inputs must be matching [B,1,H,W] RGB-device tensors")
    if valid is None:
        valid = torch.ones((observed_rgb.shape[0], 1, *observed_rgb.shape[-2:]), device=observed_rgb.device, dtype=torch.bool)
    if valid.shape != (observed_rgb.shape[0], 1, *observed_rgb.shape[-2:]) or valid.dtype is not torch.bool or valid.device != observed_rgb.device or tile_rows < 1 or tile_cols < 1:
        raise ValueError("valid must be matching bool [B,1,H,W] and tile shape positive")
    rgb_residual = (observed_rgb - predicted_rgb).abs().mean(1, keepdim=True)
    if has_depth:
        depth_residual = (observed_depth - predicted_depth).abs()
        depth_residual = torch.where(valid & torch.isfinite(depth_residual), depth_residual, torch.zeros_like(depth_residual))
        residual = 0.5 * (rgb_residual + depth_residual)
    else:
        residual = rgb_residual
    tile_residual = F.adaptive_avg_pool2d(residual, (min(tile_rows, residual.shape[-2]), min(tile_cols, residual.shape[-1])))[:, 0]
    out = {"rgb_residual": rgb_residual, "residual": residual, "frame_residual": residual.mean((-2, -1))[:, 0], "tile_residual": tile_residual}
    if has_depth:
        out["depth_residual"] = depth_residual
    return out
