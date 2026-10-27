# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Bounded health evidence for a fixed RGB-D tracking update."""

from __future__ import annotations

import torch

TRACK_REASON_SOLVE = 1
TRACK_REASON_PARTIALS = 2
TRACK_REASON_GEOMETRY = 4
TRACK_REASON_PHOTOMETRIC = 8
TRACK_REASON_RESIDUAL = 16
TRACK_REASON_CONDITION = 32


def tracking_health(
    *,
    solve_valid: torch.Tensor,
    partial_count: torch.Tensor,
    condition_proxy: torch.Tensor,
    geometry_inlier_count: torch.Tensor,
    photometric_inlier_count: torch.Tensor,
    geometry_residual_abs_mean: torch.Tensor,
    min_partials: int = 1,
    min_geometry_inliers: int = 1,
    min_photometric_inliers: int = 0,
    max_condition: float = 1e6,
    max_geometry_residual: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Classify fixed solve evidence without owning tracking state or hysteresis."""
    batch = solve_valid.shape[0] if solve_valid.ndim == 1 else 0
    integer_fields = (partial_count, geometry_inlier_count, photometric_inlier_count)
    if (
        solve_valid.shape != (batch,)
        or solve_valid.dtype != torch.bool
        or any(value.shape != (batch,) or value.dtype != torch.int64 or value.device != solve_valid.device for value in integer_fields)
        or condition_proxy.shape != (batch,)
        or not condition_proxy.is_floating_point()
        or condition_proxy.device != solve_valid.device
        or not (~torch.isnan(condition_proxy)).all()
        or (condition_proxy < 0).any()
        or geometry_residual_abs_mean.shape != (batch,)
        or not geometry_residual_abs_mean.is_floating_point()
        or geometry_residual_abs_mean.device != solve_valid.device
        or geometry_residual_abs_mean.dtype != condition_proxy.dtype
        or not torch.isfinite(geometry_residual_abs_mean).all()
        or (geometry_residual_abs_mean < 0).any()
        or min(min_partials, min_geometry_inliers, min_photometric_inliers) < 0
        or max_condition <= 1
        or max_geometry_residual <= 0
    ):
        raise ValueError("tracking evidence must be compatible finite [B] tensors with valid thresholds")
    partial_ok = partial_count >= min_partials
    geometry_ok = geometry_inlier_count >= min_geometry_inliers
    photometric_ok = photometric_inlier_count >= min_photometric_inliers
    residual_ok = geometry_residual_abs_mean <= max_geometry_residual
    condition_ok = condition_proxy <= max_condition
    valid = solve_valid & partial_ok & geometry_ok & photometric_ok & residual_ok & condition_ok
    reason_bits = torch.zeros((batch,), device=solve_valid.device, dtype=torch.int32)
    reason_bits |= (~solve_valid).to(torch.int32) * TRACK_REASON_SOLVE
    reason_bits |= (~partial_ok).to(torch.int32) * TRACK_REASON_PARTIALS
    reason_bits |= (~geometry_ok).to(torch.int32) * TRACK_REASON_GEOMETRY
    reason_bits |= (~photometric_ok).to(torch.int32) * TRACK_REASON_PHOTOMETRIC
    reason_bits |= (~residual_ok).to(torch.int32) * TRACK_REASON_RESIDUAL
    reason_bits |= (~condition_ok).to(torch.int32) * TRACK_REASON_CONDITION
    score = solve_valid.to(condition_proxy.dtype)
    for value, threshold in (
        (partial_count.to(condition_proxy.dtype), min_partials),
        (geometry_inlier_count.to(condition_proxy.dtype), min_geometry_inliers),
        (photometric_inlier_count.to(condition_proxy.dtype), min_photometric_inliers),
    ):
        if threshold:
            score = score * (value / threshold).clamp(0, 1)
    score = score * (1 - geometry_residual_abs_mean / max_geometry_residual).clamp(0, 1)
    score = score * (max_condition / condition_proxy.clamp_min(1)).clamp(0, 1)
    return {"valid": valid, "score": score, "reason_bits": reason_bits}
