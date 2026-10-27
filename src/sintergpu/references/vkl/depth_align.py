# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import torch


def _check_field(name: str, value: torch.Tensor, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape {tuple(shape)}")
    if value.device != device or value.dtype != dtype:
        raise ValueError(f"{name} must share device and dtype with predicted")


def align_depth(
    predicted: torch.Tensor,
    anchor_depth: torch.Tensor,
    anchor_mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    mode: str = "scale_only",
    predicted_valid: torch.Tensor | None = None,
    predicted_confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | str]:
    """Fit relative depth to metric anchors without modifying the raw prediction."""
    if predicted.ndim != 4 or predicted.shape[1] != 1 or not predicted.is_floating_point():
        raise ValueError("predicted must be floating point [B,1,H,W]")
    _check_field("anchor_depth", anchor_depth, predicted.shape, predicted.device, predicted.dtype)
    if anchor_mask.shape != predicted.shape or anchor_mask.dtype != torch.bool or anchor_mask.device != predicted.device:
        raise ValueError("anchor_mask must be bool and match predicted")
    if confidence is not None:
        _check_field("confidence", confidence, predicted.shape, predicted.device, predicted.dtype)
        if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError("confidence must be finite in [0,1]")
    if predicted_valid is not None and (predicted_valid.shape != predicted.shape or predicted_valid.dtype != torch.bool or predicted_valid.device != predicted.device):
        raise ValueError("predicted_valid must be bool and match predicted")
    if predicted_confidence is not None:
        _check_field("predicted_confidence", predicted_confidence, predicted.shape, predicted.device, predicted.dtype)
        if not torch.isfinite(predicted_confidence).all() or (predicted_confidence < 0).any() or (predicted_confidence > 1).any():
            raise ValueError("predicted_confidence must be finite in [0,1]")
    if mode not in {"scale_only", "scale_shift_depth", "scale_shift_inverse_depth"}:
        raise ValueError("unsupported alignment mode")
    if not torch.isfinite(predicted).all() or not torch.isfinite(anchor_depth).all():
        raise ValueError("depth inputs must be finite")

    usable = anchor_mask & (predicted > 0) & (anchor_depth > 0)
    if predicted_valid is not None:
        usable &= predicted_valid
    weights = torch.ones_like(predicted) if confidence is None else confidence
    weights = torch.where(usable, weights, torch.zeros_like(weights))
    fit_usable = usable & (weights > 0)

    p = predicted
    z = anchor_depth
    if mode == "scale_shift_inverse_depth":
        p_fit, z_fit = 1 / p.clamp_min(torch.finfo(p.dtype).tiny), 1 / z.clamp_min(torch.finfo(z.dtype).tiny)
    else:
        p_fit, z_fit = p, z
    x = p_fit.flatten(1)
    y = z_fit.flatten(1)
    w = weights.flatten(1)
    n = w.sum(1)
    if (n <= 0).any():
        raise ValueError("at least one positive-weight metric anchor is required per batch")
    if mode == "scale_only":
        scale = (w * x * y).sum(1) / (w * x.square()).sum(1).clamp_min(torch.finfo(p.dtype).eps)
        shift = torch.zeros_like(scale)
    else:
        sx = (w * x).sum(1)
        sy = (w * y).sum(1)
        sxx = (w * x.square()).sum(1)
        sxy = (w * x * y).sum(1)
        det = n * sxx - sx.square()
        if (det <= torch.finfo(p.dtype).eps).any():
            raise ValueError("scale-shift alignment requires at least two distinct positive-weight anchors per batch")
        scale = (n * sxy - sx * sy) / det
        shift = (sxx * sy - sx * sxy) / det
    fitted = scale[:, None, None, None] * p_fit + shift[:, None, None, None]
    residual = (fitted - z_fit).abs()
    valid = torch.isfinite(fitted) & (fitted > 0) & (predicted > 0)
    if predicted_valid is not None:
        valid &= predicted_valid
    if mode == "scale_shift_inverse_depth":
        depth = 1 / fitted.clamp_min(torch.finfo(p.dtype).tiny)
    else:
        depth = fitted
    depth = torch.where(valid, depth, torch.zeros_like(depth))

    fit_error = (residual * weights).flatten(1).sum(1) / n
    fit_usable_flat = fit_usable.flatten(1)
    fit_usable_count = fit_usable_flat.sum(1)
    sorted_residual = residual.flatten(1).masked_fill(~fit_usable_flat, torch.inf).sort(1).values
    median_index = ((fit_usable_count - 1) // 2).clamp_min(0)
    median_residual = sorted_residual.gather(1, median_index[:, None]).squeeze(1)
    threshold = (median_residual * 3).clamp_min(torch.finfo(p.dtype).eps)
    inlier = fit_usable & (residual <= threshold[:, None, None, None])
    inlier_fraction = (weights * inlier.to(weights.dtype)).flatten(1).sum(1) / n
    calibration_confidence = (inlier_fraction * (n / predicted[0].numel())).clamp(0, 1)
    dense_confidence = torch.ones_like(predicted) if predicted_confidence is None else predicted_confidence
    out_conf = torch.where(valid, dense_confidence * calibration_confidence[:, None, None, None], torch.zeros_like(predicted))
    return {
        "depth": depth,
        "valid": valid,
        "confidence": out_conf,
        "scale": scale,
        "shift": shift,
        "fit_error": fit_error,
        "inlier_fraction": inlier_fraction,
        "calibration_confidence": calibration_confidence,
        "metric_status": "METRIC_ANCHORED",
    }
