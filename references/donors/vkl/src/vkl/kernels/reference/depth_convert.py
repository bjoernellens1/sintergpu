"""Small, explicit conversions between depth representations."""

from __future__ import annotations

import torch


def depth_convert(
    *,
    values: torch.Tensor,
    representation: str,
    valid: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    scale: float | None = None,
    focal_length: float | None = None,
    baseline: float | None = None,
    metric_status: str | None = None,
) -> dict[str, torch.Tensor | str]:
    """Convert a floating depth representation to canonical depth tensors.

    ``scale`` converts source units to metres for ``DEPTH`` and reciprocal
    metres for ``INVERSE_DEPTH``. ``baseline`` is in metres.
    """
    if not isinstance(values, torch.Tensor) or not values.is_floating_point() or values.numel() == 0:
        raise TypeError("values must be a non-empty floating tensor")
    rep = representation.upper()
    if rep not in {"DEPTH", "INVERSE_DEPTH", "DISPARITY"}:
        raise ValueError("representation must be DEPTH, INVERSE_DEPTH, or DISPARITY")
    explicit_status = metric_status is not None
    status = "RELATIVE" if metric_status is None else metric_status.upper()
    if status not in {"RELATIVE", "METRIC_ESTIMATED", "METRIC_ANCHORED"}:
        raise ValueError("metric_status must be RELATIVE, METRIC_ESTIMATED, or METRIC_ANCHORED")
    if scale is not None and (not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(scale)) or scale <= 0):
        raise ValueError("scale must be a finite positive scalar")
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    if not isinstance(valid, torch.Tensor) or valid.shape != values.shape or valid.dtype is not torch.bool or valid.device != values.device:
        raise ValueError("valid must be a bool tensor matching values on the same device")
    if confidence is None:
        confidence = valid.to(values.dtype)
    if (
        not isinstance(confidence, torch.Tensor)
        or confidence.shape != values.shape
        or confidence.device != values.device
        or not confidence.is_floating_point()
        or not torch.isfinite(confidence).all()
        or not ((confidence >= 0) & (confidence <= 1)).all()
    ):
        raise ValueError("confidence must be a finite floating tensor in [0, 1] matching values")
    if rep == "DISPARITY":
        if focal_length is None or baseline is None or focal_length <= 0 or baseline <= 0:
            raise ValueError("disparity requires positive focal_length and baseline")
        depth = (float(focal_length) * float(baseline)) / values
        status = "METRIC_ANCHORED"
    elif rep == "INVERSE_DEPTH":
        if status == "METRIC_ANCHORED" and scale is None:
            raise ValueError("metric anchored inverse depth requires an explicit scale")
        inverse = values if scale is None else values * float(scale)
        depth = inverse.reciprocal()
    else:
        if status == "METRIC_ANCHORED" and scale is None:
            raise ValueError("metric anchored depth requires an explicit scale")
        depth = values if scale is None else values * float(scale)
    if scale is not None and not explicit_status:
        status = "METRIC_ANCHORED"
    usable = valid & torch.isfinite(values) & (values > 0) & torch.isfinite(depth) & (depth > 0)
    depth = torch.where(usable, depth, torch.zeros_like(depth))
    inverse = torch.where(usable, depth.reciprocal(), torch.zeros_like(depth))
    return {
        "depth": depth,
        "inverse_depth": inverse,
        "valid": usable,
        "confidence": torch.where(usable, confidence.to(values.dtype), torch.zeros_like(depth)),
        "representation": "DEPTH",
        "metric_status": status,
    }
