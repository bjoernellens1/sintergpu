from __future__ import annotations

import torch
import torch.nn.functional as F

POINT_CONFIDENCE_CHANNELS = (
    "selected",
    "valid_anchor",
    "depth_confidence",
    "parallax_support",
    "photometric_support",
)
TILE_FEATURES = (
    "selected_fraction",
    "anchor_fraction",
    "anchor_yield",
    "high_confidence_yield",
    "confidence_mean",
    "parallax_mean_px",
    "photometric_support_mean",
    "gradient_mean",
)
FRAME_FEATURES = (
    "selected_fraction",
    "anchor_fraction",
    "anchor_yield",
    "high_confidence_yield",
    "spatial_coverage",
    "high_confidence_coverage",
    "confidence_mean",
    "parallax_mean_px",
    "photometric_support_mean",
    "gradient_mean",
)


def _tile_mean(value: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    m = mask.to(value.dtype)
    return F.adaptive_avg_pool2d(value * m, size) / F.adaptive_avg_pool2d(m, size).clamp_min(1e-6)


def _frame_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(value.dtype)
    return (value * m).sum((-2, -1))[:, 0] / m.sum((-2, -1))[:, 0].clamp_min(1)


def keypoint_evidence(
    *,
    selected: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    parallax_px: torch.Tensor,
    photometric_error: torch.Tensor,
    gradient: torch.Tensor,
    tile_rows: int = 12,
    tile_cols: int = 16,
    high_confidence: float = 0.5,
    parallax_scale_px: float = 2.0,
    photometric_scale: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Reduce keypoint/depth-anchor outputs into point, tile, and frame evidence."""
    shape = confidence.shape
    if confidence.ndim != 4 or confidence.shape[1] != 1 or not confidence.is_floating_point():
        raise ValueError("confidence must be floating [B,1,H,W]")
    if selected.shape != shape or valid.shape != shape or selected.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError("selected and valid must be bool tensors matching confidence")
    if selected.device != confidence.device or valid.device != confidence.device:
        raise ValueError("selected, valid, and confidence must share device")
    for name, value in (
        ("parallax_px", parallax_px),
        ("photometric_error", photometric_error),
        ("gradient", gradient),
    ):
        if value.shape != shape or value.device != confidence.device or value.dtype != confidence.dtype:
            raise ValueError(f"{name} must match confidence shape, device, and dtype")
    if (
        not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
        or not torch.isfinite(parallax_px).all()
        or (parallax_px < 0).any()
        or not torch.isfinite(photometric_error).all()
        or (photometric_error < 0).any()
        or not torch.isfinite(gradient).all()
        or (gradient < 0).any()
    ):
        raise ValueError("evidence tensors must be finite with non-negative costs/strengths and confidence in [0,1]")
    if tile_rows < 1 or tile_cols < 1:
        raise ValueError("tile grid must be positive")
    if not 0 <= high_confidence <= 1 or parallax_scale_px <= 0 or photometric_scale <= 0:
        raise ValueError("confidence threshold and evidence scales must be valid")

    valid = valid & selected
    high = valid & (confidence >= high_confidence)
    sf = selected.to(confidence.dtype)
    vf = valid.to(confidence.dtype)
    hf = high.to(confidence.dtype)
    parallax_support = (parallax_px / parallax_scale_px).clamp(0, 1)
    photo_support = torch.exp(-photometric_error / photometric_scale)
    point_confidence = torch.cat(
        (sf, vf, confidence * vf, parallax_support * vf, photo_support * vf), dim=1
    )

    size = (tile_rows, tile_cols)
    tile_selected = F.adaptive_avg_pool2d(sf, size)
    tile_anchor = F.adaptive_avg_pool2d(vf, size)
    tile_high = F.adaptive_avg_pool2d(hf, size)
    tile_features = torch.cat(
        (
            tile_selected,
            tile_anchor,
            tile_anchor / tile_selected.clamp_min(1e-6),
            tile_high / tile_selected.clamp_min(1e-6),
            _tile_mean(confidence, valid, size),
            _tile_mean(parallax_px, valid, size),
            _tile_mean(photo_support, valid, size),
            _tile_mean(gradient, selected, size),
        ),
        dim=1,
    )

    selected_count = sf.sum((-2, -1))[:, 0]
    anchor_count = vf.sum((-2, -1))[:, 0]
    high_count = hf.sum((-2, -1))[:, 0]
    frame_features = torch.stack(
        (
            sf.mean((-2, -1))[:, 0],
            vf.mean((-2, -1))[:, 0],
            anchor_count / selected_count.clamp_min(1),
            high_count / selected_count.clamp_min(1),
            (tile_anchor > 0).to(confidence.dtype).mean((-2, -1))[:, 0],
            (tile_high > 0).to(confidence.dtype).mean((-2, -1))[:, 0],
            _frame_mean(confidence, valid),
            _frame_mean(parallax_px, valid),
            _frame_mean(photo_support, valid),
            _frame_mean(gradient, selected),
        ),
        dim=1,
    )
    counts = torch.stack((selected_count, anchor_count, high_count), dim=1)
    return {
        "point_confidence": point_confidence,
        "tile_features": tile_features,
        "frame_features": frame_features,
        "counts": counts,
    }
