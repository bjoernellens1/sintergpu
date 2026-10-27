# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Bounded modality-aware tracker preparation reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .depth_evidence import depth_evidence
from .depth_pyramid import depth_pyramid
from .evidence import rgbd_evidence
from .geometry import depth_geometry
from .pyramid import resolution_pyramid
from .rescale import canonical_rescale

_RGB_LEVEL = {"/1": "native", "/2": "half", "/4": "quarter", "/8": "eighth"}


def _resize_valid(valid: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(valid.to(torch.float32), size=shape, mode="nearest") > 0.5


def _scale_intrinsics(intrinsics: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    sx, sy = target_w / source_w, target_h / source_h
    result = intrinsics.clone()
    result[:, 0, 0] *= sx
    result[:, 1, 1] *= sy
    result[:, 0, 2] = (result[:, 0, 2] + 0.5) * sx - 0.5
    result[:, 1, 2] = (result[:, 1, 2] + 0.5) * sy - 0.5
    return result


def tracking_prepare(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
    pose_c2w: torch.Tensor | None = None,
    rgb_valid: torch.Tensor | None = None,
    depth_valid: torch.Tensor | None = None,
    depth_measurement_valid: torch.Tensor | None = None,
    depth_confidence: torch.Tensor | None = None,
    scales: tuple[str, ...] = ("/4", "/2", "/1"),
    target_pixels: int = 307_200,
    min_width: int = 320,
    min_height: int = 240,
    allow_upscale: bool = False,
) -> dict[str, object]:
    """Prepare a fixed subset of canonical RGB-D pyramid levels for tracking.

    Inputs must already share an RGB image grid; camera rectification and depth
    registration remain preceding conditioning operations. Geometry is emitted
    only when metric depth and intrinsics are present.
    """
    if (
        not isinstance(scales, tuple)
        or not scales
        or len(set(scales)) != len(scales)
        or any(scale not in _RGB_LEVEL for scale in scales)
    ):
        raise ValueError("scales must be unique bounded pyramid levels")
    if depth is not None and intrinsics is None:
        raise ValueError("depth tracking preparation requires intrinsics")
    if depth_confidence is not None and (
        depth is None
        or depth_confidence.shape != depth.shape
        or depth_confidence.device != depth.device
        or depth_confidence.dtype != depth.dtype
        or not torch.isfinite(depth_confidence).all()
        or not ((depth_confidence >= 0) & (depth_confidence <= 1)).all()
    ):
        raise ValueError("depth_confidence must be finite in [0,1] and match depth")

    canonical = canonical_rescale(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        rgb_valid=rgb_valid,
        depth_valid=depth_valid,
        depth_measurement_valid=depth_measurement_valid,
        target_pixels=target_pixels,
        min_width=min_width,
        min_height=min_height,
        allow_upscale=allow_upscale,
    )
    canonical_rgb = canonical["rgb"]
    rgb_levels = resolution_pyramid(rgb=canonical_rgb)
    canonical_hw = tuple(canonical_rgb.shape[-2:])
    levels: dict[str, dict[str, torch.Tensor]] = {}
    evidence: dict[str, object] = {
        "rgbd": rgbd_evidence(
            rgb=canonical_rgb,
            depth=canonical.get("depth"),
            rgb_valid=canonical["rgb_valid"],
            depth_valid=canonical.get("depth_valid"),
            depth_measurement_valid=canonical.get("depth_measurement_valid"),
        )
    }

    if depth is not None:
        confidence = depth_confidence
        if confidence is None:
            confidence = torch.ones_like(depth)
        if confidence.shape[-2:] != canonical_hw:
            confidence = F.interpolate(confidence, size=canonical_hw, mode="nearest")
        measurement = canonical["depth_measurement_valid"]
        confidence = torch.where(measurement, confidence, torch.zeros_like(confidence))
        depth_levels = depth_pyramid(
            depth=canonical["depth"], valid=measurement, confidence=confidence
        )
        evidence["depth"] = depth_evidence(
            canonical["depth"], measurement, confidence
        )
    else:
        depth_levels = None

    for scale in scales:
        rgb_level = rgb_levels[_RGB_LEVEL[scale]]
        level: dict[str, torch.Tensor] = {
            "rgb": rgb_level,
            "rgb_valid": _resize_valid(canonical["rgb_valid"], tuple(rgb_level.shape[-2:])),
        }
        if intrinsics is not None:
            level["intrinsics"] = _scale_intrinsics(
                canonical["intrinsics"], canonical_hw, tuple(rgb_level.shape[-2:])
            )
        if depth_levels is not None:
            depth_level = depth_levels[scale]
            level.update(
                {
                    "depth": depth_level["depth"],
                    "depth_valid": depth_level["valid"],
                    "depth_confidence": depth_level["confidence"],
                }
            )
            geometry = depth_geometry(
                depth_level["depth"],
                level["intrinsics"],
                pose_c2w=pose_c2w,
                confidence=depth_level["confidence"],
            )
            level.update(geometry)
        levels[scale] = level
    return {"levels": levels, "evidence": evidence}
