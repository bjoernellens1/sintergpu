# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _valid_mask(value: torch.Tensor | None, rgb: torch.Tensor, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    expected = (rgb.shape[0], 1, *rgb.shape[-2:])
    if value.shape != expected or value.dtype != torch.bool or value.device != rgb.device:
        raise ValueError(f"{name} must be bool [B,1,H,W] on RGB device")
    return value


def _domain_reduce(
    feature_map: torch.Tensor,
    domain: torch.Tensor,
    tile_rows: int,
    tile_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = domain.to(feature_map.dtype)
    frame_den = weight.sum((-2, -1)).clamp_min(1)
    frame = feature_map.sum((-2, -1)) / frame_den
    tile_num = F.adaptive_avg_pool2d(feature_map, (tile_rows, tile_cols))
    tile_den = F.adaptive_avg_pool2d(weight, (tile_rows, tile_cols))
    tile = tile_num / tile_den.clamp_min(torch.finfo(feature_map.dtype).eps)
    frame_fraction = weight.mean((-2, -1))[:, 0]
    return frame, tile, frame_fraction, tile_den


def rgbd_evidence(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor | None = None,
    rgb_valid: torch.Tensor | None = None,
    depth_valid: torch.Tensor | None = None,
    depth_measurement_valid: torch.Tensor | None = None,
    tile_rows: int = 12,
    tile_cols: int = 16,
    depth_min: float = 0.1,
    depth_max: float = 8.0,
    white_clip: float = 0.98,
    black_clip: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Reduce RGB-D into validity-aware frame and tile evidence.

    ``rgb_valid`` and ``depth_valid`` describe geometric/conditioning domains.
    ``depth_measurement_valid`` is separate occupancy/provenance: a pixel can be
    inside the depth conditioning domain while containing no registered sensor
    sample. Missing measurements therefore do not have to be encoded only by a
    sentinel depth value.
    """
    if (
        rgb.ndim != 4
        or rgb.shape[1] not in {1, 3}
        or not rgb.is_floating_point()
        or tile_rows < 1
        or tile_cols < 1
    ):
        raise ValueError("rgb must be floating [B,1|3,H,W] with positive tile dimensions")
    if (
        not math.isfinite(depth_min)
        or not math.isfinite(depth_max)
        or not 0 <= depth_min < depth_max
        or not math.isfinite(white_clip)
        or not math.isfinite(black_clip)
        or not 0 <= black_clip <= white_clip <= 1
    ):
        raise ValueError("depth and clip thresholds must be finite and ordered")
    if depth is not None and (
        depth.shape != (rgb.shape[0], 1, *rgb.shape[-2:])
        or depth.device != rgb.device
        or depth.dtype != rgb.dtype
        or not depth.is_floating_point()
    ):
        raise ValueError("depth must be floating [B,1,H,W] on the RGB device/dtype")
    if depth is None and (depth_valid is not None or depth_measurement_valid is not None):
        raise ValueError("depth validity masks require depth")
    rgb_valid = _valid_mask(rgb_valid, rgb, "rgb_valid")
    depth_valid = _valid_mask(depth_valid, rgb, "depth_valid")
    depth_measurement_valid = _valid_mask(
        depth_measurement_valid, rgb, "depth_measurement_valid"
    )

    output_dtype = rgb.dtype
    cpu_low_precision = rgb.device.type == "cpu" and rgb.dtype in {
        torch.float16,
        torch.bfloat16,
    }
    work_rgb = rgb.float() if cpu_low_precision else rgb
    rgb_domain = (
        torch.ones((rgb.shape[0], 1, *rgb.shape[-2:]), dtype=torch.bool, device=rgb.device)
        if rgb_valid is None
        else rgb_valid
    )
    finite_rgb = torch.isfinite(work_rgb).all(1, keepdim=True)
    sample_valid = rgb_domain & finite_rgb
    work_rgb = torch.where(sample_valid, work_rgb, torch.zeros_like(work_rgb))
    rgb3 = work_rgb.expand(-1, 3, -1, -1) if work_rgb.shape[1] == 1 else work_rgb[:, :3]
    gray = 0.299 * rgb3[:, 0:1] + 0.587 * rgb3[:, 1:2] + 0.114 * rgb3[:, 2:3]

    gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    grad = 0.5 * (gx.abs() + gy.abs())
    up = F.pad(gray[..., :-1, :], (0, 0, 1, 0))
    down = F.pad(gray[..., 1:, :], (0, 0, 0, 1))
    left = F.pad(gray[..., :, :-1], (1, 0, 0, 0))
    right = F.pad(gray[..., :, 1:], (0, 1, 0, 0))
    lap = (-4.0 * gray + up + down + left + right).abs()
    invalid_neighborhood = (
        F.max_pool2d((~sample_valid).to(gray.dtype), 3, stride=1, padding=1) > 0
    )
    grad = torch.where(invalid_neighborhood, torch.zeros_like(grad), grad)
    lap = torch.where(invalid_neighborhood, torch.zeros_like(lap), lap)
    gray = torch.where(sample_valid, gray, torch.zeros_like(gray))
    white = sample_valid & (gray >= white_clip)
    black = rgb_domain & ((~finite_rgb) | (gray <= black_clip))
    rgb_maps = torch.cat(
        (gray, gray.square(), grad, lap, white.to(gray.dtype), black.to(gray.dtype)), 1
    )
    frame_rgb, tile_rgb, frame_rgb_fraction, tile_rgb_fraction = _domain_reduce(
        rgb_maps, rgb_domain, tile_rows, tile_cols
    )
    frame_features = frame_rgb
    tile_features = tile_rgb

    rgb_quality = torch.clamp(
        2.0 * frame_rgb[:, 2]
        + 2.0 * frame_rgb[:, 3]
        - frame_rgb[:, 4]
        - frame_rgb[:, 5],
        0.0,
        1.0,
    )
    rgb_tile_score = torch.clamp(
        2.0 * tile_rgb[:, 2]
        + 2.0 * tile_rgb[:, 3]
        - tile_rgb[:, 4]
        - tile_rgb[:, 5],
        0.0,
        1.0,
    )
    quality = rgb_quality * frame_rgb_fraction
    tile_score = rgb_tile_score * tile_rgb_fraction[:, 0]

    result: dict[str, torch.Tensor] = {
        "frame_rgb_valid_fraction": frame_rgb_fraction.to(output_dtype),
        "tile_rgb_valid_fraction": tile_rgb_fraction.to(output_dtype),
    }
    if depth is not None:
        work_depth = depth.float() if cpu_low_precision else depth
        depth_domain = (
            torch.ones_like(work_depth, dtype=torch.bool)
            if depth_valid is None
            else depth_valid
        )
        measurement_domain = (
            torch.ones_like(work_depth, dtype=torch.bool)
            if depth_measurement_valid is None
            else depth_measurement_valid
        )
        measurement_domain = measurement_domain & depth_domain
        sensor_valid = (
            measurement_domain
            & torch.isfinite(work_depth)
            & (work_depth > depth_min)
            & (work_depth < depth_max)
        )
        safe = torch.where(sensor_valid, work_depth, torch.zeros_like(work_depth))
        depth_maps = torch.cat((sensor_valid.to(work_depth.dtype), safe, safe.square()), 1)
        frame_depth, tile_depth, frame_depth_domain_fraction, tile_depth_domain_fraction = (
            _domain_reduce(depth_maps, depth_domain, tile_rows, tile_cols)
        )
        frame_features = torch.cat((frame_rgb, frame_depth), 1)
        tile_features = torch.cat((tile_rgb, tile_depth), 1)
        frame_depth_measurement_fraction = measurement_domain.to(work_depth.dtype).mean(
            (-2, -1)
        )[:, 0]
        tile_depth_measurement_fraction = F.adaptive_avg_pool2d(
            measurement_domain.to(work_depth.dtype), (tile_rows, tile_cols)
        )[:, 0]
        frame_depth_coverage = sensor_valid.to(work_depth.dtype).mean((-2, -1))[:, 0]
        tile_depth_coverage = F.adaptive_avg_pool2d(
            sensor_valid.to(work_depth.dtype), (tile_rows, tile_cols)
        )[:, 0]
        quality = torch.clamp(0.75 * quality + 0.25 * frame_depth_coverage, 0.0, 1.0)
        tile_score = torch.clamp(
            0.75 * tile_score + 0.25 * tile_depth_coverage, 0.0, 1.0
        )
        result.update(
            {
                "frame_depth_domain_fraction": frame_depth_domain_fraction.to(output_dtype),
                "tile_depth_domain_fraction": tile_depth_domain_fraction.to(output_dtype),
                "frame_depth_measurement_fraction": frame_depth_measurement_fraction.to(
                    output_dtype
                ),
                "tile_depth_measurement_fraction": tile_depth_measurement_fraction[
                    :, None
                ].to(output_dtype),
                "frame_depth_coverage": frame_depth_coverage.to(output_dtype),
                "tile_depth_coverage": tile_depth_coverage[:, None].to(output_dtype),
            }
        )

    result.update(
        {
            "frame_features": frame_features.to(output_dtype),
            "tile_features": tile_features.to(output_dtype),
            "frame_score": quality.to(output_dtype),
            "tile_score": tile_score.to(output_dtype),
        }
    )
    return result
