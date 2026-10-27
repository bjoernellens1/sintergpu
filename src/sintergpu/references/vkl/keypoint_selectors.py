# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Small tensor-only classical keypoint selector baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[1] == 1:
        return rgb
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _rgb_domain(
    rgb: torch.Tensor,
    rgb_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.ndim != 4 or rgb.shape[1] not in {1, 3} or not rgb.is_floating_point():
        raise ValueError("rgb must be floating [B,1|3,H,W]")
    expected = (rgb.shape[0], 1, *rgb.shape[-2:])
    if rgb_valid is not None and (
        rgb_valid.shape != expected or rgb_valid.dtype != torch.bool or rgb_valid.device != rgb.device
    ):
        raise ValueError("rgb_valid must be bool [B,1,H,W] on RGB device")
    finite = torch.isfinite(rgb).all(1, keepdim=True)
    domain = finite if rgb_valid is None else (finite & rgb_valid)
    safe = torch.where(domain, rgb, torch.zeros_like(rgb))
    return _gray(safe), domain


def _erode(domain: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return domain
    invalid = F.max_pool2d(
        (~domain).to(torch.float32),
        2 * radius + 1,
        stride=1,
        padding=radius,
    ) > 0
    return ~invalid


def _points(
    score: torch.Tensor,
    num_points: int,
    margin: int,
    domain: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if num_points < 1 or margin < 0:
        raise ValueError("num_points must be positive and margin non-negative")
    batch, _, height, width = score.shape
    if height <= 2 * margin or width <= 2 * margin or num_points > (height - 2 * margin) * (width - 2 * margin):
        raise ValueError("margin leaves fewer usable pixels than num_points")
    if domain is not None and (
        domain.shape != (batch, 1, height, width)
        or domain.dtype != torch.bool
        or domain.device != score.device
    ):
        raise ValueError("selector domain must be bool [B,1,H,W] on score device")
    usable = torch.zeros_like(score, dtype=torch.bool)
    usable[..., margin : height - margin, margin : width - margin] = True
    if domain is not None:
        usable &= domain
    values, indices = score.masked_fill(~usable, -torch.inf).flatten(1).topk(num_points, dim=1)
    valid = torch.isfinite(values) & (values > 0)
    xy = torch.stack((indices % width, indices // width), dim=-1).long()
    return {
        "pixels_xy": xy,
        "score": torch.where(valid, values, torch.zeros_like(values)),
        "valid": valid,
    }


def _nms(score: torch.Tensor) -> torch.Tensor:
    local_max = F.max_pool2d(score, 3, stride=1, padding=1)
    return torch.where((score > 0) & (score == local_max), score, torch.zeros_like(score))


def spatial_select_points(
    pixels_xy: torch.Tensor,
    score: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_points: int,
    height: int,
    width: int,
    tiles: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Apply a common per-tile quota to an overcomplete selector candidate set."""
    rows, cols = tiles
    if (
        pixels_xy.ndim != 3 or pixels_xy.shape[-1] != 2 or pixels_xy.shape[1] < 1 or score.shape != pixels_xy.shape[:2]
        or not score.is_floating_point() or valid.shape != score.shape or valid.dtype is not torch.bool or pixels_xy.dtype not in (torch.int32, torch.int64)
        or num_points < 1 or height < 1 or width < 1 or rows < 1 or cols < 1
    ):
        raise ValueError("matching candidate tensors, positive image shape, point count, and tile shape are required")
    if pixels_xy.device != score.device or valid.device != score.device:
        raise ValueError("candidate tensors must share a device")
    x, y = pixels_xy[..., 0], pixels_xy[..., 1]
    candidate_valid = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height) & torch.isfinite(score)
    tile = (y.clamp(0, height - 1) * rows // height) * cols + x.clamp(0, width - 1) * cols // width
    quota = (num_points + rows * cols - 1) // (rows * cols)
    candidates: list[torch.Tensor] = []
    candidate_values: list[torch.Tensor] = []
    for index in range(rows * cols):
        values = score.masked_fill(~(candidate_valid & (tile == index)), -torch.inf)
        values, indices = values.topk(min(quota, values.shape[1]), dim=1)
        candidates.append(indices)
        candidate_values.append(values)
    indices = torch.cat(candidates, dim=1)
    values = torch.cat(candidate_values, dim=1)
    selected_valid = torch.isfinite(values)
    values, order = values.masked_fill(~selected_valid, -torch.inf).topk(min(num_points, values.shape[1]), dim=1)
    indices = indices.gather(1, order)
    selected_valid = torch.isfinite(values)
    if indices.shape[1] < num_points:
        padding = num_points - indices.shape[1]
        indices = torch.cat((indices, torch.zeros((indices.shape[0], padding), dtype=indices.dtype, device=indices.device)), 1)
        values = torch.cat((values, torch.zeros((values.shape[0], padding), dtype=values.dtype, device=values.device)), 1)
        selected_valid = torch.cat((selected_valid, torch.zeros((selected_valid.shape[0], padding), dtype=torch.bool, device=selected_valid.device)), 1)
    pixels = pixels_xy.gather(1, indices[..., None].expand(-1, -1, 2))
    return {"pixels_xy": torch.where(selected_valid[..., None], pixels, torch.zeros_like(pixels)), "score": torch.where(selected_valid, values, torch.zeros_like(values)), "valid": selected_valid}


def uniform_select(
    rgb: torch.Tensor,
    *,
    rgb_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    margin: int = 3,
) -> dict[str, torch.Tensor]:
    """Deterministic no-saliency control with the common selector contract."""
    gray, domain = _rgb_domain(rgb, rgb_valid)
    batch, _, height, width = gray.shape
    if margin < 0 or height <= 2 * margin or width <= 2 * margin or num_points < 1 or num_points > (height - 2 * margin) * (width - 2 * margin):
        raise ValueError("margin must be non-negative and leave at least num_points usable pixels")
    linear = torch.linspace(0, (height - 2 * margin) * (width - 2 * margin) - 1, num_points, device=gray.device).round().long()
    usable_width = width - 2 * margin
    xy = torch.stack((linear % usable_width + margin, linear // usable_width + margin), dim=-1)[None].expand(batch, -1, -1)
    flat = xy[..., 1] * width + xy[..., 0]
    valid = domain[:, 0].flatten(1).gather(1, flat)
    return {
        "pixels_xy": xy,
        "score": valid.to(gray.dtype),
        "valid": valid,
    }


def shi_tomasi_select(
    rgb: torch.Tensor,
    *,
    rgb_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    margin: int = 3,
    window: int = 5,
) -> dict[str, torch.Tensor]:
    """Shi--Tomasi structure-tensor corners with a bounded common output."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be positive and odd")
    if margin < window // 2 + 1:
        raise ValueError("margin must cover the structure-tensor window and gradient stencil")
    gray, domain = _rgb_domain(rgb, rgb_valid)
    gx = F.pad((gray[..., 2:] - gray[..., :-2]) * 0.5, (1, 1, 0, 0))
    gy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * 0.5, (0, 0, 1, 1))
    a = F.avg_pool2d(gx.square(), window, stride=1, padding=window // 2)
    b = F.avg_pool2d(gx * gy, window, stride=1, padding=window // 2)
    c = F.avg_pool2d(gy.square(), window, stride=1, padding=window // 2)
    score = (a + c - ((a - c).square() + 4 * b.square()).sqrt()).clamp_min(0) * 0.5
    return _points(_nms(score), num_points, margin, _erode(domain, window // 2 + 1))


def fast_select(
    rgb: torch.Tensor,
    *,
    rgb_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    margin: int = 3,
    threshold: float = 0.08,
    arc: int = 9,
) -> dict[str, torch.Tensor]:
    """Tensorized FAST-9/16 score, followed by the common bounded selection."""
    if threshold <= 0 or arc < 1 or arc > 16:
        raise ValueError("threshold must be positive and arc must be in [1,16]")
    if margin < 3:
        raise ValueError("FAST margin must be at least the three-pixel ring radius")
    gray, domain = _rgb_domain(rgb, rgb_valid)
    offsets = ((0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3))
    ring = torch.stack([torch.roll(gray, shifts=(-dy, -dx), dims=(-2, -1))[:, 0] for dx, dy in offsets], dim=-1)
    center = gray[:, 0, ..., None]
    bright, dark = ring > center + threshold, ring < center - threshold
    runs = [
        bright.roll(-start, -1)[..., :arc].all(-1) | dark.roll(-start, -1)[..., :arc].all(-1)
        for start in range(16)
    ]
    detected = torch.stack(runs, -1).any(-1)
    score = (ring - center).abs().amax(-1) * detected
    return _points(_nms(score[:, None]), num_points, margin, _erode(domain, 3))


def pose_observable_select(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_reference_c2w: torch.Tensor,
    pose_neighbor_c2w: torch.Tensor,
    *,
    rgb_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    margin: int = 3,
    min_depth: float = 0.25,
    max_depth: float = 20.0,
) -> dict[str, torch.Tensor]:
    """Select pixels whose image gradient constrains known-pose depth search."""
    gray, domain = _rgb_domain(rgb, rgb_valid)
    batch, _, height, width = gray.shape
    if (
        intrinsics.shape != (batch, 3, 3)
        or pose_reference_c2w.shape != (batch, 4, 4)
        or pose_neighbor_c2w.shape != (batch, 4, 4)
        or any(value.device != rgb.device or value.dtype != rgb.dtype for value in (intrinsics, pose_reference_c2w, pose_neighbor_c2w))
        or min_depth <= 0
        or max_depth <= min_depth
    ):
        raise ValueError("camera tensors must be [B,*] on RGB device/dtype with a valid depth interval")
    y, x = torch.meshgrid(
        torch.arange(height, device=rgb.device, dtype=rgb.dtype),
        torch.arange(width, device=rgb.device, dtype=rgb.dtype),
        indexing="ij",
    )
    rays = torch.stack((
        (x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0],
        (y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1],
        torch.ones((batch, height, width), device=rgb.device, dtype=rgb.dtype),
    ), -1)
    rotation = pose_neighbor_c2w[:, :3, :3].transpose(-1, -2) @ pose_reference_c2w[:, :3, :3]
    translation = torch.einsum("bij,bj->bi", pose_neighbor_c2w[:, :3, :3].transpose(-1, -2), pose_reference_c2w[:, :3, 3] - pose_neighbor_c2w[:, :3, 3])

    def project(depth: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.einsum("bij,bhwj->bhwi", rotation, rays * depth) + translation[:, None, None]
        z = points[..., 2]
        safe_z = torch.where(z > torch.finfo(rgb.dtype).eps, z, torch.ones_like(z))
        return (
            points[..., 0] / safe_z * intrinsics[:, None, None, 0, 0] + intrinsics[:, None, None, 0, 2],
            points[..., 1] / safe_z * intrinsics[:, None, None, 1, 1] + intrinsics[:, None, None, 1, 2],
            z,
        )

    near_u, near_v, near_z = project(min_depth)
    far_u, far_v, far_z = project(max_depth)
    du, dv = far_u - near_u, far_v - near_v
    magnitude = (du.square() + dv.square()).sqrt()
    gx = F.pad((gray[..., 2:] - gray[..., :-2]) * 0.5, (1, 1, 0, 0))[:, 0]
    gy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * 0.5, (0, 0, 1, 1))[:, 0]
    directional_gradient = (gx * du + gy * dv).abs() / magnitude.clamp_min(torch.finfo(rgb.dtype).eps)
    projection_valid = torch.isfinite(near_u) & torch.isfinite(near_v) & torch.isfinite(far_u) & torch.isfinite(far_v) & (near_z > 0) & (far_z > 0)
    score = directional_gradient * magnitude.clamp(max=1) * projection_valid
    return _points(_nms(score[:, None]), num_points, margin, _erode(domain, 1))
