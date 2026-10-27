# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .sparse_geometry import _intrinsics, _pose


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[1] == 1:
        return rgb
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[-2:]
    x = 2.0 * u / max(w - 1, 1) - 1.0
    y = 2.0 * v / max(h - 1, 1) - 1.0
    grid = torch.stack((x, y), -1).reshape(image.shape[0], -1, 1, 2)
    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[:, 0, :, 0].reshape(u.shape)


def _rgb_mask(
    rgb: torch.Tensor,
    value: torch.Tensor | None,
    *,
    name: str,
) -> torch.Tensor:
    expected = (rgb.shape[0], 1, *rgb.shape[-2:])
    if value is not None and (
        value.shape != expected or value.dtype != torch.bool or value.device != rgb.device
    ):
        raise ValueError(f"{name} must be bool [B,1,H,W] on RGB device")
    finite = torch.isfinite(rgb).all(1, keepdim=True)
    return finite if value is None else (finite & value)


def _neighbor_rgb_mask(
    rgb: torch.Tensor,
    value: torch.Tensor | None,
) -> torch.Tensor:
    expected = (rgb.shape[0], rgb.shape[1], 1, *rgb.shape[-2:])
    if value is not None and (
        value.shape != expected or value.dtype != torch.bool or value.device != rgb.device
    ):
        raise ValueError("rgb_neighbors_valid must be bool [B,N,1,H,W] on RGB device")
    finite = torch.isfinite(rgb).all(2, keepdim=True)
    return finite if value is None else (finite & value)


def _erode_valid(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    invalid = F.max_pool2d(
        (~mask).to(torch.float32),
        2 * radius + 1,
        stride=1,
        padding=radius,
    ) > 0
    return ~invalid


def _sample_valid(mask: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    sampled = _sample(mask.to(dtype), u, v)
    return sampled >= 1.0 - 8.0 * torch.finfo(dtype).eps


def _masked_stats(gray: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = valid.to(gray.dtype)
    count = weight.sum((-2, -1)).clamp_min(1)
    mean = (gray * weight).sum((-2, -1)) / count
    centered = gray - mean[..., None, None]
    variance = (centered.square() * weight).sum((-2, -1)) / count
    return mean[:, 0], variance[:, 0].sqrt().clamp_min(1e-4)


def _first_unique(flat: torch.Tensor, valid: torch.Tensor, size: int) -> torch.Tensor:
    batch, points = flat.shape
    slots = torch.arange(points, device=flat.device, dtype=torch.long)[None].expand(batch, -1)
    sentinel = torch.full_like(slots, points)
    first = torch.full((batch, size), points, device=flat.device, dtype=torch.long)
    first.scatter_reduce_(
        1,
        flat,
        torch.where(valid, slots, sentinel),
        reduce="amin",
        include_self=True,
    )
    return valid & (first.gather(1, flat) == slots)


def _select_points(
    gray: torch.Tensor,
    count: int,
    margin: int,
    center_valid: torch.Tensor | None = None,
):
    gx = F.pad((gray[..., 2:] - gray[..., :-2]) * 0.5, (1, 1, 0, 0))
    gy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * 0.5, (0, 0, 1, 1))
    score = (gx.square() + gy.square()).sqrt()
    batch, _, h, w = score.shape
    usable = torch.ones_like(score, dtype=torch.bool)
    usable[..., :margin, :] = usable[..., -margin:, :] = False
    usable[..., :, :margin] = usable[..., :, -margin:] = False
    if center_valid is not None:
        if center_valid.shape != (batch, 1, h, w) or center_valid.dtype != torch.bool or center_valid.device != gray.device:
            raise ValueError("D6 selection validity must be bool [B,1,H,W] on RGB device")
        usable &= center_valid
    score = score.masked_fill(~usable, -torch.inf)
    aspect = w / max(h, 1)
    rows = min(h, max(1, round(math.sqrt(count / max(aspect, 1e-6)))))
    cols = min(w, max(1, math.ceil(count / rows)))
    values, indices = F.adaptive_max_pool2d(score, (rows, cols), return_indices=True)
    values, indices = values.flatten(1), indices.flatten(1)
    k = min(count, values.shape[1])
    order = values.topk(k, dim=1).indices
    indices, values = indices.gather(1, order), values.gather(1, order)
    valid = torch.isfinite(values) & (values > 0)
    valid = _first_unique(indices, valid, h * w)
    return (
        indices % w,
        indices // w,
        torch.where(valid, values, torch.zeros_like(values)),
        valid,
    )


def _external_points(
    gray: torch.Tensor,
    pixels_xy: torch.Tensor,
    pixels_valid: torch.Tensor | None,
    margin: int,
    center_valid: torch.Tensor,
):
    """Preserve fixed-capacity external points without host synchronization."""
    batch, _, height, width = gray.shape
    if (
        pixels_xy.ndim != 3
        or pixels_xy.shape[0] != batch
        or pixels_xy.shape[-1] != 2
        or pixels_xy.shape[1] < 1
        or pixels_xy.device != gray.device
        or pixels_xy.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("pixels_xy must be nonempty integer [B,P,2] on the RGB device")
    points = pixels_xy.shape[1]
    if pixels_valid is None:
        pixels_valid = torch.ones((batch, points), dtype=torch.bool, device=gray.device)
    elif pixels_valid.shape != (batch, points) or pixels_valid.dtype != torch.bool or pixels_valid.device != gray.device:
        raise ValueError("pixels_valid must be bool [B,P] on the RGB device")
    raw_x, raw_y = pixels_xy[..., 0].long(), pixels_xy[..., 1].long()
    in_bounds = (
        (raw_x >= margin)
        & (raw_x < width - margin)
        & (raw_y >= margin)
        & (raw_y < height - margin)
    )
    px = raw_x.clamp(margin, width - margin - 1)
    py = raw_y.clamp(margin, height - margin - 1)
    flat = py * width + px
    center_ok = center_valid[:, 0].flatten(1).gather(1, flat)
    selected = pixels_valid & in_bounds & center_ok
    selected = _first_unique(flat, selected, height * width)
    gx = F.pad((gray[..., 2:] - gray[..., :-2]) * 0.5, (1, 1, 0, 0))
    gy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * 0.5, (0, 0, 1, 1))
    gradient = (gx.square() + gy.square()).sqrt().flatten(1).gather(1, flat)
    return px, py, torch.where(selected, gradient, torch.zeros_like(gradient)), selected


def _validate_d6(
    rgb_reference: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    pose_reference: torch.Tensor,
    pose_neighbors: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    rgb_reference_valid: torch.Tensor | None,
    rgb_neighbors_valid: torch.Tensor | None,
    pixels_xy: torch.Tensor | None,
    pixels_valid: torch.Tensor | None,
    prior_inverse_depth: torch.Tensor | None,
    num_points: int,
    num_hypotheses: int,
    min_depth: float,
    max_depth: float,
    patch_radius: int,
    min_views: int,
    min_parallax_px: float,
    huber_delta: float,
    min_confidence: float,
    validate: bool,
) -> tuple[int, int, int, int]:
    if rgb_reference.ndim != 4 or rgb_reference.shape[1] not in {1, 3} or not rgb_reference.is_floating_point():
        raise ValueError("rgb_reference must be floating [B,1|3,H,W]")
    batch, _, height, width = rgb_reference.shape
    if rgb_reference.device.type == "cpu" and rgb_reference.dtype not in {torch.float32, torch.float64}:
        raise ValueError("CPU D6 reference supports float32/float64; use GPU for low precision")
    if (
        rgb_neighbors.ndim != 5
        or rgb_neighbors.shape[0] != batch
        or rgb_neighbors.shape[1] < 1
        or rgb_neighbors.shape[2] not in {1, 3}
        or rgb_neighbors.shape[-2:] != (height, width)
        or rgb_neighbors.device != rgb_reference.device
        or rgb_neighbors.dtype != rgb_reference.dtype
    ):
        raise ValueError("rgb_neighbors must be matching floating [B,N,1|3,H,W]")
    neighbors = rgb_neighbors.shape[1]
    if (
        pose_reference.shape != (batch, 4, 4)
        or pose_neighbors.shape != (batch, neighbors, 4, 4)
        or intrinsics.shape != (batch, 3, 3)
        or any(value.device != rgb_reference.device or value.dtype != rgb_reference.dtype for value in (pose_reference, pose_neighbors, intrinsics))
    ):
        raise ValueError("poses and intrinsics must match RGB batch/device/dtype")
    if rgb_reference_valid is not None and (
        rgb_reference_valid.shape != (batch, 1, height, width)
        or rgb_reference_valid.dtype != torch.bool
        or rgb_reference_valid.device != rgb_reference.device
    ):
        raise ValueError("rgb_reference_valid must be bool [B,1,H,W] on RGB device")
    if rgb_neighbors_valid is not None and (
        rgb_neighbors_valid.shape != (batch, neighbors, 1, height, width)
        or rgb_neighbors_valid.dtype != torch.bool
        or rgb_neighbors_valid.device != rgb_reference.device
    ):
        raise ValueError("rgb_neighbors_valid must be bool [B,N,1,H,W] on RGB device")
    if (
        num_points < 1
        or num_hypotheses < 2
        or patch_radius < 0
        or not 1 <= min_views <= neighbors
        or not math.isfinite(min_depth)
        or not math.isfinite(max_depth)
        or not 0 < min_depth < max_depth
        or not math.isfinite(min_parallax_px)
        or min_parallax_px < 0
        or not math.isfinite(huber_delta)
        or huber_delta <= 0
        or not math.isfinite(min_confidence)
        or not 0 <= min_confidence <= 1
    ):
        raise ValueError("D6 point/search/quality parameters are invalid")
    margin = patch_radius + 2
    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError("image is too small for the requested D6 patch margin")
    if pixels_valid is not None and pixels_xy is None:
        raise ValueError("pixels_valid requires pixels_xy")
    if prior_inverse_depth is not None and (
        prior_inverse_depth.shape != (batch, 1, height, width)
        or prior_inverse_depth.device != rgb_reference.device
        or prior_inverse_depth.dtype != rgb_reference.dtype
    ):
        raise ValueError("prior_inverse_depth must be [B,1,H,W] on RGB device/dtype")
    if validate:
        _intrinsics(intrinsics, batch, rgb_reference.device, rgb_reference.dtype)
        _pose(pose_reference, batch, rgb_reference.device, rgb_reference.dtype)
        _pose(
            pose_neighbors.reshape(batch * neighbors, 4, 4),
            batch * neighbors,
            rgb_reference.device,
            rgb_reference.dtype,
        )
    return batch, neighbors, height, width


def _dense(
    values: torch.Tensor,
    flat: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype or values.dtype
    if target_dtype == torch.bool:
        out = torch.zeros((batch, height * width), device=values.device, dtype=torch.int8)
        out.scatter_reduce_(1, flat, values.to(torch.int8), reduce="amax", include_self=True)
        return out.bool().reshape(batch, 1, height, width)
    out = torch.zeros((batch, height * width), device=values.device, dtype=target_dtype)
    out.scatter_reduce_(1, flat, values.to(target_dtype), reduce="amax", include_self=True)
    return out.reshape(batch, 1, height, width)


def dso_select_points(
    rgb: torch.Tensor,
    *,
    rgb_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    patch_radius: int = 1,
) -> dict[str, torch.Tensor]:
    """Select bounded gradient-supported D6a points as ``[B,P,2]`` xy indices."""
    if rgb.ndim != 4 or rgb.shape[1] not in {1, 3} or not rgb.is_floating_point() or num_points < 1 or patch_radius < 0:
        raise ValueError("rgb must be floating [B,1|3,H,W] with positive point count")
    if rgb.device.type == "cpu" and rgb.dtype not in {torch.float32, torch.float64}:
        raise ValueError("CPU D6 selector supports float32/float64")
    margin = patch_radius + 2
    if rgb.shape[-2] <= 2 * margin or rgb.shape[-1] <= 2 * margin:
        raise ValueError("image is too small for the requested D6 patch margin")
    pixel_valid = _rgb_mask(rgb, rgb_valid, name="rgb_valid")
    safe_rgb = torch.where(pixel_valid, rgb, torch.zeros_like(rgb))
    center_valid = _erode_valid(pixel_valid, max(1, patch_radius))
    px, py, gradient, valid = _select_points(_gray(safe_rgb), num_points, margin, center_valid)
    return {"pixels_xy": torch.stack((px, py), -1), "score": gradient, "gradient": gradient, "valid": valid}


def direct_pose_depth_dso(
    *,
    rgb_reference: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    pose_reference: torch.Tensor,
    pose_neighbors: torch.Tensor,
    intrinsics: torch.Tensor,
    rgb_reference_valid: torch.Tensor | None = None,
    rgb_neighbors_valid: torch.Tensor | None = None,
    pixels_xy: torch.Tensor | None = None,
    pixels_valid: torch.Tensor | None = None,
    prior_inverse_depth: torch.Tensor | None = None,
    num_points: int = 2048,
    num_hypotheses: int = 64,
    min_depth: float = 0.25,
    max_depth: float = 20.0,
    patch_radius: int = 1,
    min_views: int = 1,
    min_parallax_px: float = 0.5,
    huber_delta: float = 0.03,
    prior_rel_radius: float = 0.5,
    affine_brightness: bool = True,
    min_confidence: float = 0.01,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Fixed-pose DSO depth from sparse inverse-depth photometric consistency.

    Poses are camera-to-world. ``pixels_valid`` preserves the fixed-capacity
    selector contract. RGB validity masks additionally keep camera-conditioning
    borders out of reference selection and projected neighbour patches.
    """
    if not math.isfinite(prior_rel_radius) or not 0 <= prior_rel_radius < 1:
        raise ValueError("prior_rel_radius must be finite in [0,1)")
    b, n, h, w = _validate_d6(
        rgb_reference,
        rgb_neighbors,
        pose_reference,
        pose_neighbors,
        intrinsics,
        rgb_reference_valid=rgb_reference_valid,
        rgb_neighbors_valid=rgb_neighbors_valid,
        pixels_xy=pixels_xy,
        pixels_valid=pixels_valid,
        prior_inverse_depth=prior_inverse_depth,
        num_points=num_points,
        num_hypotheses=num_hypotheses,
        min_depth=min_depth,
        max_depth=max_depth,
        patch_radius=patch_radius,
        min_views=min_views,
        min_parallax_px=min_parallax_px,
        huber_delta=huber_delta,
        min_confidence=min_confidence,
        validate=validate,
    )
    ref_valid = _rgb_mask(rgb_reference, rgb_reference_valid, name="rgb_reference_valid")
    neighbor_valid = _neighbor_rgb_mask(rgb_neighbors, rgb_neighbors_valid)
    safe_reference = torch.where(ref_valid, rgb_reference, torch.zeros_like(rgb_reference))
    safe_neighbors = torch.where(neighbor_valid, rgb_neighbors, torch.zeros_like(rgb_neighbors))
    ref = _gray(safe_reference)
    neighbors = _gray(safe_neighbors.reshape(b * n, rgb_neighbors.shape[2], h, w)).reshape(b, n, 1, h, w)
    center_valid = _erode_valid(ref_valid, max(1, patch_radius))
    margin_px = patch_radius + 2
    if pixels_xy is None:
        px, py, gradient, selected = _select_points(ref, num_points, margin_px, center_valid)
    else:
        px, py, gradient, selected = _external_points(
            ref,
            pixels_xy,
            pixels_valid,
            margin_px,
            center_valid,
        )
    p = px.shape[1]
    offsets = torch.tensor(
        [(dx, dy) for dy in range(-patch_radius, patch_radius + 1) for dx in range(-patch_radius, patch_radius + 1)],
        device=ref.device,
        dtype=ref.dtype,
    )
    samples = offsets.shape[0]
    u = px[..., None].to(ref.dtype) + offsets[:, 0]
    v = py[..., None].to(ref.dtype) + offsets[:, 1]
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    rays = torch.stack(((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), -1)
    reference_patch = _sample(ref, u, v)
    rho_min, rho_max = 1.0 / max_depth, 1.0 / min_depth
    if prior_inverse_depth is None:
        low = torch.full((b, p), rho_min, device=ref.device, dtype=ref.dtype)
        high = torch.full((b, p), rho_max, device=ref.device, dtype=ref.dtype)
    else:
        prior = prior_inverse_depth.flatten(1).gather(1, py * w + px)
        has_prior = selected & torch.isfinite(prior) & (prior > 0)
        safe_prior = torch.where(has_prior, prior, torch.zeros_like(prior))
        low = torch.where(
            has_prior,
            (safe_prior * (1.0 - prior_rel_radius)).clamp(rho_min, rho_max),
            torch.full_like(prior, rho_min),
        )
        high = torch.where(
            has_prior,
            (safe_prior * (1.0 + prior_rel_radius)).clamp(rho_min, rho_max),
            torch.full_like(prior, rho_max),
        )
        high = torch.maximum(high, low + torch.finfo(ref.dtype).eps)
    alpha = torch.linspace(0, 1, num_hypotheses, device=ref.device, dtype=ref.dtype)
    rho = low[..., None] + (high - low)[..., None] * alpha
    cost_sum = torch.zeros_like(rho)
    view_count = torch.zeros_like(rho)
    parallax_sum = torch.zeros_like(rho)
    ref_mean, ref_std = _masked_stats(ref, ref_valid)
    flat_neighbor_valid = neighbor_valid.reshape(b * n, 1, h, w)
    neighbor_mean, neighbor_std = _masked_stats(
        neighbors.reshape(b * n, 1, h, w),
        flat_neighbor_valid,
    )
    neighbor_mean = neighbor_mean.reshape(b, n)
    neighbor_std = neighbor_std.reshape(b, n)
    r_ref, t_ref = pose_reference[:, :3, :3], pose_reference[:, :3, 3]
    center = samples // 2
    min_patch_samples = max(1, math.ceil(samples * 0.7))
    eps = torch.finfo(ref.dtype).eps
    for i in range(n):
        r_n, t_n = pose_neighbors[:, i, :3, :3], pose_neighbors[:, i, :3, 3]
        r_rel = r_n.transpose(-1, -2) @ r_ref
        t_rel = (r_n.transpose(-1, -2) @ (t_ref - t_n)[..., None])[..., 0]
        xyz = rays[:, :, None] / rho[:, :, :, None, None]
        xyz = torch.einsum("bij,bpdsj->bpdsi", r_rel, xyz) + t_rel[:, None, None, None]
        z = xyz[..., 2]
        positive_z = z > eps
        inv_z = torch.where(positive_z, z.reciprocal(), torch.zeros_like(z))
        uu = intrinsics[:, 0, 0, None, None, None] * xyz[..., 0] * inv_z + intrinsics[:, 0, 2, None, None, None]
        vv = intrinsics[:, 1, 1, None, None, None] * xyz[..., 1] * inv_z + intrinsics[:, 1, 2, None, None, None]
        finite_uv = torch.isfinite(uu) & torch.isfinite(vv)
        sample_u = torch.where(finite_uv, uu, torch.zeros_like(uu))
        sample_v = torch.where(finite_uv, vv, torch.zeros_like(vv))
        inside = positive_z & finite_uv & (uu >= 0) & (uu <= w - 1) & (vv >= 0) & (vv <= h - 1)
        inside &= _sample_valid(neighbor_valid[:, i], sample_u, sample_v, ref.dtype)
        observed = _sample(neighbors[:, i], sample_u, sample_v)
        if affine_brightness:
            scale = (ref_std / neighbor_std[:, i]).clamp(0.5, 2.0)
            observed = observed * scale[:, None, None, None] + (
                ref_mean - scale * neighbor_mean[:, i]
            )[:, None, None, None]
        residual = observed - reference_patch[:, :, None]
        absolute = residual.abs()
        robust = torch.where(
            absolute <= huber_delta,
            0.5 * residual.square(),
            huber_delta * (absolute - 0.5 * huber_delta),
        )
        patch_count = inside.sum(-1)
        view_valid = patch_count >= min_patch_samples
        view_cost = (robust * inside.to(robust.dtype)).sum(-1) / patch_count.clamp_min(1)
        parallax = ((uu[..., center] - px[..., None]).square() + (vv[..., center] - py[..., None]).square()).sqrt()
        cost_sum += torch.where(view_valid, view_cost, torch.zeros_like(view_cost))
        view_count += view_valid
        parallax_sum += torch.where(view_valid, parallax, torch.zeros_like(parallax))
    cost = cost_sum / view_count.clamp_min(1)
    parallax = parallax_sum / view_count.clamp_min(1)
    usable = (view_count >= min_views) & selected[..., None]
    cost = cost.masked_fill(~usable, torch.inf)
    best_two, best_indices = cost.topk(2, dim=-1, largest=False)
    best, second, index = best_two[..., 0], best_two[..., 1], best_indices[..., 0]
    rho_best = rho.gather(-1, index[..., None])[..., 0]
    if num_hypotheses >= 3:
        previous = cost.gather(-1, (index - 1).clamp_min(0)[..., None])[..., 0]
        following = cost.gather(-1, (index + 1).clamp_max(num_hypotheses - 1)[..., None])[..., 0]
        curvature = previous - 2.0 * best + following
        interior = (index > 0) & (index < num_hypotheses - 1)
        delta = torch.where(
            interior & torch.isfinite(previous) & torch.isfinite(following) & (curvature.abs() > 1e-12),
            0.5 * (previous - following) / curvature,
            torch.zeros_like(best),
        ).clamp(-0.5, 0.5)
        rho_best = (
            rho_best + delta * (high - low) / (num_hypotheses - 1)
        ).clamp_min(torch.finfo(ref.dtype).tiny)
    parallax_best = parallax.gather(-1, index[..., None])[..., 0]
    views_best = view_count.gather(-1, index[..., None])[..., 0]
    margin = ((second - best).clamp_min(0) / (second.abs() + eps)).nan_to_num()
    photo = torch.exp(-best.nan_to_num(1e6) / max(huber_delta * huber_delta, float(eps)))
    parallax_conf = (parallax_best / max(min_parallax_px, float(eps))).clamp(0, 1)
    confidence = (margin * photo * parallax_conf * (views_best / max(n, 1))).clamp(0, 1)
    valid = selected & torch.isfinite(best) & (views_best >= min_views) & (parallax_best >= min_parallax_px) & (confidence >= min_confidence)
    depth = rho_best.clamp_min(torch.finfo(ref.dtype).tiny).reciprocal()
    flat = py * w + px
    return {
        "inverse_depth": _dense(torch.where(valid, rho_best, torch.zeros_like(rho_best)), flat, batch=b, height=h, width=w),
        "depth": _dense(torch.where(valid, depth, torch.zeros_like(depth)), flat, batch=b, height=h, width=w),
        "valid": _dense(valid, flat, batch=b, height=h, width=w, dtype=torch.bool),
        "selected": _dense(selected, flat, batch=b, height=h, width=w, dtype=torch.bool),
        "confidence": _dense(torch.where(valid, confidence, torch.zeros_like(confidence)), flat, batch=b, height=h, width=w),
        "photometric_error": _dense(torch.where(valid, best, torch.zeros_like(best)), flat, batch=b, height=h, width=w),
        "parallax_px": _dense(torch.where(valid, parallax_best, torch.zeros_like(parallax_best)), flat, batch=b, height=h, width=w),
        "gradient": _dense(gradient, flat, batch=b, height=h, width=w),
    }


def direct_pose_depth_tensorized(
    *,
    rgb_reference: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    pose_reference: torch.Tensor,
    pose_neighbors: torch.Tensor,
    intrinsics: torch.Tensor,
    rgb_reference_valid: torch.Tensor | None = None,
    rgb_neighbors_valid: torch.Tensor | None = None,
    pixels_xy: torch.Tensor | None = None,
    pixels_valid: torch.Tensor | None = None,
    num_points: int = 2048,
    num_hypotheses: int = 64,
    min_depth: float = 0.25,
    max_depth: float = 20.0,
    patch_radius: int = 1,
    min_views: int = 1,
    min_parallax_px: float = 0.5,
    huber_delta: float = 0.03,
    min_confidence: float = 0.01,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Tensorized D6 comparison reference with no Python neighbour loop."""
    b, n, h, w = _validate_d6(
        rgb_reference,
        rgb_neighbors,
        pose_reference,
        pose_neighbors,
        intrinsics,
        rgb_reference_valid=rgb_reference_valid,
        rgb_neighbors_valid=rgb_neighbors_valid,
        pixels_xy=pixels_xy,
        pixels_valid=pixels_valid,
        prior_inverse_depth=None,
        num_points=num_points,
        num_hypotheses=num_hypotheses,
        min_depth=min_depth,
        max_depth=max_depth,
        patch_radius=patch_radius,
        min_views=min_views,
        min_parallax_px=min_parallax_px,
        huber_delta=huber_delta,
        min_confidence=min_confidence,
        validate=validate,
    )
    ref_valid = _rgb_mask(rgb_reference, rgb_reference_valid, name="rgb_reference_valid")
    neighbor_valid = _neighbor_rgb_mask(rgb_neighbors, rgb_neighbors_valid)
    safe_reference = torch.where(ref_valid, rgb_reference, torch.zeros_like(rgb_reference))
    safe_neighbors = torch.where(neighbor_valid, rgb_neighbors, torch.zeros_like(rgb_neighbors))
    ref = _gray(safe_reference)
    neighbors = _gray(safe_neighbors.reshape(b * n, rgb_neighbors.shape[2], h, w)).reshape(b, n, 1, h, w)
    center_valid = _erode_valid(ref_valid, max(1, patch_radius))
    margin_px = patch_radius + 2
    if pixels_xy is None:
        px, py, gradient, selected = _select_points(ref, num_points, margin_px, center_valid)
    else:
        px, py, gradient, selected = _external_points(
            ref,
            pixels_xy,
            pixels_valid,
            margin_px,
            center_valid,
        )
    p = px.shape[1]
    offsets = torch.tensor(
        [(dx, dy) for dy in range(-patch_radius, patch_radius + 1) for dx in range(-patch_radius, patch_radius + 1)],
        device=ref.device,
        dtype=ref.dtype,
    )
    samples = offsets.shape[0]
    u = px[..., None].to(ref.dtype) + offsets[:, 0]
    v = py[..., None].to(ref.dtype) + offsets[:, 1]
    rays = torch.stack(
        (
            (u - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None],
            (v - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None],
            torch.ones_like(u),
        ),
        -1,
    )
    reference_patch = _sample(ref, u, v)
    rho = torch.linspace(1.0 / max_depth, 1.0 / min_depth, num_hypotheses, device=ref.device, dtype=ref.dtype)
    xyz = rays[:, :, None] / rho[None, None, :, None, None]
    r_ref, t_ref = pose_reference[:, :3, :3], pose_reference[:, :3, 3]
    r_n, t_n = pose_neighbors[:, :, :3, :3], pose_neighbors[:, :, :3, 3]
    r_rel = r_n.transpose(-1, -2) @ r_ref[:, None]
    t_rel = (r_n.transpose(-1, -2) @ (t_ref[:, None] - t_n)[..., None])[..., 0]
    transformed = torch.einsum("bnij,bpdsj->bnpdsi", r_rel, xyz) + t_rel[:, :, None, None, None]
    z = transformed[..., 2]
    eps = torch.finfo(ref.dtype).eps
    safe_z = torch.where(z > eps, z, torch.ones_like(z))
    uu = intrinsics[:, None, 0, 0, None, None, None] * transformed[..., 0] / safe_z + intrinsics[:, None, 0, 2, None, None, None]
    vv = intrinsics[:, None, 1, 1, None, None, None] * transformed[..., 1] / safe_z + intrinsics[:, None, 1, 2, None, None, None]
    finite_uv = torch.isfinite(uu) & torch.isfinite(vv)
    sample_u = torch.where(finite_uv, uu, torch.zeros_like(uu))
    sample_v = torch.where(finite_uv, vv, torch.zeros_like(vv))
    inside = (z > eps) & finite_uv & (uu >= 0) & (uu <= w - 1) & (vv >= 0) & (vv <= h - 1)
    flat_neighbor_valid = neighbor_valid.reshape(b * n, 1, h, w)
    sampled_neighbor_valid = _sample_valid(
        flat_neighbor_valid,
        sample_u.reshape(b * n, p, num_hypotheses, samples),
        sample_v.reshape(b * n, p, num_hypotheses, samples),
        ref.dtype,
    ).reshape(b, n, p, num_hypotheses, samples)
    inside &= sampled_neighbor_valid
    observed = _sample(
        neighbors.reshape(b * n, 1, h, w),
        sample_u.reshape(b * n, p, num_hypotheses, samples),
        sample_v.reshape(b * n, p, num_hypotheses, samples),
    ).reshape(b, n, p, num_hypotheses, samples)
    residual = observed - reference_patch[:, None, :, None]
    absolute = residual.abs()
    robust = torch.where(
        absolute <= huber_delta,
        0.5 * residual.square(),
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    patch_count = inside.sum(-1)
    view_valid = patch_count >= max(1, math.ceil(samples * 0.7))
    view_cost = (robust * inside.to(robust.dtype)).sum(-1) / patch_count.clamp_min(1)
    view_count = view_valid.sum(1)
    cost = torch.where(view_valid, view_cost, torch.zeros_like(view_cost)).sum(1) / view_count.clamp_min(1)
    center = samples // 2
    parallax_each = ((uu[..., center] - px[:, None, :, None]).square() + (vv[..., center] - py[:, None, :, None]).square()).sqrt()
    parallax = torch.where(view_valid, parallax_each, torch.zeros_like(parallax_each)).sum(1) / view_count.clamp_min(1)
    cost = cost.masked_fill(~((view_count >= min_views) & selected[..., None]), torch.inf)
    best_two, best_indices = cost.topk(2, dim=-1, largest=False)
    best, second, index = best_two[..., 0], best_two[..., 1], best_indices[..., 0]
    rho_best = rho[index]
    parallax_best = parallax.gather(-1, index[..., None])[..., 0]
    views_best = view_count.gather(-1, index[..., None])[..., 0]
    margin = ((second - best).clamp_min(0) / (second.abs() + eps)).nan_to_num()
    photo = torch.exp(-best.nan_to_num(1e6) / max(huber_delta * huber_delta, float(eps)))
    confidence = (
        margin
        * photo
        * (parallax_best / max(min_parallax_px, float(eps))).clamp(0, 1)
        * (views_best / max(n, 1))
    ).clamp(0, 1)
    valid = selected & torch.isfinite(best) & (views_best >= min_views) & (parallax_best >= min_parallax_px) & (confidence >= min_confidence)
    flat = py * w + px
    depth = rho_best.clamp_min(torch.finfo(ref.dtype).tiny).reciprocal()
    return {
        "inverse_depth": _dense(torch.where(valid, rho_best, torch.zeros_like(rho_best)), flat, batch=b, height=h, width=w),
        "depth": _dense(torch.where(valid, depth, torch.zeros_like(depth)), flat, batch=b, height=h, width=w),
        "valid": _dense(valid, flat, batch=b, height=h, width=w, dtype=torch.bool),
        "selected": _dense(selected, flat, batch=b, height=h, width=w, dtype=torch.bool),
        "confidence": _dense(torch.where(valid, confidence, torch.zeros_like(confidence)), flat, batch=b, height=h, width=w),
        "photometric_error": _dense(torch.where(valid, best, torch.zeros_like(best)), flat, batch=b, height=h, width=w),
        "parallax_px": _dense(torch.where(valid, parallax_best, torch.zeros_like(parallax_best)), flat, batch=b, height=h, width=w),
        "gradient": _dense(gradient, flat, batch=b, height=h, width=w),
    }
