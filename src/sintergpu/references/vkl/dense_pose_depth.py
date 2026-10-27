# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Dense fixed-pose plane-sweep depth reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .depth_cost_volume import aggregate_depth_cost_volume


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[:, :1].float() if rgb.shape[1] == 1 else (rgb[:, :3].float() * torch.tensor([0.299, 0.587, 0.114], device=rgb.device)[None, :, None, None]).sum(1, keepdim=True)


def dense_pose_depth(
    rgb_reference: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    pose_reference: torch.Tensor,
    pose_neighbors: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    num_hypotheses: int = 64,
    min_depth: float = 0.25,
    max_depth: float = 20.0,
    patch_radius: int = 1,
    min_views: int = 1,
    min_parallax_px: float = 0.5,
    min_confidence: float = 0.01,
    max_photometric_error: float = float("inf"),
    max_view_cost_std: float = float("inf"),
    spatial_cost_radius: int = 0,
    spatial_edge_threshold: float = 0.15,
    photometric_metric: str = "sad",
    view_reduce: str = "mean",
    cost_aggregation: str = "none",
    aggregation_smoothness: float = 0.05,
    aggregation_edge_scale: float = 0.10,
    aggregation_scale: int = 1,
    inverse_depth_center: torch.Tensor | None = None,
    inverse_depth_radius: float | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate semi-dense depth by streaming plane-sweep hypotheses on GPU."""
    if rgb_reference.ndim != 4 or rgb_reference.shape[1] not in {1, 3} or not rgb_reference.is_floating_point() or rgb_neighbors.ndim != 5:
        raise ValueError("reference must be floating [B,C,H,W] and neighbors [B,N,C,H,W]")
    b, _, h, w = rgb_reference.shape
    if rgb_neighbors.shape[0] != b or rgb_neighbors.shape[2] not in {1, 3} or rgb_neighbors.shape[-2:] != (h, w) or pose_reference.shape != (b, 4, 4) or pose_neighbors.shape != (b, rgb_neighbors.shape[1], 4, 4) or intrinsics.shape != (b, 3, 3):
        raise ValueError("neighbor, pose, and intrinsics shapes must match reference")
    if any(t.device != rgb_reference.device or t.dtype != rgb_reference.dtype for t in (rgb_neighbors, pose_reference, pose_neighbors, intrinsics)) or num_hypotheses < 2 or patch_radius < 0 or spatial_cost_radius < 0 or not (0 < min_depth < max_depth) or max_photometric_error < 0 or max_view_cost_std < 0 or spatial_edge_threshold < 0 or photometric_metric not in {"sad", "zncc"} or view_reduce not in {"mean", "median"} or cost_aggregation not in {"none", "sgm"} or aggregation_smoothness < 0 or aggregation_edge_scale <= 0 or aggregation_scale < 1:
        raise ValueError("all tensors must share device/dtype and depth search must be valid")
    if (inverse_depth_center is None) != (inverse_depth_radius is None):
        raise ValueError("local inverse-depth refinement needs a matching center and positive radius")
    if inverse_depth_center is not None and (
        inverse_depth_center.shape != (b, 1, h, w)
        or inverse_depth_center.device != rgb_reference.device
        or inverse_depth_center.dtype != rgb_reference.dtype
        or inverse_depth_radius is None
        or inverse_depth_radius <= 0
    ):
        raise ValueError("local inverse-depth refinement needs a matching center and positive radius")
    prior_valid = (
        torch.isfinite(inverse_depth_center)
        & (inverse_depth_center > 0)
        if inverse_depth_center is not None
        else None
    )
    n = rgb_neighbors.shape[1]
    reference = _gray(rgb_reference)
    neighbors = _gray(rgb_neighbors.reshape(b * n, rgb_neighbors.shape[2], h, w)).reshape(b, n, 1, h, w)
    offsets = torch.tensor(
        [(dx, dy) for dy in range(-patch_radius, patch_radius + 1) for dx in range(-patch_radius, patch_radius + 1)],
        device=reference.device,
        dtype=reference.dtype,
    )
    samples = offsets.shape[0]
    reference_patch = F.unfold(reference, 2 * patch_radius + 1, padding=patch_radius).transpose(1, 2).reshape(b, h, w, samples)
    if spatial_cost_radius:
        spatial_size = 2 * spatial_cost_radius + 1
        spatial_gray = F.unfold(reference, spatial_size, padding=spatial_cost_radius).reshape(b, spatial_size * spatial_size, h, w)
        spatial_inside = F.unfold(torch.ones_like(reference), spatial_size, padding=spatial_cost_radius).reshape(b, spatial_size * spatial_size, h, w)
        spatial_weight = ((spatial_gray - reference).abs() <= spatial_edge_threshold) * spatial_inside
    min_patch_samples = max(1, (samples * 7 + 9) // 10)
    y, x = torch.meshgrid(torch.arange(h, device=reference.device, dtype=reference.dtype), torch.arange(w, device=reference.device, dtype=reference.dtype), indexing="ij")
    rays = torch.stack(((x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0], (y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1], torch.ones((b, h, w), device=reference.device, dtype=reference.dtype)), 1)
    rel_r = pose_neighbors[:, :, :3, :3].transpose(-1, -2) @ pose_reference[:, None, :3, :3]
    rel_t = (pose_neighbors[:, :, :3, :3].transpose(-1, -2) @ (pose_reference[:, None, :3, 3] - pose_neighbors[:, :, :3, 3])[..., None])[..., 0]
    best = torch.full((b, h, w), torch.inf, device=reference.device, dtype=reference.dtype)
    second = best.clone()
    best_rho = torch.zeros_like(best)
    best_views = torch.zeros_like(best)
    best_parallax = torch.zeros_like(best)
    best_view_cost_std = torch.full_like(best, torch.inf)
    rho_values = (
        torch.linspace(-inverse_depth_radius, inverse_depth_radius, num_hypotheses, device=reference.device, dtype=reference.dtype)
        if inverse_depth_center is not None
        else torch.linspace(1 / max_depth, 1 / min_depth, num_hypotheses, device=reference.device, dtype=reference.dtype)
    )
    if cost_aggregation == "sgm":
        candidate_cost = torch.full((b, num_hypotheses, h, w), torch.inf, device=reference.device, dtype=reference.dtype)
        candidate_views = torch.zeros_like(candidate_cost)
        candidate_parallax = torch.zeros_like(candidate_cost)
        candidate_std = torch.full_like(candidate_cost, torch.inf)
    for rho_index, rho in enumerate(rho_values):
        rho_field = (
            (inverse_depth_center[:, 0] + rho).clamp(1 / max_depth, 1 / min_depth)
            if inverse_depth_center is not None
            else rho
        )
        xyz = rays / rho_field[:, None] if inverse_depth_center is not None else rays / rho_field
        transformed = torch.einsum("bnij,bjhw->bnihw", rel_r, xyz) + rel_t[..., None, None]
        z = transformed[:, :, 2]
        u = intrinsics[:, None, 0, 0, None, None] * transformed[:, :, 0] / z.clamp_min(torch.finfo(z.dtype).eps) + intrinsics[:, None, 0, 2, None, None]
        v = intrinsics[:, None, 1, 1, None, None] * transformed[:, :, 1] / z.clamp_min(torch.finfo(z.dtype).eps) + intrinsics[:, None, 1, 2, None, None]
        uu = u[..., None] + offsets[:, 0]
        vv = v[..., None] + offsets[:, 1]
        inside = (z[..., None] > 0) & (uu >= 0) & (uu <= w - 1) & (vv >= 0) & (vv <= h - 1)
        grid = torch.stack((2 * uu / max(w - 1, 1) - 1, 2 * vv / max(h - 1, 1) - 1), -1).reshape(b * n, h, w * samples, 2)
        observed = F.grid_sample(neighbors.reshape(b * n, 1, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(b, n, h, w, samples)
        patch_count = inside.sum(-1)
        view_valid = patch_count >= min_patch_samples
        if photometric_metric == "sad":
            view_cost = ((observed - reference_patch[:, None]).abs() * inside).sum(-1) / patch_count.clamp_min(1)
        else:
            weight = inside.to(observed.dtype)
            reference_mean = (reference_patch[:, None] * weight).sum(-1, keepdim=True) / patch_count[..., None].clamp_min(1)
            observed_mean = (observed * weight).sum(-1, keepdim=True) / patch_count[..., None].clamp_min(1)
            reference_centered = (reference_patch[:, None] - reference_mean) * weight
            observed_centered = (observed - observed_mean) * weight
            correlation = (reference_centered * observed_centered).sum(-1) / (reference_centered.square().sum(-1) * observed_centered.square().sum(-1)).sqrt().clamp_min(1e-6)
            view_cost = (1 - correlation).clamp(0, 2) * 0.5
        views = view_valid.sum(1)
        cost = (view_cost * view_valid).sum(1) / views.clamp_min(1) if view_reduce == "mean" else torch.nanmedian(view_cost.masked_fill(~view_valid, torch.nan), dim=1).values.nan_to_num(float("inf"))
        if spatial_cost_radius:
            cost_patch = F.unfold(cost[:, None], spatial_size, padding=spatial_cost_radius).reshape(b, spatial_size * spatial_size, h, w)
            cost = (cost_patch * spatial_weight).sum(1) / spatial_weight.sum(1).clamp_min(1)
        view_cost_std = (((view_cost - cost[:, None]).square() * view_valid).sum(1) / views.clamp_min(1)).sqrt()
        center_parallax = ((u - x).square() + (v - y).square()).sqrt()
        parallax = (center_parallax * view_valid).sum(1) / views.clamp_min(1)
        usable = views >= min_views
        if cost_aggregation == "sgm":
            candidate_cost[:, rho_index] = torch.where(usable, cost, torch.full_like(cost, torch.inf))
            candidate_views[:, rho_index] = views
            candidate_parallax[:, rho_index] = parallax
            candidate_std[:, rho_index] = view_cost_std
        else:
            improved = usable & (cost < best)
            second = torch.where(improved, best, torch.minimum(second, cost))
            best = torch.where(improved, cost, best)
            best_rho = torch.where(
                improved,
                rho_field if inverse_depth_center is not None else torch.full_like(best, rho),
                best_rho,
            )
            best_views, best_parallax = torch.where(improved, views, best_views), torch.where(improved, parallax, best_parallax)
            best_view_cost_std = torch.where(improved, view_cost_std, best_view_cost_std)
    if cost_aggregation == "sgm":
        aggregated = aggregate_depth_cost_volume(candidate_cost, rgb_reference, smoothness_penalty=aggregation_smoothness, edge_scale=aggregation_edge_scale, aggregation_scale=aggregation_scale)
        best_pair, best_index = torch.topk(aggregated, k=2, dim=1, largest=False)
        best, second = best_pair[:, 0], best_pair[:, 1]
        best_rho = (
            (inverse_depth_center[:, 0] + rho_values[best_index[:, 0]]).clamp(1 / max_depth, 1 / min_depth)
            if inverse_depth_center is not None
            else rho_values[best_index[:, 0]]
        )
        gather_index = best_index[:, :1]
        best_raw_cost = candidate_cost.gather(1, gather_index)[:, 0]
        best_views = candidate_views.gather(1, gather_index)[:, 0]
        best_parallax = candidate_parallax.gather(1, gather_index)[:, 0]
        best_view_cost_std = candidate_std.gather(1, gather_index)[:, 0]
    else:
        best_raw_cost = best
    margin = ((second - best).clamp_min(0) / (second.abs() + 1e-6)).nan_to_num()
    confidence = (margin * (best_parallax / min_parallax_px).clamp(0, 1) * (best_views / n)).clamp(0, 1)
    valid = torch.isfinite(best) & torch.isfinite(best_raw_cost) & (best_views >= min_views) & (best_parallax >= min_parallax_px) & (confidence >= min_confidence) & (best_raw_cost <= max_photometric_error) & (best_view_cost_std <= max_view_cost_std)
    if prior_valid is not None:
        valid &= prior_valid[:, 0]
    depth = best_rho.clamp_min(torch.finfo(best_rho.dtype).eps).reciprocal()
    return {"depth": torch.where(valid, depth[:, None], torch.zeros_like(depth[:, None])), "inverse_depth": torch.where(valid, best_rho[:, None], torch.zeros_like(best_rho[:, None])), "valid": valid[:, None], "confidence": torch.where(valid, confidence[:, None], torch.zeros_like(confidence[:, None])), "photometric_error": torch.where(valid, best_raw_cost[:, None], torch.zeros_like(best_raw_cost[:, None])), "view_cost_std": torch.where(valid, best_view_cost_std[:, None], torch.zeros_like(best_view_cost_std[:, None])), "support_views": best_views[:, None], "parallax_px": best_parallax[:, None]}
