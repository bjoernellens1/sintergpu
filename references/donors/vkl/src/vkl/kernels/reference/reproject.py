from __future__ import annotations

import torch

from .geometry import depth_geometry


def reproject_rgbd(source_depth: torch.Tensor, source_intrinsics: torch.Tensor, source_pose_c2w: torch.Tensor, target_intrinsics: torch.Tensor, target_pose_c2w: torch.Tensor, target_depth: torch.Tensor | None = None, *, target_hw: tuple[int, int] | None = None, consistency_threshold: float = 0.05) -> dict[str, torch.Tensor]:
    """Project source depth into a target camera; visibility is in-bounds and front-facing."""
    tensors = (source_depth, source_intrinsics, source_pose_c2w, target_intrinsics, target_pose_c2w)
    if source_depth.ndim != 4 or source_depth.shape[1] != 1 or source_intrinsics.shape != (source_depth.shape[0], 3, 3) or source_pose_c2w.shape != (source_depth.shape[0], 4, 4) or target_intrinsics.shape != source_intrinsics.shape or target_pose_c2w.shape != source_pose_c2w.shape:
        raise ValueError("source and target geometry must have matching batch shapes")
    if any(t.device != source_depth.device or t.dtype != source_depth.dtype for t in tensors):
        raise ValueError("geometry tensors must share device and dtype")
    if consistency_threshold <= 0:
        raise ValueError("consistency_threshold must be positive")
    geom = depth_geometry(source_depth, source_intrinsics)
    xyz = geom["xyz_camera"]
    world = torch.einsum("bij,bjhw->bihw", source_pose_c2w[:, :3, :3], xyz) + source_pose_c2w[:, :3, 3, None, None]
    rel = torch.einsum("bij,bjhw->bihw", target_pose_c2w[:, :3, :3].transpose(1, 2), world - target_pose_c2w[:, :3, 3, None, None])
    raw_z = rel[:, 2]
    z = raw_z.clamp_min(torch.finfo(rel.dtype).eps)
    u = target_intrinsics[:, None, None, 0, 0] * rel[:, 0] / z + target_intrinsics[:, None, None, 0, 2]
    v = target_intrinsics[:, None, None, 1, 1] * rel[:, 1] / z + target_intrinsics[:, None, None, 1, 2]
    if target_hw is None and target_depth is None:
        raise ValueError("target_hw is required when target_depth is absent")
    target_h, target_w = target_hw or source_depth.shape[-2:]
    if target_depth is not None:
        if target_depth.ndim != 4 or target_depth.shape[:2] != source_depth.shape[:2]:
            raise ValueError("target_depth must have shape [B,1,H,W]")
        if target_depth.device != source_depth.device or target_depth.dtype != source_depth.dtype:
            raise ValueError("target_depth must share device and dtype")
        depth_hw = target_depth.shape[-2:]
        if target_hw is not None and tuple(target_hw) != tuple(depth_hw):
            raise ValueError("target_hw must match target_depth spatial shape")
        target_h, target_w = depth_hw
    visible = geom["valid"] & (raw_z[:, None] > 0) & (u[:, None] >= 0) & (u[:, None] <= target_w - 1) & (v[:, None] >= 0) & (v[:, None] <= target_h - 1)
    displacement = torch.sqrt((u - torch.arange(source_depth.shape[-1], device=u.device)[None, None, :]) ** 2 + (v - torch.arange(source_depth.shape[-2], device=v.device)[:, None]) ** 2)
    visible_pixels = visible[:, 0]
    counts = visible_pixels.sum(dim=(-2, -1))
    parallax = (displacement * visible_pixels).sum(dim=(-2, -1)) / counts.clamp_min(1)
    out = {
        "projected_xy": torch.stack((u, v), dim=1),
        "projected_depth": raw_z[:, None],
        "visible": visible,
        "displacement": displacement[:, None],
        "overlap": visible_pixels.to(source_depth.dtype).mean(dim=(-2, -1)),
        "parallax": parallax,
    }
    if target_depth is not None:
        grid = torch.stack((2 * u / max(target_w - 1, 1) - 1, 2 * v / max(target_h - 1, 1) - 1), dim=-1)
        sample = torch.nn.functional.grid_sample(target_depth, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        target_valid = torch.isfinite(sample) & (sample > 0)
        consistency_valid = visible & target_valid
        consistency = torch.where(consistency_valid, (sample - raw_z[:, None]).abs(), torch.zeros_like(sample))
        out["depth_consistency"] = consistency
        out["depth_consistency_valid"] = consistency_valid
        out["occluded"] = consistency_valid & (sample < (raw_z[:, None] - consistency_threshold))
        out["occlusion_valid"] = consistency_valid
    return out


def rasterize_reprojection(
    source_depth: torch.Tensor,
    source_confidence: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_pose_c2w: torch.Tensor,
    *,
    target_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Z-buffer source depth/confidence into target pixels using D4 geometry."""
    if source_confidence.shape != source_depth.shape or source_confidence.device != source_depth.device or source_confidence.dtype != source_depth.dtype or not torch.isfinite(source_confidence).all() or (source_confidence < 0).any() or (source_confidence > 1).any():
        raise ValueError("source_confidence must be finite [0,1] and match source_depth")
    target_h, target_w = target_hw
    if target_h < 1 or target_w < 1:
        raise ValueError("target_hw must be positive")
    projected = reproject_rgbd(
        source_depth, source_intrinsics, source_pose_c2w, target_intrinsics, target_pose_c2w, target_hw=target_hw,
    )
    b = source_depth.shape[0]
    xy = projected["projected_xy"]
    u = xy[:, 0].round().long().clamp(0, target_w - 1)
    v = xy[:, 1].round().long().clamp(0, target_h - 1)
    visible = projected["visible"][:, 0]
    index = (v * target_w + u).reshape(b, -1)
    z = projected["projected_depth"][:, 0].reshape(b, -1)
    source_conf = source_confidence[:, 0].reshape(b, -1)
    visible_flat = visible.reshape(b, -1)
    depth_flat = torch.full((b, target_h * target_w), torch.inf, device=source_depth.device, dtype=source_depth.dtype)
    depth_flat.scatter_reduce_(1, index, torch.where(visible_flat, z, torch.full_like(z, torch.inf)), reduce="amin", include_self=True)
    winning_depth = depth_flat.gather(1, index)
    winning = visible_flat & ((z - winning_depth).abs() <= torch.finfo(source_depth.dtype).eps * winning_depth.abs().clamp_min(1))
    confidence_flat = torch.zeros_like(depth_flat)
    confidence_flat.scatter_reduce_(1, index, torch.where(winning, source_conf, torch.zeros_like(source_conf)), reduce="amax", include_self=True)
    support_flat = torch.zeros((b, target_h * target_w), device=source_depth.device, dtype=torch.int64)
    support_flat.scatter_add_(1, index, winning.to(torch.int64))
    valid = torch.isfinite(depth_flat)
    return {
        "depth": torch.where(valid, depth_flat, torch.zeros_like(depth_flat)).reshape(b, 1, target_h, target_w),
        "valid": valid.reshape(b, 1, target_h, target_w),
        "confidence": confidence_flat.reshape(b, 1, target_h, target_w),
        "support_count": support_flat.reshape(b, 1, target_h, target_w),
    }


depth_reproject = reproject_rgbd
