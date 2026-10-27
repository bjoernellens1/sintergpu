"""Fixed-capacity world-cell reductions."""

from __future__ import annotations

import torch


def spatial_bin_reduce(
    points: torch.Tensor,
    valid: torch.Tensor,
    *,
    voxel_size: float | torch.Tensor,
    origin: torch.Tensor,
    features: torch.Tensor | None = None,
    capacity: int,
) -> dict[str, torch.Tensor]:
    """Reduce valid ``[B,N,3]`` points into lexicographically ordered voxel cells."""
    if points.ndim != 3 or points.shape[-1] != 3 or not points.is_floating_point() or valid.shape != points.shape[:2] or valid.dtype is not torch.bool or valid.device != points.device or capacity < 0:
        raise ValueError("points [B,N,3], matching bool valid, and non-negative capacity are required")
    if features is not None and (features.ndim != 3 or features.shape[:2] != points.shape[:2] or features.device != points.device or not features.is_floating_point()):
        raise ValueError("features must be floating [B,N,F] on the point device")
    size = torch.as_tensor(voxel_size, device=points.device, dtype=points.dtype)
    origin = torch.as_tensor(origin, device=points.device, dtype=points.dtype)
    if size.numel() not in {1, 3} or origin.shape != (3,) or not torch.isfinite(size).all() or (size <= 0).any() or not torch.isfinite(origin).all():
        raise ValueError("voxel_size must be finite positive scalar/[3] and origin finite [3]")
    cells = torch.floor((points - origin) / size).to(torch.int64)
    b = points.shape[0]
    cell_ids = torch.zeros((b, capacity, 3), device=points.device, dtype=torch.int64)
    point_count = torch.zeros((b, capacity), device=points.device, dtype=torch.int64)
    point_sum = torch.zeros((b, capacity, 3), device=points.device, dtype=points.dtype)
    occupied = torch.zeros((b, capacity), device=points.device, dtype=torch.bool)
    feature_mean = torch.zeros((b, capacity, features.shape[-1]), device=points.device, dtype=features.dtype) if features is not None else None
    count = torch.zeros((b,), device=points.device, dtype=torch.int64)
    for batch in range(b):
        selected = valid[batch]
        if not selected.any() or capacity == 0:
            continue
        ids, inverse = torch.unique(cells[batch, selected], dim=0, sorted=True, return_inverse=True)
        retained = min(ids.shape[0], capacity)
        keep = inverse < retained
        inverse = inverse[keep]
        source_points = points[batch, selected][keep]
        cell_ids[batch, :retained] = ids[:retained]
        point_count[batch, :retained].scatter_add_(0, inverse, torch.ones_like(inverse))
        point_sum[batch, :retained].index_add_(0, inverse, source_points)
        occupied[batch, :retained] = point_count[batch, :retained] > 0
        count[batch] = occupied[batch].sum()
        if feature_mean is not None:
            sums = torch.zeros_like(feature_mean[batch, :retained])
            sums.index_add_(0, inverse, features[batch, selected][keep])
            feature_mean[batch, :retained] = sums / point_count[batch, :retained, None].clamp_min(1)
    out = {"cell_ids": cell_ids, "point_count": point_count, "point_sum": point_sum, "valid": occupied, "count": count}
    if feature_mean is not None:
        out["feature_mean"] = feature_mean
    return out
