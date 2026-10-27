"""Conservative point-cloud voxel and surface-candidate primitives."""

from __future__ import annotations

import torch


def voxelize_points(
    xyz: torch.Tensor,
    voxel_size: float,
    *,
    valid: torch.Tensor | None = None,
    rgb: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compact valid [B,N,3] points into world voxels without cross-batch merging."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or not xyz.is_floating_point():
        raise ValueError("xyz must be floating [B,N,3]")
    if not isinstance(voxel_size, (float, int)) or isinstance(voxel_size, bool) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    batch, points, _ = xyz.shape
    if valid is None:
        valid = torch.ones((batch, points), dtype=torch.bool, device=xyz.device)
    elif valid.shape != (batch, points) or valid.dtype != torch.bool or valid.device != xyz.device:
        raise ValueError("valid must be bool [B,N] on xyz device")
    if rgb is not None and (
        rgb.shape != xyz.shape or rgb.device != xyz.device or rgb.dtype != xyz.dtype or not torch.isfinite(rgb).all()
    ):
        raise ValueError("rgb must be finite [B,N,3] on xyz device/dtype")
    if confidence is not None and (
        confidence.shape != (batch, points)
        or confidence.device != xyz.device
        or confidence.dtype != xyz.dtype
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError("confidence must be finite [B,N] in [0,1]")

    selected = valid & torch.isfinite(xyz).all(-1)
    batch_index = torch.arange(batch, device=xyz.device, dtype=torch.int64)[:, None].expand(-1, points)
    selected_xyz = xyz[selected]
    selected_batch = batch_index[selected]
    if selected_xyz.numel() == 0:
        empty = torch.empty((0,), device=xyz.device, dtype=torch.int64)
        return {
            "xyz": xyz.new_empty((0, 3)),
            "voxel_coords": torch.empty((0, 3), device=xyz.device, dtype=torch.int64),
            "batch_index": empty,
            "count": empty,
            **({"rgb": xyz.new_empty((0, 3))} if rgb is not None else {}),
            **({"confidence": xyz.new_empty((0,))} if confidence is not None else {}),
        }

    coords = torch.floor(selected_xyz / voxel_size).to(torch.int64)
    cells, inverse, count = torch.unique(
        torch.cat((selected_batch[:, None], coords), -1), dim=0, return_inverse=True, return_counts=True
    )
    voxel_count = count.shape[0]
    sums = xyz.new_zeros((voxel_count, 3)).index_add_(0, inverse, selected_xyz)
    out = {
        "xyz": sums / count[:, None].to(xyz.dtype),
        "voxel_coords": cells[:, 1:],
        "batch_index": cells[:, 0],
        "count": count,
    }
    if rgb is not None:
        out["rgb"] = rgb.new_zeros((voxel_count, 3)).index_add_(0, inverse, rgb[selected]) / count[:, None].to(rgb.dtype)
    if confidence is not None:
        out["confidence"] = confidence.new_zeros((voxel_count,)).index_add_(0, inverse, confidence[selected]) / count.to(confidence.dtype)
    return out


def world_voxel_reduce(
    xyz: torch.Tensor,
    source_ids: torch.Tensor,
    voxel_size: float,
    *,
    valid: torch.Tensor | None = None,
    rgb: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    min_source_support: int = 1,
) -> dict[str, torch.Tensor]:
    """Compact world voxels and retain only evidence from distinct source IDs.

    Source IDs are scoped to each batch item. Repeated pixels from one source
    contribute to voxel averages but count once toward `source_support_count`.
    """
    if (
        source_ids.shape != xyz.shape[:2]
        or source_ids.dtype != torch.int64
        or source_ids.device != xyz.device
    ):
        raise ValueError("source_ids must be int64 [B,N] on xyz device")
    if not isinstance(min_source_support, int) or isinstance(min_source_support, bool) or min_source_support < 1:
        raise ValueError("min_source_support must be a positive integer")
    out = voxelize_points(
        xyz, voxel_size, valid=valid, rgb=rgb, confidence=confidence
    )
    batch, points, _ = xyz.shape
    if valid is None:
        valid = torch.ones((batch, points), dtype=torch.bool, device=xyz.device)
    selected = valid & torch.isfinite(xyz).all(-1)
    if not selected.any():
        out["source_support_count"] = out["count"]
        out["valid"] = torch.zeros_like(out["count"], dtype=torch.bool)
        return out
    batch_index = torch.arange(batch, device=xyz.device, dtype=torch.int64)[:, None].expand(-1, points)
    voxel_keys = torch.cat(
        (
            batch_index[selected][:, None],
            torch.floor(xyz[selected] / voxel_size).to(torch.int64),
        ),
        -1,
    )
    source_cells = torch.unique(
        torch.cat((voxel_keys, source_ids[selected][:, None]), -1), dim=0
    )
    support_cells, source_support_count = torch.unique(
        source_cells[:, :4], dim=0, return_counts=True
    )
    output_cells = torch.cat((out["batch_index"][:, None], out["voxel_coords"]), -1)
    if not torch.equal(output_cells, support_cells):
        raise RuntimeError("voxel compaction ordering changed")
    out["source_support_count"] = source_support_count
    out["valid"] = source_support_count >= min_source_support
    return out


def fit_local_planes(
    xyz: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fit one least-squares plane per point from explicit fixed-size neighborhoods."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or not xyz.is_floating_point():
        raise ValueError("xyz must be floating [B,N,3]")
    batch, points, _ = xyz.shape
    if (
        neighbor_indices.ndim != 3
        or neighbor_indices.shape[:2] != (batch, points)
        or neighbor_indices.shape[-1] < 3
        or neighbor_indices.dtype != torch.int64
        or neighbor_indices.device != xyz.device
        or (neighbor_indices < 0).any()
        or (neighbor_indices >= points).any()
    ):
        raise ValueError("neighbor_indices must be in-range int64 [B,N,K] with K >= 3")
    if valid is None:
        valid = torch.ones((batch, points), dtype=torch.bool, device=xyz.device)
    elif valid.shape != (batch, points) or valid.dtype != torch.bool or valid.device != xyz.device:
        raise ValueError("valid must be bool [B,N] on xyz device")

    neighbors = xyz[
        torch.arange(batch, device=xyz.device)[:, None, None],
        neighbor_indices,
    ]
    neighbor_valid = valid[
        torch.arange(batch, device=xyz.device)[:, None, None],
        neighbor_indices,
    ] & torch.isfinite(neighbors).all(-1)
    weight = neighbor_valid.to(xyz.dtype)
    count = weight.sum(-1)
    centroid = (neighbors * weight[..., None]).sum(-2) / count[..., None].clamp_min(1)
    delta = neighbors - centroid[..., None, :]
    covariance = torch.einsum("bnki,bnkj,bnk->bnij", delta, delta, weight) / count[..., None, None].clamp_min(1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[..., :, 0]
    plane_valid = valid & (count >= 3) & (eigenvalues[..., 1] > torch.finfo(xyz.dtype).eps)
    residual = torch.sqrt(((delta * normal[..., None, :]).sum(-1).square() * weight).sum(-1) / count.clamp_min(1))
    normal = torch.where(plane_valid[..., None], normal, torch.zeros_like(normal))
    offset = -(normal * centroid).sum(-1)
    return {
        "normal": normal,
        "offset": torch.where(plane_valid, offset, torch.zeros_like(offset)),
        "residual": torch.where(plane_valid, residual, torch.zeros_like(residual)),
        "valid": plane_valid,
        "neighbor_count": count.to(torch.int64),
    }


def voxel_surface_mesh(
    voxel_coords: torch.Tensor,
    voxel_size: float,
    *,
    batch_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract exposed faces of occupied voxels as a triangle-mesh candidate.

    This is the closed boundary of occupied voxels, not surface completion. Sparse
    or open observation support therefore remains sparse or open at its own scale.
    """
    if voxel_coords.ndim != 2 or voxel_coords.shape[-1] != 3 or voxel_coords.dtype != torch.int64:
        raise ValueError("voxel_coords must be int64 [M,3]")
    if not isinstance(voxel_size, (float, int)) or isinstance(voxel_size, bool) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    count = voxel_coords.shape[0]
    if batch_index is None:
        batch_index = torch.zeros(count, device=voxel_coords.device, dtype=torch.int64)
    elif batch_index.shape != (count,) or batch_index.dtype != torch.int64 or batch_index.device != voxel_coords.device:
        raise ValueError("batch_index must be int64 [M] on voxel device")
    if count == 0:
        return {
            "vertices": torch.empty((0, 3), device=voxel_coords.device, dtype=torch.float32),
            "faces": torch.empty((0, 3), device=voxel_coords.device, dtype=torch.int64),
            "vertex_batch_index": batch_index,
            "face_batch_index": batch_index,
        }

    cells, inverse = torch.unique(torch.cat((batch_index[:, None], voxel_coords), -1), dim=0, return_inverse=True)
    source = torch.unique(inverse)
    cells = cells[source]
    batch_index, voxel_coords = cells[:, 0], cells[:, 1:]
    directions = torch.tensor(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        device=voxel_coords.device,
        dtype=torch.int64,
    )
    neighbors = torch.cat((batch_index[:, None, None].expand(-1, 6, 1), voxel_coords[:, None] + directions), -1)
    _, membership = torch.unique(
        torch.cat((cells[:, None], neighbors), 1).reshape(-1, 4), dim=0, return_inverse=True
    )
    membership = membership.reshape(-1, 7)
    source_ids = membership[:, 0]
    neighbor_ids = membership[:, 1:]
    sorted_source = source_ids.sort().values
    position = torch.searchsorted(sorted_source, neighbor_ids.contiguous())
    present = (position < sorted_source.numel()) & (sorted_source[position.clamp_max(sorted_source.numel() - 1)] == neighbor_ids)
    exposed_voxels, exposed_faces = (~present).nonzero(as_tuple=True)
    corners = torch.tensor(
        [[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 2, 1], [4, 5, 6, 7]],
        device=voxel_coords.device,
        dtype=torch.int64,
    )
    corner_offsets = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        device=voxel_coords.device,
        dtype=torch.int64,
    )
    face_vertices = voxel_coords[exposed_voxels, None] + corner_offsets[corners[exposed_faces]]
    vertex_cells, vertex_inverse = torch.unique(
        torch.cat((batch_index[exposed_voxels, None, None].expand(-1, 4, 1), face_vertices), -1).reshape(-1, 4),
        dim=0,
        return_inverse=True,
    )
    quads = vertex_inverse.reshape(-1, 4)
    faces = torch.cat((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    face_batch = batch_index[exposed_voxels].repeat(2)
    return {
        "vertices": vertex_cells[:, 1:].to(torch.float32) * voxel_size,
        "faces": faces,
        "vertex_batch_index": vertex_cells[:, 0],
        "face_batch_index": face_batch,
    }
