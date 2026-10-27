# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Eager dense RGB-D pose normal-equation reduction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import depth_geometry


def _pose(pose: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype, name: str) -> None:
    if pose.shape != (batch, 4, 4) or pose.device != device or pose.dtype != dtype or not torch.isfinite(pose).all():
        raise ValueError(f"{name} must be finite [B,4,4] on depth device/dtype")


def _sample(value: torch.Tensor, grid: torch.Tensor, *, mode: str) -> torch.Tensor:
    return F.grid_sample(value, grid, mode=mode, padding_mode="zeros", align_corners=True)


def _rgb(rgb: torch.Tensor | None, depth: torch.Tensor, name: str) -> None:
    if rgb is None:
        return
    if (
        rgb.shape != (depth.shape[0], 3, depth.shape[2], depth.shape[3])
        or rgb.device != depth.device
        or rgb.dtype != depth.dtype
        or not torch.isfinite(rgb).all()
    ):
        raise ValueError(f"{name} must be finite [B,3,H,W] on its depth device/dtype")


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[:, 0:1] * 0.2126) + (rgb[:, 1:2] * 0.7152) + (rgb[:, 2:3] * 0.0722)


def _normal_equations(
    point: torch.Tensor, normal: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.cross(point, normal, dim=1)
    # Keep the singleton channel axis: removing it lets PyTorch align ``B``
    # with the channel axis of ``weight`` and accidentally cross-broadcast B8.
    terms = (
        normal[:, 0:1],
        normal[:, 1:2],
        normal[:, 2:3],
        rotation[:, 0:1],
        rotation[:, 1:2],
        rotation[:, 2:3],
    )
    b = torch.cat([(weight * residual * term).sum((-2, -1)) for term in terms], dim=1)
    rows = [torch.cat([(weight * left * right).sum((-2, -1)) for right in terms], dim=1) for left in terms]
    return torch.stack(rows, dim=1), b


def dense_track_reduce(
    *,
    source_depth: torch.Tensor | None,
    source_intrinsics: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    reference_depth: torch.Tensor | None,
    reference_intrinsics: torch.Tensor,
    reference_pose_c2w: torch.Tensor,
    source_confidence: torch.Tensor | None = None,
    reference_confidence: torch.Tensor | None = None,
    source_rgb: torch.Tensor | None = None,
    reference_rgb: torch.Tensor | None = None,
    huber_delta: float = 0.05,
    partial_grid: tuple[int, int] = (1, 1),
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Accumulate fixed point-to-plane SE(3) normal equations without a Jacobian map.

    The source/reference contract serves F2F and F2M equally. ``source_pose_c2w``
    is the candidate pose; the fixed reference pose determines the camera in
    which residuals and the left-perturbation Jacobian are evaluated.
    """
    if (source_depth is None) != (reference_depth is None):
        raise ValueError("source_depth and reference_depth must be supplied together")
    if source_depth is None:
        if source_rgb is None or reference_rgb is None:
            raise ValueError("RGB-only tracking requires source_rgb and reference_rgb")
        if validate and (
            source_rgb.ndim != 4
            or source_rgb.shape[1] != 3
            or source_rgb.shape != reference_rgb.shape
            or not source_rgb.is_floating_point()
            or source_rgb.device != reference_rgb.device
            or source_rgb.dtype != reference_rgb.dtype
            or not torch.isfinite(source_rgb).all()
            or not torch.isfinite(reference_rgb).all()
            or source_confidence is not None
            or reference_confidence is not None
            or len(partial_grid) != 2
            or not all(isinstance(value, int) and value > 0 for value in partial_grid)
            or partial_grid[0] > source_rgb.shape[-2]
            or partial_grid[1] > source_rgb.shape[-1]
        ):
            raise ValueError("RGB-only tracking requires matched finite RGB and no depth confidence")
        batch = source_rgb.shape[0]
        tiles = partial_grid[0] * partial_grid[1]
        H = torch.zeros((batch, 6, 6), device=source_rgb.device, dtype=source_rgb.dtype)
        b = torch.zeros((batch, 6), device=source_rgb.device, dtype=source_rgb.dtype)
        count = torch.zeros((batch,), device=source_rgb.device, dtype=torch.int64)
        partial_H = torch.zeros((batch, tiles, 6, 6), device=source_rgb.device, dtype=source_rgb.dtype)
        partial_b = torch.zeros((batch, tiles, 6), device=source_rgb.device, dtype=source_rgb.dtype)
        return {
            "H": H,
            "b": b,
            "geometry_H": H,
            "geometry_b": b,
            "photometric_H": H,
            "photometric_b": b,
            "geometry_inlier_count": count,
            "photometric_inlier_count": count,
            "geometry_residual_abs_mean": torch.zeros((batch,), device=source_rgb.device, dtype=source_rgb.dtype),
            "partial_H": partial_H,
            "partial_b": partial_b,
            "partial_valid": torch.zeros((batch, tiles), device=source_rgb.device, dtype=torch.bool),
        }
    if validate and (
        source_depth.ndim != 4
        or source_depth.shape[1] != 1
        or not source_depth.is_floating_point()
        or reference_depth.ndim != 4
        or reference_depth.shape[1] != 1
        or source_depth.shape[0] != reference_depth.shape[0]
        or reference_depth.device != source_depth.device
        or reference_depth.dtype != source_depth.dtype
        or not torch.isfinite(torch.tensor(huber_delta))
        or huber_delta <= 0
    ):
        raise ValueError("source/reference depth must be floating [B,1,H,W] with positive huber_delta")
    batch = source_depth.shape[0]
    if validate:
        _pose(source_pose_c2w, batch, source_depth.device, source_depth.dtype, "source_pose_c2w")
        _pose(reference_pose_c2w, batch, source_depth.device, source_depth.dtype, "reference_pose_c2w")
    if validate and (source_rgb is None) != (reference_rgb is None):
        raise ValueError("source_rgb and reference_rgb must be supplied together")
    if validate and (
        len(partial_grid) != 2
        or not all(isinstance(value, int) and value > 0 for value in partial_grid)
        or partial_grid[0] > source_depth.shape[-2]
        or partial_grid[1] > source_depth.shape[-1]
    ):
        raise ValueError("partial_grid must be positive and no larger than the source image")
    if validate:
        _rgb(source_rgb, source_depth, "source_rgb")
        _rgb(reference_rgb, reference_depth, "reference_rgb")
    for confidence, depth, name in (
        (source_confidence, source_depth, "source_confidence"),
        (reference_confidence, reference_depth, "reference_confidence"),
    ):
        if validate and confidence is not None and (
            confidence.shape != depth.shape
            or confidence.device != depth.device
            or confidence.dtype != depth.dtype
            or not torch.isfinite(confidence).all()
            or not ((confidence >= 0) & (confidence <= 1)).all()
        ):
            raise ValueError(f"{name} must be finite [0,1] and match its depth")

    source = depth_geometry(source_depth, source_intrinsics, validate=validate)
    reference = depth_geometry(reference_depth, reference_intrinsics, validate=validate)
    source_world = (
        torch.einsum("bij,bjhw->bihw", source_pose_c2w[:, :3, :3], source["xyz_camera"])
        + source_pose_c2w[:, :3, 3, None, None]
    )
    point = torch.einsum(
        "bij,bjhw->bihw",
        reference_pose_c2w[:, :3, :3].transpose(1, 2),
        source_world - reference_pose_c2w[:, :3, 3, None, None],
    )
    z = point[:, 2:3]
    epsilon = torch.finfo(source_depth.dtype).eps
    height, width = reference_depth.shape[-2:]
    fx = reference_intrinsics[:, 0, 0, None, None, None]
    fy = reference_intrinsics[:, 1, 1, None, None, None]
    cx = reference_intrinsics[:, 0, 2, None, None, None]
    cy = reference_intrinsics[:, 1, 2, None, None, None]
    u = fx * point[:, 0:1] / z.clamp_min(epsilon) + cx
    v = fy * point[:, 1:2] / z.clamp_min(epsilon) + cy
    grid = torch.stack((2 * u[:, 0] / max(width - 1, 1) - 1, 2 * v[:, 0] / max(height - 1, 1) - 1), dim=-1)
    reference_xyz = _sample(reference["xyz_camera"], grid, mode="bilinear")
    reference_normals = _sample(reference["normals"], grid, mode="bilinear")
    reference_normals = F.normalize(reference_normals, dim=1, eps=epsilon)
    # Residuals live in the reference camera, while TR1 applies its resulting
    # twist on the right of source c2w (source-camera local coordinates).  Map
    # the reference-space normal/Jacobian into that local basis before forming
    # normal equations.  The old point/reference-normal pairing happened to
    # agree only when the two camera orientations were identical.
    relative_rotation = reference_pose_c2w[:, :3, :3].transpose(1, 2) @ source_pose_c2w[:, :3, :3]
    local_normals = torch.einsum("bji,bjhw->bihw", relative_rotation, reference_normals)
    reference_valid = _sample(
        (reference["valid"] & (reference["geometry_confidence"] > 0)).to(source_depth.dtype), grid, mode="nearest"
    ) > 0.5
    source_weight = source_confidence if source_confidence is not None else source["valid"].to(source_depth.dtype)
    reference_weight = _sample(
        reference_confidence if reference_confidence is not None else reference["geometry_confidence"], grid, mode="bilinear"
    )
    in_bounds = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    support = source["valid"] & reference_valid & torch.isfinite(point).all(1, keepdim=True) & (z > epsilon) & in_bounds
    residual = ((point - reference_xyz) * reference_normals).sum(1, keepdim=True)
    absolute = residual.abs()
    robust = torch.where(absolute <= huber_delta, torch.ones_like(absolute), huber_delta / absolute.clamp_min(epsilon))
    weight = torch.where(support, robust * source_weight * reference_weight, torch.zeros_like(residual))
    H_geometry, b_geometry = _normal_equations(source["xyz_camera"], local_normals, residual, weight)
    count = support.sum((1, 2, 3)).to(torch.int64)
    residual_mean = (absolute * support).sum((1, 2, 3)) / count.to(source_depth.dtype).clamp_min(1)
    zeros_H = torch.zeros_like(H_geometry)
    zeros_b = torch.zeros_like(b_geometry)
    photo_count = torch.zeros_like(count)
    H_photo, b_photo = zeros_H, zeros_b
    photo_point = photo_gradient = photo_residual = photo_weight = None
    if source_rgb is not None and reference_rgb is not None:
        source_luma = _luminance(source_rgb)
        reference_luma = _luminance(reference_rgb)
        gradient_x = F.pad(reference_luma[..., :, 1:] - reference_luma[..., :, :-1], (0, 1, 0, 0))
        gradient_y = F.pad(reference_luma[..., 1:, :] - reference_luma[..., :-1, :], (0, 0, 0, 1))
        ref_luma = _sample(reference_luma, grid, mode="bilinear")
        ref_gradient_x = _sample(gradient_x, grid, mode="bilinear")
        ref_gradient_y = _sample(gradient_y, grid, mode="bilinear")
        spatial_gradient = torch.cat(
            (
                ref_gradient_x * fx / z.clamp_min(epsilon),
                ref_gradient_y * fy / z.clamp_min(epsilon),
                -(ref_gradient_x * fx * point[:, 0:1] + ref_gradient_y * fy * point[:, 1:2])
                / z.square().clamp_min(epsilon),
            ),
            dim=1,
        )
        photo_residual = source_luma - ref_luma
        photo_absolute = photo_residual.abs()
        photo_robust = torch.where(
            photo_absolute <= huber_delta,
            torch.ones_like(photo_absolute),
            huber_delta / photo_absolute.clamp_min(epsilon),
        )
        photo_support = source["valid"] & reference_valid & torch.isfinite(spatial_gradient).all(1, keepdim=True) & (z > epsilon) & in_bounds
        photo_weight = torch.where(
            photo_support,
            photo_robust * source_weight * reference_weight,
            torch.zeros_like(photo_residual),
        )
        local_photo_gradient = torch.einsum("bji,bjhw->bihw", relative_rotation, spatial_gradient)
        H_photo, b_photo = _normal_equations(source["xyz_camera"], local_photo_gradient, photo_residual, photo_weight)
        photo_count = photo_support.sum((1, 2, 3)).to(torch.int64)
        photo_point, photo_gradient = source["xyz_camera"], local_photo_gradient
    rows, cols = partial_grid
    source_height, source_width = source_depth.shape[-2:]
    partial_H: list[torch.Tensor] = []
    partial_b: list[torch.Tensor] = []
    partial_valid: list[torch.Tensor] = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * source_height // rows, (row + 1) * source_height // rows
            x0, x1 = col * source_width // cols, (col + 1) * source_width // cols
            tile = torch.zeros_like(weight, dtype=torch.bool)
            tile[..., y0:y1, x0:x1] = True
            geometry_H, geometry_b = _normal_equations(
                source["xyz_camera"], local_normals, residual, torch.where(tile, weight, torch.zeros_like(weight))
            )
            if photo_point is None or photo_gradient is None or photo_residual is None or photo_weight is None:
                tile_H, tile_b = geometry_H, geometry_b
                tile_count = tile & support
            else:
                photo_H, photo_b = _normal_equations(
                    photo_point,
                    photo_gradient,
                    photo_residual,
                    torch.where(tile, photo_weight, torch.zeros_like(photo_weight)),
                )
                tile_H, tile_b = geometry_H + photo_H, geometry_b + photo_b
                tile_count = tile & (support | (photo_weight > 0))
            partial_H.append(tile_H)
            partial_b.append(tile_b)
            partial_valid.append(tile_count.any((1, 2, 3)))
    partial_H_out = torch.stack(partial_H, dim=1)
    partial_b_out = torch.stack(partial_b, dim=1)
    partial_valid_out = torch.stack(partial_valid, dim=1)
    return {
        "H": partial_H_out.sum(1),
        "b": partial_b_out.sum(1),
        "geometry_H": H_geometry,
        "geometry_b": b_geometry,
        "photometric_H": H_photo,
        "photometric_b": b_photo,
        "geometry_inlier_count": count,
        "photometric_inlier_count": photo_count,
        "geometry_residual_abs_mean": residual_mean,
        "partial_H": partial_H_out,
        "partial_b": partial_b_out,
        "partial_valid": partial_valid_out,
    }
