# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Sparse RGB-D validation and geometry for detector-selected image points."""

from __future__ import annotations

import torch


def _points(depth: torch.Tensor, pixels_xy: torch.Tensor) -> tuple[int, int]:
    if depth.ndim != 4 or depth.shape[1] != 1 or not depth.is_floating_point():
        raise ValueError("depth must be floating [B,1,H,W]")
    if (
        pixels_xy.ndim != 3
        or pixels_xy.shape[0] != depth.shape[0]
        or pixels_xy.shape[-1] != 2
        or pixels_xy.dtype != torch.int64
        or pixels_xy.device != depth.device
    ):
        raise ValueError("pixels_xy must be int64 [B,P,2] on depth device")
    return depth.shape[0], pixels_xy.shape[1]


def _intrinsics(value: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> None:
    if value.shape != (batch, 3, 3) or value.device != device or value.dtype != dtype:
        raise ValueError("intrinsics must be [B,3,3] on geometry device/dtype")
    if (
        not torch.isfinite(value).all()
        or (value[:, 0, 0] <= 0).any()
        or (value[:, 1, 1] <= 0).any()
    ):
        raise ValueError("intrinsics must be finite with positive focal lengths")


def _pose(value: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> None:
    if value.shape != (batch, 4, 4) or value.device != device or value.dtype != dtype:
        raise ValueError("pose_c2w must be [B,4,4] on geometry device/dtype")
    if not torch.isfinite(value).all():
        raise ValueError("pose_c2w must be finite")
    rotation = value[:, :3, :3]
    identity = torch.eye(3, device=device, dtype=dtype).expand(batch, -1, -1)
    if (
        not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=1e-4)
        or (torch.linalg.det(rotation) <= 0).any()
        or not torch.allclose(
            value[:, 3],
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(batch, -1),
            atol=1e-6,
            rtol=0,
        )
    ):
        raise ValueError("pose_c2w must be a rigid homogeneous camera-to-world transform")


def sparse_depth_validate(
    depth: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    pixels_valid: torch.Tensor | None = None,
    max_relative_dispersion: float = 0.1,
    min_neighbors: int = 5,
) -> dict[str, torch.Tensor]:
    """Validate sensor depth only at supplied pixels using a 3x3 local stencil."""
    batch, points = _points(depth, pixels_xy)
    if max_relative_dispersion < 0 or not 1 <= min_neighbors <= 9:
        raise ValueError("depth-validation thresholds must be valid")
    if pixels_valid is None:
        pixels_valid = torch.ones((batch, points), dtype=torch.bool, device=depth.device)
    elif pixels_valid.shape != (batch, points) or pixels_valid.dtype != torch.bool or pixels_valid.device != depth.device:
        raise ValueError("pixels_valid must be bool [B,P] on depth device")

    _, _, height, width = depth.shape
    x, y = pixels_xy[..., 0], pixels_xy[..., 1]
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    offsets = torch.tensor(
        [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        device=depth.device,
        dtype=torch.int64,
    )
    raw_x = x[..., None] + offsets[:, 0]
    raw_y = y[..., None] + offsets[:, 1]
    stencil_in_bounds = (
        (raw_x >= 0)
        & (raw_x < width)
        & (raw_y >= 0)
        & (raw_y < height)
        & in_bounds[..., None]
    )
    xx = raw_x.clamp(0, width - 1)
    yy = raw_y.clamp(0, height - 1)
    stencil = (
        depth[:, 0]
        .reshape(batch, -1)
        .gather(1, (yy * width + xx).reshape(batch, -1))
        .reshape(batch, points, 9)
    )
    stencil_valid = stencil_in_bounds & torch.isfinite(stencil) & (stencil > 0)
    count = stencil_valid.sum(-1)
    center_index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    center = depth[:, 0].reshape(batch, -1).gather(1, center_index)
    center_valid = pixels_valid & in_bounds & torch.isfinite(center) & (center > 0)
    median = torch.nanmedian(stencil.masked_fill(~stencil_valid, torch.nan), dim=-1).values
    absolute = (stencil - median[..., None]).abs().masked_fill(~stencil_valid, torch.nan)
    dispersion = torch.nanmedian(absolute, dim=-1).values / median.clamp_min(torch.finfo(depth.dtype).eps)
    local_min = stencil.masked_fill(~stencil_valid, torch.inf).amin(-1)
    local_max = stencil.masked_fill(~stencil_valid, -torch.inf).amax(-1)
    edge_risk = (local_max - local_min) / median.clamp_min(torch.finfo(depth.dtype).eps)
    valid = (
        center_valid
        & (count >= min_neighbors)
        & torch.isfinite(dispersion)
        & torch.isfinite(edge_risk)
        & (dispersion <= max_relative_dispersion)
        & (edge_risk <= max_relative_dispersion)
    )
    confidence = (count.to(depth.dtype) / 9) * (
        1 - torch.maximum(dispersion, edge_risk).clamp(0, 1)
    )
    return {
        "depth": torch.where(valid, center, torch.zeros_like(center)),
        "valid": valid,
        "local_dispersion": torch.nan_to_num(dispersion),
        "edge_risk": torch.nan_to_num(edge_risk),
        "neighbor_count": count,
        "confidence": torch.where(valid, confidence, torch.zeros_like(confidence)),
    }


def sparse_rgbd_observation(
    pixels_xy: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_c2w: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Convert sparse pixel/depth observations into camera and optional world XYZ."""
    if depth.ndim != 2 or not depth.is_floating_point():
        raise ValueError("depth must be floating [B,P]")
    batch, points = depth.shape
    if (
        pixels_xy.shape != (batch, points, 2)
        or pixels_xy.dtype != torch.int64
        or pixels_xy.device != depth.device
    ):
        raise ValueError("pixels_xy must be int64 [B,P,2] on depth device")
    _intrinsics(intrinsics, batch, depth.device, depth.dtype)
    if pose_c2w is not None:
        _pose(pose_c2w, batch, depth.device, depth.dtype)

    valid = torch.isfinite(depth) & (depth > 0)
    u = pixels_xy[..., 0].to(depth.dtype)
    v = pixels_xy[..., 1].to(depth.dtype)
    xyz_camera = torch.stack(
        (
            (u - intrinsics[:, None, 0, 2]) * depth / intrinsics[:, None, 0, 0],
            (v - intrinsics[:, None, 1, 2]) * depth / intrinsics[:, None, 1, 1],
            depth,
        ),
        -1,
    )
    xyz_camera = torch.where(valid[..., None], xyz_camera, torch.zeros_like(xyz_camera))
    out = {"xyz_camera": xyz_camera, "valid": valid}
    if pose_c2w is not None:
        xyz_world = (
            torch.einsum("bij,bpj->bpi", pose_c2w[:, :3, :3], xyz_camera)
            + pose_c2w[:, None, :3, 3]
        )
        out["xyz_world"] = torch.where(valid[..., None], xyz_world, torch.zeros_like(xyz_world))
    return out


def project_landmarks(
    xyz_world: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_c2w: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    landmark_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Project world landmarks into one camera and report local-search eligibility."""
    if xyz_world.ndim != 3 or xyz_world.shape[-1] != 3 or not xyz_world.is_floating_point():
        raise ValueError("xyz_world must be floating [B,M,3]")
    batch, landmarks, _ = xyz_world.shape
    _intrinsics(intrinsics, batch, xyz_world.device, xyz_world.dtype)
    _pose(pose_c2w, batch, xyz_world.device, xyz_world.dtype)
    if landmark_valid is None:
        landmark_valid = torch.ones((batch, landmarks), dtype=torch.bool, device=xyz_world.device)
    elif landmark_valid.shape != (batch, landmarks) or landmark_valid.dtype != torch.bool or landmark_valid.device != xyz_world.device:
        raise ValueError("landmark_valid must be bool [B,M] on landmark device")
    if (
        not isinstance(image_hw, tuple)
        or len(image_hw) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in image_hw)
    ):
        raise ValueError("image_hw must contain two positive integers")
    height, width = image_hw
    rotation = pose_c2w[:, :3, :3].transpose(-1, -2)
    xyz_camera = torch.einsum(
        "bij,bmj->bmi",
        rotation,
        xyz_world - pose_c2w[:, None, :3, 3],
    )
    z = xyz_camera[..., 2]
    eps = torch.finfo(xyz_world.dtype).eps
    pixels = torch.stack(
        (
            intrinsics[:, None, 0, 0] * xyz_camera[..., 0] / z.clamp_min(eps)
            + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * xyz_camera[..., 1] / z.clamp_min(eps)
            + intrinsics[:, None, 1, 2],
        ),
        -1,
    )
    valid = (
        landmark_valid
        & torch.isfinite(xyz_world).all(-1)
        & torch.isfinite(pixels).all(-1)
        & (z > 0)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < height)
    )
    return {
        "pixels_xy": torch.where(valid[..., None], pixels, torch.zeros_like(pixels)),
        "depth": torch.where(valid, z, torch.zeros_like(z)),
        "valid": valid,
    }
