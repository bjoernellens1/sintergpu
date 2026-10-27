# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Explicit camera rectification and metric depth-to-RGB registration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _output_hw(value: tuple[int, int] | list[int]) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in value)
    ):
        raise ValueError("output_hw must contain two positive integers")
    return int(value[0]), int(value[1])


def _intrinsics(
    value: torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    *,
    validate: bool,
) -> None:
    if value.shape != (batch, 3, 3) or value.device != device or value.dtype != dtype:
        raise ValueError(f"{name} must be floating [B,3,3] on image device/dtype")
    if not validate:
        return
    canonical = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    if (
        not torch.isfinite(value).all()
        or (value[:, 0, 0] <= 0).any()
        or (value[:, 1, 1] <= 0).any()
        or (value[:, 0, 1] != canonical[0, 1]).any()
        or (value[:, 1, 0] != canonical[1, 0]).any()
        or (value[:, 2, :2] != canonical[2, :2]).any()
        or (value[:, 2, 2] != 1).any()
    ):
        raise ValueError(f"{name} must be finite canonical pinhole intrinsics with positive focal lengths")


def _transform(
    value: torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    validate: bool,
) -> None:
    if value.shape != (batch, 4, 4) or value.device != device or value.dtype != dtype:
        raise ValueError("transform_rgb_depth must be [B,4,4] on depth device/dtype")
    if not validate:
        return
    rotation = value[:, :3, :3]
    identity = torch.eye(3, device=device, dtype=dtype).expand(batch, -1, -1)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(batch, -1)
    if (
        not torch.isfinite(value).all()
        or not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=1e-4)
        or (torch.linalg.det(rotation) <= 0).any()
        or not torch.allclose(value[:, 3], bottom, atol=1e-6, rtol=0)
    ):
        raise ValueError("transform_rgb_depth must be a rigid homogeneous depth-to-RGB transform")


def _coefficients(
    model: str,
    value: torch.Tensor | None,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    *,
    validate: bool,
) -> torch.Tensor | None:
    if model == "pinhole":
        if value is not None:
            raise ValueError(f"{name} coefficients require brown_conrady model")
        return None
    if (
        model != "brown_conrady"
        or value is None
        or value.shape != (batch, 5)
        or value.device != device
        or value.dtype != dtype
    ):
        raise ValueError(
            f"{name} model must be pinhole or brown_conrady with [B,5] "
            "k1,k2,p1,p2,k3 coefficients"
        )
    if validate and not torch.isfinite(value).all():
        raise ValueError(f"{name} coefficients must be finite")
    return value


def _distort(
    x: torch.Tensor,
    y: torch.Tensor,
    coefficients: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coefficients is None:
        return x, y
    k1, k2, p1, p2, k3 = (coefficients[:, None, None, index] for index in range(5))
    radius2 = x.square() + y.square()
    radial = 1 + k1 * radius2 + k2 * radius2.square() + k3 * radius2.pow(3)
    return (
        x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x.square()),
        y * radial + p1 * (radius2 + 2 * y.square()) + 2 * p2 * x * y,
    )


def _undistort(
    xd: torch.Tensor,
    yd: torch.Tensor,
    coefficients: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coefficients is None:
        return xd, yd
    x, y = xd, yd
    for _ in range(5):
        projected_x, projected_y = _distort(x, y, coefficients)
        x = x + xd - projected_x
        y = y + yd - projected_y
    return x, y


def _sample_preclamped_grid(
    image: torch.Tensor,
    sample_u: torch.Tensor,
    sample_v: torch.Tensor,
    sampling: str,
) -> torch.Tensor:
    """Portable sampler for an already in-bounds pixel-coordinate grid."""
    batch, channels, source_h, source_w = image.shape
    target_h, target_w = sample_u.shape[-2:]
    flat = image.reshape(batch, channels, source_h * source_w)

    def gather(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        index = (y * source_w + x).reshape(batch, 1, -1).expand(-1, channels, -1)
        return flat.gather(2, index).reshape(batch, channels, target_h, target_w)

    if sampling == "nearest":
        return gather(sample_u.round().long(), sample_v.round().long())
    x0 = sample_u.floor().long()
    y0 = sample_v.floor().long()
    x1 = (x0 + 1).clamp_max(source_w - 1)
    y1 = (y0 + 1).clamp_max(source_h - 1)
    wx = (sample_u - x0).unsqueeze(1)
    wy = (sample_v - y0).unsqueeze(1)
    top = gather(x0, y0).lerp(gather(x1, y0), wx)
    bottom = gather(x0, y1).lerp(gather(x1, y1), wx)
    return top.lerp(bottom, wy)


def camera_rectify_grid(
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_hw: tuple[int, int] | list[int],
    output_hw: tuple[int, int] | list[int],
    *,
    source_distortion_model: str = "pinhole",
    source_distortion_coefficients: torch.Tensor | None = None,
    debug: bool = False,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Build reusable source-pixel coordinates for target-pinhole rectification."""
    if (
        source_intrinsics.ndim != 3
        or source_intrinsics.shape[1:] != (3, 3)
        or not source_intrinsics.is_floating_point()
    ):
        raise ValueError("source_intrinsics must be floating [B,3,3]")
    source_h, source_w = _output_hw(source_hw)
    target_h, target_w = _output_hw(output_hw)
    batch = source_intrinsics.shape[0]
    if batch < 1:
        raise ValueError("source_intrinsics batch must be positive")
    device, dtype = source_intrinsics.device, source_intrinsics.dtype
    _intrinsics(source_intrinsics, batch, device, dtype, "source_intrinsics", validate=validate)
    _intrinsics(target_intrinsics, batch, device, dtype, "target_intrinsics", validate=validate)
    coefficients = _coefficients(
        source_distortion_model,
        source_distortion_coefficients,
        batch,
        device,
        dtype,
        "source_distortion",
        validate=validate,
    )
    y, x = torch.meshgrid(
        torch.arange(target_h, device=device, dtype=dtype),
        torch.arange(target_w, device=device, dtype=dtype),
        indexing="ij",
    )
    xn = (x[None] - target_intrinsics[:, None, None, 0, 2]) / target_intrinsics[:, None, None, 0, 0]
    yn = (y[None] - target_intrinsics[:, None, None, 1, 2]) / target_intrinsics[:, None, None, 1, 1]
    xd, yd = _distort(xn, yn, coefficients)
    source_u = xd * source_intrinsics[:, None, None, 0, 0] + source_intrinsics[:, None, None, 0, 2]
    source_v = yd * source_intrinsics[:, None, None, 1, 1] + source_intrinsics[:, None, None, 1, 2]
    border_tolerance = 16 * torch.finfo(dtype).eps * max(source_h, source_w)
    valid = (
        torch.isfinite(source_u)
        & torch.isfinite(source_v)
        & (source_u >= -border_tolerance)
        & (source_u <= source_w - 1 + border_tolerance)
        & (source_v >= -border_tolerance)
        & (source_v <= source_h - 1 + border_tolerance)
    )
    out = {
        "source_uv": torch.stack((source_u, source_v), -1),
        "valid": valid[:, None],
    }
    if debug:
        out.update(
            {
                "target_ray": torch.stack((xn, yn, torch.ones_like(xn)), -1),
                "displacement": (torch.stack((source_u - x, source_v - y), -1).square().sum(-1).sqrt())[:, None],
            }
        )
    return out


def camera_rectify_from_grid(
    image: torch.Tensor,
    source_uv: torch.Tensor,
    valid: torch.Tensor,
    *,
    sampling: str = "bilinear",
) -> dict[str, torch.Tensor]:
    """Rectify an image from reusable source-pixel coordinates and validity."""
    if image.ndim != 4 or not image.is_floating_point() or sampling not in {"bilinear", "nearest"}:
        raise ValueError("image must be floating [B,C,H,W] and sampling bilinear or nearest")
    batch, _, source_h, source_w = image.shape
    target_shape = (batch, source_uv.shape[1] if source_uv.ndim == 4 else -1, source_uv.shape[2] if source_uv.ndim == 4 else -1)
    if (
        source_uv.ndim != 4
        or source_uv.shape != (*target_shape, 2)
        or source_uv.device != image.device
        or source_uv.dtype != image.dtype
        or valid.shape != (batch, 1, target_shape[1], target_shape[2])
        or valid.device != image.device
        or valid.dtype != torch.bool
    ):
        raise ValueError("source_uv and valid must be [B,H,W,2] and bool [B,1,H,W] on image device")
    source_u, source_v = source_uv.unbind(-1)
    sample_u = torch.where(torch.isfinite(source_u), source_u, torch.zeros_like(source_u)).clamp(0, source_w - 1)
    sample_v = torch.where(torch.isfinite(source_v), source_v, torch.zeros_like(source_v)).clamp(0, source_h - 1)
    grid = torch.stack(
        (2 * sample_u / max(source_w - 1, 1) - 1, 2 * sample_v / max(source_h - 1, 1) - 1),
        -1,
    )
    if torch.compiler.is_compiling():
        rectified = _sample_preclamped_grid(image, sample_u, sample_v, sampling)
    else:
        rectified = F.grid_sample(image, grid, mode=sampling, padding_mode="zeros", align_corners=True)
    return {"image": torch.where(valid, rectified, torch.zeros_like(rectified)), "valid": valid}


def camera_rectify(
    image: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    output_hw: tuple[int, int] | list[int],
    *,
    source_distortion_model: str = "pinhole",
    source_distortion_coefficients: torch.Tensor | None = None,
    sampling: str = "bilinear",
    debug: bool = False,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Sample an explicit target pinhole grid from a pinhole or Brown source camera.

    Brown coefficients are ordered ``k1,k2,p1,p2,k3``. ``sampling`` is caller
    policy: use bilinear for RGB and normally nearest for metric depth. Public
    calls validate calibration by default; prevalidated compiled hot paths may
    pass ``validate=False`` to avoid data-dependent Python checks.
    """
    if image.ndim != 4 or not image.is_floating_point() or sampling not in {"bilinear", "nearest"}:
        raise ValueError("image must be floating [B,C,H,W] and sampling bilinear or nearest")
    batch, _, source_h, source_w = image.shape
    if source_h < 1 or source_w < 1:
        raise ValueError("image spatial dimensions must be positive")
    if source_intrinsics.shape != (batch, 3, 3) or source_intrinsics.device != image.device or source_intrinsics.dtype != image.dtype:
        raise ValueError("source_intrinsics must be floating [B,3,3] on image device/dtype")
    grid = camera_rectify_grid(
        source_intrinsics,
        target_intrinsics,
        (source_h, source_w),
        output_hw,
        source_distortion_model=source_distortion_model,
        source_distortion_coefficients=source_distortion_coefficients,
        debug=debug,
        validate=validate,
    )
    out = camera_rectify_from_grid(image, grid["source_uv"], grid["valid"], sampling=sampling)
    if debug:
        out.update({key: grid[key] for key in ("source_uv", "target_ray", "displacement")})
    return out


def depth_camera_rays(
    depth_intrinsics: torch.Tensor,
    depth_hw: tuple[int, int] | list[int],
    *,
    depth_distortion_model: str = "pinhole",
    depth_distortion_coefficients: torch.Tensor | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Build reusable undistorted normalized depth-camera rays ``[B,H,W,2]``."""
    if (
        depth_intrinsics.ndim != 3
        or depth_intrinsics.shape[1:] != (3, 3)
        or not depth_intrinsics.is_floating_point()
    ):
        raise ValueError("depth_intrinsics must be floating [B,3,3]")
    depth_h, depth_w = _output_hw(depth_hw)
    batch = depth_intrinsics.shape[0]
    if batch < 1:
        raise ValueError("depth_intrinsics batch must be positive")
    device, dtype = depth_intrinsics.device, depth_intrinsics.dtype
    _intrinsics(depth_intrinsics, batch, device, dtype, "depth_intrinsics", validate=validate)
    coefficients = _coefficients(
        depth_distortion_model,
        depth_distortion_coefficients,
        batch,
        device,
        dtype,
        "depth_distortion",
        validate=validate,
    )
    y, x = torch.meshgrid(
        torch.arange(depth_h, device=device, dtype=dtype),
        torch.arange(depth_w, device=device, dtype=dtype),
        indexing="ij",
    )
    xd = (x[None] - depth_intrinsics[:, None, None, 0, 2]) / depth_intrinsics[:, None, None, 0, 0]
    yd = (y[None] - depth_intrinsics[:, None, None, 1, 2]) / depth_intrinsics[:, None, None, 1, 1]
    xn, yn = _undistort(xd, yd, coefficients)
    return torch.stack((xn, yn), -1)


def depth_register(
    depth: torch.Tensor,
    depth_intrinsics: torch.Tensor,
    rgb_intrinsics: torch.Tensor,
    transform_rgb_depth: torch.Tensor,
    output_hw: tuple[int, int] | list[int],
    *,
    depth_distortion_model: str = "pinhole",
    depth_distortion_coefficients: torch.Tensor | None = None,
    rgb_distortion_model: str = "pinhole",
    rgb_distortion_coefficients: torch.Tensor | None = None,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
    depth_rays: torch.Tensor | None = None,
    debug: bool = False,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Register metric depth to an RGB grid using nearest positive RGB-camera Z."""
    if depth.ndim != 4 or depth.shape[1] != 1 or not depth.is_floating_point():
        raise ValueError("depth must be floating [B,1,H,W]")
    target_h, target_w = _output_hw(output_hw)
    if (
        not math.isfinite(min_depth)
        or min_depth <= 0
        or (max_depth is not None and (not math.isfinite(max_depth) or max_depth <= min_depth))
    ):
        raise ValueError("depth interval must be finite and ordered")
    batch, _, depth_h, depth_w = depth.shape
    if depth_h < 1 or depth_w < 1:
        raise ValueError("depth spatial dimensions must be positive")
    _intrinsics(
        depth_intrinsics,
        batch,
        depth.device,
        depth.dtype,
        "depth_intrinsics",
        validate=validate,
    )
    _intrinsics(
        rgb_intrinsics,
        batch,
        depth.device,
        depth.dtype,
        "rgb_intrinsics",
        validate=validate,
    )
    _transform(transform_rgb_depth, batch, depth.device, depth.dtype, validate=validate)
    depth_coefficients = _coefficients(
        depth_distortion_model,
        depth_distortion_coefficients,
        batch,
        depth.device,
        depth.dtype,
        "depth_distortion",
        validate=validate,
    )
    rgb_coefficients = _coefficients(
        rgb_distortion_model,
        rgb_distortion_coefficients,
        batch,
        depth.device,
        depth.dtype,
        "rgb_distortion",
        validate=validate,
    )
    if depth_rays is None:
        depth_rays = depth_camera_rays(
            depth_intrinsics,
            (depth_h, depth_w),
            depth_distortion_model=depth_distortion_model,
            depth_distortion_coefficients=depth_coefficients,
            validate=False,
        )
    elif (
        depth_rays.shape != (batch, depth_h, depth_w, 2)
        or depth_rays.device != depth.device
        or depth_rays.dtype != depth.dtype
        or (validate and not torch.isfinite(depth_rays).all())
    ):
        raise ValueError("depth_rays must be finite [B,H,W,2] on depth device/dtype")
    xn, yn = depth_rays.unbind(-1)
    source_depth = depth[:, 0]
    depth_valid = torch.isfinite(source_depth) & (source_depth >= min_depth)
    if max_depth is not None:
        depth_valid &= source_depth <= max_depth
    safe_depth = torch.where(depth_valid, source_depth, torch.zeros_like(source_depth))
    point_depth = torch.stack((xn * safe_depth, yn * safe_depth, safe_depth), -1).reshape(batch, -1, 3)
    point_rgb = point_depth @ transform_rgb_depth[:, :3, :3].transpose(-1, -2) + transform_rgb_depth[:, None, :3, 3]
    z = point_rgb[..., 2]
    positive_z = torch.isfinite(z) & (z > torch.finfo(depth.dtype).eps)
    safe_z = torch.where(positive_z, z, torch.ones_like(z))
    xn_rgb = point_rgb[..., 0] / safe_z
    yn_rgb = point_rgb[..., 1] / safe_z
    xd_rgb, yd_rgb = _distort(xn_rgb[:, :, None], yn_rgb[:, :, None], rgb_coefficients)
    u = xd_rgb[:, :, 0] * rgb_intrinsics[:, None, 0, 0] + rgb_intrinsics[:, None, 0, 2]
    v = yd_rgb[:, :, 0] * rgb_intrinsics[:, None, 1, 1] + rgb_intrinsics[:, None, 1, 2]
    projected_finite = torch.isfinite(u) & torch.isfinite(v) & positive_z
    safe_u = torch.where(projected_finite, u, torch.zeros_like(u))
    safe_v = torch.where(projected_finite, v, torch.zeros_like(v))
    ui, vi = safe_u.round().long(), safe_v.round().long()
    source_valid = (
        depth_valid.reshape(batch, -1)
        & projected_finite
        & (ui >= 0)
        & (ui < target_w)
        & (vi >= 0)
        & (vi < target_h)
    )
    target_index = vi.clamp(0, target_h - 1) * target_w + ui.clamp(0, target_w - 1)
    points = depth_h * depth_w
    infinite = torch.full_like(z, torch.inf)
    nearest_z = torch.full(
        (batch, target_h * target_w),
        torch.inf,
        device=depth.device,
        dtype=depth.dtype,
    )
    nearest_z.scatter_reduce_(
        1,
        target_index,
        torch.where(source_valid, z, infinite),
        reduce="amin",
        include_self=True,
    )
    winner_z = nearest_z.gather(1, target_index)
    source_index = torch.arange(points, device=depth.device, dtype=torch.long)[None].expand(batch, -1)
    candidate_index = torch.where(
        source_valid & (z == winner_z),
        source_index,
        torch.full_like(source_index, points),
    )
    winner = torch.full(
        (batch, target_h * target_w),
        points,
        device=depth.device,
        dtype=torch.long,
    )
    winner.scatter_reduce_(1, target_index, candidate_index, reduce="amin", include_self=True)
    registered = nearest_z.reshape(batch, 1, target_h, target_w)
    valid = torch.isfinite(registered)
    out = {
        "depth": torch.where(valid, registered, torch.zeros_like(registered)),
        "valid": valid,
    }
    if debug:
        collision = torch.zeros(
            (batch, target_h * target_w),
            device=depth.device,
            dtype=torch.int64,
        )
        collision.scatter_add_(1, target_index, source_valid.to(torch.int64))
        winner_image = winner.reshape(batch, 1, target_h, target_w)
        out.update(
            {
                "winner_source_index": torch.where(
                    valid,
                    winner_image,
                    torch.full_like(winner_image, -1),
                ),
                "collision_count": collision.reshape(batch, 1, target_h, target_w),
                "source_uv": torch.stack((u, v), -1),
                "projected_z": z,
                "source_valid": source_valid,
            }
        )
    return out
