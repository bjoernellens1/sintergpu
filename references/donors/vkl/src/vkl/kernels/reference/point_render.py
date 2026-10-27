"""Minimal depth-buffer point renderer for geometry diagnostics."""

from __future__ import annotations

import torch

from .sparse_geometry import project_landmarks


def point_splat_render(
    xyz_world: torch.Tensor,
    rgb: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_c2w: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    valid: torch.Tensor | None = None,
    splat_radius: int = 0,
) -> dict[str, torch.Tensor]:
    """Nearest-depth rasterize colored points into a diagnostic image.

    This is a one-pixel point splat diagnostic, not a mapping renderer: it has
    no interpolation, learned appearance, antialiasing, or differentiable
    visibility policy.
    """
    if (
        xyz_world.ndim != 3
        or xyz_world.shape[-1] != 3
        or not xyz_world.is_floating_point()
        or rgb.shape != xyz_world.shape
        or rgb.device != xyz_world.device
        or rgb.dtype != xyz_world.dtype
        or not torch.isfinite(rgb).all()
        or (rgb < 0).any()
        or (rgb > 1).any()
    ):
        raise ValueError("xyz_world and finite rgb [0,1] must be matching floating [B,N,3]")
    batch, points, _ = xyz_world.shape
    if valid is None:
        valid = torch.ones((batch, points), device=xyz_world.device, dtype=torch.bool)
    elif valid.shape != (batch, points) or valid.device != xyz_world.device or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,N] on xyz_world device")
    if (
        not isinstance(image_hw, tuple)
        or len(image_hw) != 2
        or min(image_hw) < 1
        or not isinstance(splat_radius, int)
        or isinstance(splat_radius, bool)
        or splat_radius < 0
    ):
        raise ValueError("image_hw must contain positive height and width")
    height, width = image_hw
    projected = project_landmarks(
        xyz_world, intrinsics, pose_c2w, image_hw=image_hw, landmark_valid=valid
    )
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-splat_radius, splat_radius + 1, device=xyz_world.device),
        torch.arange(-splat_radius, splat_radius + 1, device=xyz_world.device),
        indexing="ij",
    )
    offsets = torch.stack((offsets_x.reshape(-1), offsets_y.reshape(-1)), -1)
    pixel_x = projected["pixels_xy"][..., 0].round().to(torch.int64)[..., None] + offsets[:, 0]
    pixel_y = projected["pixels_xy"][..., 1].round().to(torch.int64)[..., None] + offsets[:, 1]
    visible = (
        projected["valid"][..., None]
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    pixel_x = pixel_x.clamp(0, width - 1)
    pixel_y = pixel_y.clamp(0, height - 1)
    index = pixel_y * width + pixel_x
    projected_depth = projected["depth"][..., None].expand_as(index)
    inf = torch.full_like(projected_depth, torch.inf)
    depth_flat = torch.full((batch, height * width), torch.inf, device=xyz_world.device, dtype=xyz_world.dtype)
    depth_flat.scatter_reduce_(1, index.flatten(1), torch.where(visible, projected_depth, inf).flatten(1), reduce="amin", include_self=True)
    winning_depth = depth_flat.gather(1, index.flatten(1)).reshape_as(index)
    winning = visible & (projected_depth - winning_depth).abs().le(
        torch.finfo(xyz_world.dtype).eps * winning_depth.abs().clamp_min(1)
    )
    splat_rgb = rgb.repeat_interleave(offsets.shape[0], dim=1).transpose(1, 2)
    image_flat = rgb.new_zeros((batch, 3, height * width))
    image_flat.scatter_reduce_(
        2,
        index.flatten(1)[:, None].expand(-1, 3, -1),
        torch.where(
            winning.flatten(1)[:, None],
            splat_rgb,
            torch.zeros_like(splat_rgb),
        ),
        reduce="amax",
        include_self=True,
    )
    support_flat = torch.zeros((batch, height * width), device=xyz_world.device, dtype=torch.int64)
    support_flat.scatter_add_(1, index.flatten(1), visible.to(torch.int64).flatten(1))
    valid_image = torch.isfinite(depth_flat)
    return {
        "rgb": image_flat.reshape(batch, 3, height, width),
        "depth": torch.where(valid_image, depth_flat, torch.zeros_like(depth_flat)).reshape(batch, 1, height, width),
        "valid": valid_image.reshape(batch, 1, height, width),
        "support_count": support_flat.reshape(batch, 1, height, width),
    }
