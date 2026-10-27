"""Fixed-capacity pose-seeded translational Lucas--Kanade tracking."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return rgb if rgb.shape[1] == 1 else (rgb * rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]).sum(1, keepdim=True)


def _domain(rgb: torch.Tensor, valid: torch.Tensor | None, name: str) -> torch.Tensor:
    expected = (rgb.shape[0], 1, *rgb.shape[-2:])
    if valid is not None and (valid.shape != expected or valid.dtype != torch.bool or valid.device != rgb.device):
        raise ValueError(f"{name} must be bool [B,1,H,W] on RGB device")
    finite = torch.isfinite(rgb).all(1, keepdim=True)
    return finite if valid is None else (finite & valid)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    # Outside-image support is invalid too. Padding the invalid mask with ones
    # prevents zero-padded central differences at the border from looking valid.
    invalid = F.pad((~mask).to(torch.float32), (radius, radius, radius, radius), value=1.0)
    invalid = F.max_pool2d(invalid, 2 * radius + 1, stride=1) > 0
    return ~invalid


def lk_track(
    rgb_reference: torch.Tensor,
    rgb_target: torch.Tensor,
    pixels_reference: torch.Tensor,
    pixels_initial: torch.Tensor,
    *,
    pixels_valid: torch.Tensor | None = None,
    rgb_reference_valid: torch.Tensor | None = None,
    rgb_target_valid: torch.Tensor | None = None,
    patch_radius: int = 2,
    iterations: int = 4,
    min_determinant: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Refine known-pose projected pixels with fixed-iteration translational LK.

    RGB validity masks carry camera-conditioning support through both patches.
    A track remains valid only while every bilinear sample and target-gradient
    support pixel stays inside finite, valid image content.
    """
    if (
        rgb_reference.ndim != 4
        or rgb_reference.shape[1] not in {1, 3}
        or not rgb_reference.is_floating_point()
        or rgb_target.shape != rgb_reference.shape
        or rgb_target.device != rgb_reference.device
        or rgb_target.dtype != rgb_reference.dtype
    ):
        raise ValueError("reference and target RGB must be matching floating [B,1|3,H,W]")
    if rgb_reference.device.type == "cpu" and rgb_reference.dtype not in {torch.float32, torch.float64}:
        raise ValueError("CPU LK supports float32/float64; use GPU for low precision")
    batch, _, height, width = rgb_reference.shape
    if (
        pixels_reference.ndim != 3
        or pixels_reference.shape[0] != batch
        or pixels_reference.shape[-1] != 2
        or pixels_reference.dtype != torch.int64
        or pixels_reference.device != rgb_reference.device
        or pixels_initial.shape != pixels_reference.shape
        or not pixels_initial.is_floating_point()
        or pixels_initial.device != rgb_reference.device
        or pixels_initial.dtype != rgb_reference.dtype
    ):
        raise ValueError("reference pixels must be int64 [B,P,2] and initial pixels floating [B,P,2] on RGB device/dtype")
    if (
        not isinstance(patch_radius, int)
        or isinstance(patch_radius, bool)
        or patch_radius < 1
        or not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
        or not isinstance(min_determinant, (int, float))
        or isinstance(min_determinant, bool)
        or not math.isfinite(min_determinant)
        or min_determinant <= 0
    ):
        raise ValueError("patch_radius, iterations, and min_determinant must be positive")
    points = pixels_reference.shape[1]
    if points < 1:
        raise ValueError("LK requires at least one fixed-capacity point")
    if pixels_valid is None:
        pixels_valid = torch.ones((batch, points), dtype=torch.bool, device=rgb_reference.device)
    elif pixels_valid.shape != (batch, points) or pixels_valid.dtype != torch.bool or pixels_valid.device != rgb_reference.device:
        raise ValueError("pixels_valid must be bool [B,P] on RGB device")

    reference_domain = _domain(rgb_reference, rgb_reference_valid, "rgb_reference_valid")
    target_domain = _domain(rgb_target, rgb_target_valid, "rgb_target_valid")
    target_gradient_domain = _erode(target_domain, 1)
    safe_reference = torch.where(reference_domain, rgb_reference, torch.zeros_like(rgb_reference))
    safe_target = torch.where(target_domain, rgb_target, torch.zeros_like(rgb_target))

    offset_y, offset_x = torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=rgb_reference.device, dtype=rgb_reference.dtype),
        torch.arange(-patch_radius, patch_radius + 1, device=rgb_reference.device, dtype=rgb_reference.dtype),
        indexing="ij",
    )
    offsets = torch.stack((offset_x.reshape(-1), offset_y.reshape(-1)), -1)

    def finite_inside(points_xy: torch.Tensor) -> torch.Tensor:
        sampled = points_xy[:, :, None] + offsets
        return (
            torch.isfinite(sampled).all((-1, -2))
            & (sampled[..., 0] >= 0).all(-1)
            & (sampled[..., 0] < width).all(-1)
            & (sampled[..., 1] >= 0).all(-1)
            & (sampled[..., 1] < height).all(-1)
        )

    def sample(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(points_xy).all(-1, keepdim=True)
        safe_points = torch.where(finite, points_xy, torch.zeros_like(points_xy))
        sampled = safe_points[:, :, None] + offsets
        grid = torch.stack(
            (sampled[..., 0] * (2 / max(width - 1, 1)) - 1, sampled[..., 1] * (2 / max(height - 1, 1)) - 1),
            -1,
        )
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]

    def patch_supported(mask: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
        sampled = sample(mask.to(rgb_reference.dtype), points_xy)
        tolerance = 8.0 * torch.finfo(rgb_reference.dtype).eps
        return (sampled >= 1.0 - tolerance).all(-1)

    reference_xy = pixels_reference.to(rgb_reference.dtype)
    valid = (
        pixels_valid
        & finite_inside(reference_xy)
        & finite_inside(pixels_initial)
        & patch_supported(reference_domain, reference_xy)
        & patch_supported(target_gradient_domain, pixels_initial)
    )
    reference = sample(_gray(safe_reference), reference_xy)
    target = _gray(safe_target)
    grad_x = F.pad((target[..., 2:] - target[..., :-2]) * 0.5, (1, 1, 0, 0))
    grad_y = F.pad((target[..., 2:, :] - target[..., :-2, :]) * 0.5, (0, 0, 1, 1))
    current = pixels_initial
    for _ in range(iterations):
        target_patch, gx, gy = sample(target, current), sample(grad_x, current), sample(grad_y, current)
        residual = target_patch - reference
        h00, h01, h11 = gx.square().sum(-1), (gx * gy).sum(-1), gy.square().sum(-1)
        b0, b1 = (gx * residual).sum(-1), (gy * residual).sum(-1)
        determinant = h00 * h11 - h01.square()
        step_valid = (
            valid
            & finite_inside(current)
            & patch_supported(target_gradient_domain, current)
            & torch.isfinite(residual).all(-1)
            & torch.isfinite(gx).all(-1)
            & torch.isfinite(gy).all(-1)
            & torch.isfinite(determinant)
            & (determinant >= min_determinant)
        )
        safe = torch.where(step_valid, determinant, torch.ones_like(determinant))
        delta = torch.stack(((h11 * b0 - h01 * b1) / safe, (h00 * b1 - h01 * b0) / safe), -1)
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        current = torch.where(step_valid[..., None], current - delta, current)
        valid = step_valid & finite_inside(current) & patch_supported(target_gradient_domain, current)
    residual = (sample(target, current) - reference).abs().mean(-1)
    valid &= torch.isfinite(residual)
    return {
        "pixels_xy": torch.where(valid[..., None], current, torch.zeros_like(current)),
        "valid": valid,
        "residual": torch.where(valid, residual, torch.zeros_like(residual)),
    }
