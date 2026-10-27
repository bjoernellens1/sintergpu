"""Unpacked BRIEF reference descriptors for pose-gated local matching."""

from __future__ import annotations

import torch


_PATCH_SIDE = 31
_PATCH_POINTS = _PATCH_SIDE * _PATCH_SIDE
_MAX_DEFAULT_BITS = _PATCH_POINTS * (_PATCH_POINTS - 1)


def _default_pairs(bits: int, device: torch.device) -> torch.Tensor:
    """Generate deterministic distinct ordered comparisons over a 31x31 patch."""
    if not 1 <= bits <= _MAX_DEFAULT_BITS:
        raise ValueError(f"num_bits must be in [1,{_MAX_DEFAULT_BITS}]")
    index = torch.arange(bits, device=device, dtype=torch.int64)
    # 104729 is coprime with 961*960, so this affine map is a permutation of
    # all ordered distinct sample pairs rather than a short-period pattern.
    code = (index * 104729 + 13007).remainder(_MAX_DEFAULT_BITS)
    first = code // (_PATCH_POINTS - 1)
    second_rank = code.remainder(_PATCH_POINTS - 1)
    second = second_rank + (second_rank >= first).to(torch.int64)

    def xy(position: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (position.remainder(_PATCH_SIDE) - 15, position // _PATCH_SIDE - 15),
            dim=-1,
        )

    return torch.stack((xy(first), xy(second)), dim=1)


def brief_describe(
    rgb: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    pixels_valid: torch.Tensor | None = None,
    rgb_valid: torch.Tensor | None = None,
    pairs: torch.Tensor | None = None,
    num_bits: int = 256,
) -> dict[str, torch.Tensor]:
    """Return deterministic unpacked BRIEF comparisons at integer keypoints.

    ``pixels_valid`` preserves fixed-capacity detector validity. ``rgb_valid``
    carries camera-conditioning validity into the descriptor patch. A descriptor
    is valid only when its keypoint center and every requested comparison sample
    lie inside the image on finite, valid RGB support.
    """
    if rgb.ndim != 4 or rgb.shape[1] not in {1, 3} or not rgb.is_floating_point():
        raise ValueError("rgb must be floating [B,1|3,H,W]")
    batch, _, height, width = rgb.shape
    if (
        pixels_xy.ndim != 3
        or pixels_xy.shape[0] != batch
        or pixels_xy.shape[-1] != 2
        or pixels_xy.dtype != torch.int64
        or pixels_xy.device != rgb.device
    ):
        raise ValueError("pixels_xy must be int64 [B,P,2] on rgb device")
    point_shape = pixels_xy.shape[:2]
    if pixels_valid is None:
        pixels_valid = torch.ones(point_shape, dtype=torch.bool, device=rgb.device)
    elif pixels_valid.shape != point_shape or pixels_valid.dtype != torch.bool or pixels_valid.device != rgb.device:
        raise ValueError("pixels_valid must be bool [B,P] on rgb device")
    if rgb_valid is not None and (
        rgb_valid.shape != (batch, 1, height, width)
        or rgb_valid.dtype != torch.bool
        or rgb_valid.device != rgb.device
    ):
        raise ValueError("rgb_valid must be bool [B,1,H,W] on rgb device")
    if pairs is None:
        pairs = _default_pairs(num_bits, rgb.device)
    if (
        pairs.ndim != 3
        or pairs.shape[0] < 1
        or pairs.shape[1:] != (2, 2)
        or pairs.dtype != torch.int64
        or pairs.device != rgb.device
    ):
        raise ValueError("pairs must be non-empty int64 [D,2,2] on rgb device")
    if (pairs[:, 0] == pairs[:, 1]).all(-1).any():
        raise ValueError("BRIEF comparisons must use distinct sample locations")

    finite_rgb = torch.isfinite(rgb).all(1, keepdim=True)
    sample_domain = finite_rgb if rgb_valid is None else (finite_rgb & rgb_valid)
    safe_rgb = torch.where(sample_domain, rgb, torch.zeros_like(rgb))
    gray = (
        safe_rgb[:, :1]
        if safe_rgb.shape[1] == 1
        else (
            safe_rgb
            * torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)[
                None, :, None, None
            ]
        ).sum(1, keepdim=True)
    )

    center_x, center_y = pixels_xy[..., 0], pixels_xy[..., 1]
    center_inside = (
        (center_x >= 0)
        & (center_x < width)
        & (center_y >= 0)
        & (center_y < height)
    )
    center_index = center_y.clamp(0, height - 1) * width + center_x.clamp(0, width - 1)
    center_domain = sample_domain[:, 0].reshape(batch, -1).gather(1, center_index)

    samples = pixels_xy[:, :, None, None, :] + pairs[None, None]
    inside = (
        (samples[..., 0] >= 0)
        & (samples[..., 0] < width)
        & (samples[..., 1] >= 0)
        & (samples[..., 1] < height)
    )
    xx = samples[..., 0].clamp(0, width - 1)
    yy = samples[..., 1].clamp(0, height - 1)
    flat_index = yy * width + xx
    sampled_domain = (
        sample_domain[:, 0]
        .reshape(batch, -1)
        .gather(1, flat_index.reshape(batch, -1))
        .reshape(batch, pixels_xy.shape[1], pairs.shape[0], 2)
    )
    valid = (
        pixels_valid
        & center_inside
        & center_domain
        & inside.all((-1, -2))
        & sampled_domain.all((-1, -2))
    )
    values = (
        gray[:, 0]
        .reshape(batch, -1)
        .gather(1, flat_index.reshape(batch, -1))
        .reshape(batch, pixels_xy.shape[1], pairs.shape[0], 2)
    )
    descriptor = values[..., 0] < values[..., 1]
    return {"descriptor": descriptor & valid[..., None], "valid": valid}
