"""Reference RGB resolution pyramid."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resolution_pyramid(*, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build native, half, quarter, and eighth-resolution RGB images.

    ``rgb`` must be a floating-point ``[B, 3, H, W]`` tensor.  Lower levels
    use antialiased bilinear interpolation and ``ceil(H / scale)`` /
    ``ceil(W / scale)`` dimensions, so every non-empty input dimension stays
    represented by at least one output pixel.  The native tensor is returned
    unchanged; every level remains on the same device and has its dtype.
    """
    if not isinstance(rgb, torch.Tensor):
        raise TypeError("rgb must be a torch.Tensor")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B, 3, H, W]")
    if not rgb.is_floating_point():
        raise TypeError("rgb must be a floating point tensor")
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError("rgb must have positive spatial dimensions")

    levels = {"native": rgb}
    interpolation_rgb = rgb
    if rgb.device.type == "cpu" and rgb.dtype in (torch.float16, torch.bfloat16):
        interpolation_rgb = rgb.float()
    for name, scale in (("half", 2), ("quarter", 4), ("eighth", 8)):
        size = ((height + scale - 1) // scale, (width + scale - 1) // scale)
        level = F.interpolate(interpolation_rgb, size=size, mode="bilinear", align_corners=False, antialias=True)
        levels[name] = level.to(rgb.dtype)
    return levels
