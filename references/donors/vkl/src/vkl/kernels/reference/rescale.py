from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _canonical_shape(
    h: int,
    w: int,
    target_pixels: int,
    min_width: int,
    min_height: int,
    allow_upscale: bool,
) -> tuple[int, int, float, float]:
    scale = math.sqrt(float(target_pixels) / float(h * w))
    if not allow_upscale:
        scale = min(1.0, scale)
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    if out_w < min_width and w >= min_width:
        scale = min_width / w
        out_w = min_width
        out_h = max(1, int(round(h * scale)))
    if out_h < min_height and h >= min_height:
        scale = min_height / h
        out_h = min_height
        out_w = max(1, int(round(w * scale)))
    if not allow_upscale:
        out_h, out_w = min(out_h, h), min(out_w, w)
    return out_h, out_w, out_w / w, out_h / h


def _mask(
    value: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape != (batch, 1, height, width) or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be bool [B,1,H,W] on RGB device")
    return value


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.to(torch.float32), size=size, mode="nearest") > 0.5


def canonical_rescale(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
    rgb_valid: torch.Tensor | None = None,
    depth_valid: torch.Tensor | None = None,
    depth_measurement_valid: torch.Tensor | None = None,
    target_pixels: int = 307_200,
    min_width: int = 320,
    min_height: int = 240,
    allow_upscale: bool = False,
) -> dict[str, torch.Tensor]:
    """Rescale RGB-D while keeping conditioning and measurement validity distinct.

    ``depth_valid`` is the geometric/conditioning domain. ``depth_measurement_valid``
    marks locations that actually contain a depth measurement. They are separate
    because a registered RGB-grid pixel can be inside the valid camera overlap
    while still having no projected depth sample. When no explicit measurement
    mask is supplied, finite positive depth is the portable fallback; zero is the
    canonical missing-depth sentinel used by the bundled RGB-D loaders.
    """
    if (
        rgb.ndim != 4
        or rgb.shape[1] != 3
        or not rgb.is_floating_point()
        or min(rgb.shape[-2:]) < 1
        or target_pixels < 1
        or min_width < 1
        or min_height < 1
    ):
        raise ValueError("rgb must be non-empty floating [B,3,H,W] with positive resize limits")
    batch, _, h, w = rgb.shape
    if depth is not None and (
        depth.shape != (batch, 1, h, w)
        or not depth.is_floating_point()
        or depth.device != rgb.device
        or depth.dtype != rgb.dtype
    ):
        raise ValueError("depth must be floating [B,1,H,W] on RGB device/dtype")
    if depth is None and (depth_valid is not None or depth_measurement_valid is not None):
        raise ValueError("depth validity masks require depth")
    if intrinsics is not None and (
        intrinsics.shape != (batch, 3, 3)
        or not intrinsics.is_floating_point()
        or intrinsics.device != rgb.device
        or intrinsics.dtype != rgb.dtype
    ):
        raise ValueError("intrinsics must be floating [B,3,3] on RGB device/dtype")
    rgb_valid = _mask(rgb_valid, batch=batch, height=h, width=w, device=rgb.device, name="rgb_valid")
    depth_valid = _mask(depth_valid, batch=batch, height=h, width=w, device=rgb.device, name="depth_valid")
    depth_measurement_valid = _mask(
        depth_measurement_valid,
        batch=batch,
        height=h,
        width=w,
        device=rgb.device,
        name="depth_measurement_valid",
    )

    finite_rgb = torch.isfinite(rgb).all(1, keepdim=True)
    rgb_domain = finite_rgb if rgb_valid is None else (rgb_valid & finite_rgb)
    safe_rgb = torch.where(rgb_domain, rgb, torch.zeros_like(rgb))

    if depth is None:
        depth_domain = None
        depth_measurement = None
        safe_depth = None
    else:
        finite_depth = torch.isfinite(depth)
        depth_domain = (
            torch.ones_like(depth, dtype=torch.bool) if depth_valid is None else depth_valid
        )
        depth_measurement = (
            finite_depth & (depth > 0)
            if depth_measurement_valid is None
            else (depth_measurement_valid & finite_depth)
        )
        depth_measurement &= depth_domain
        safe_depth = torch.where(depth_measurement, depth, torch.zeros_like(depth))

    oh, ow, sx, sy = _canonical_shape(
        h,
        w,
        int(target_pixels),
        int(min_width),
        int(min_height),
        bool(allow_upscale),
    )
    if (oh, ow) == (h, w):
        out_rgb = safe_rgb
        out_depth = safe_depth
        out_rgb_valid = rgb_domain
        out_depth_valid = depth_domain
        out_depth_measurement = depth_measurement
    else:
        low_precision_cpu = rgb.device.type == "cpu" and rgb.dtype in (torch.float16, torch.bfloat16)
        work_rgb = safe_rgb.float() if low_precision_cpu else safe_rgb
        weight = rgb_domain.to(work_rgb.dtype)
        numerator = F.interpolate(
            work_rgb * weight,
            size=(oh, ow),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        denominator = F.interpolate(
            weight,
            size=(oh, ow),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        out_rgb_valid = _resize_mask(rgb_domain, (oh, ow))
        normalized = numerator / denominator.clamp_min(torch.finfo(work_rgb.dtype).eps)
        out_rgb = torch.where(out_rgb_valid, normalized, torch.zeros_like(normalized)).to(rgb.dtype)
        if depth is None:
            out_depth = None
            out_depth_valid = None
            out_depth_measurement = None
        else:
            work_depth = safe_depth.float() if low_precision_cpu else safe_depth
            out_depth = F.interpolate(work_depth, size=(oh, ow), mode="nearest").to(depth.dtype)
            out_depth_valid = _resize_mask(depth_domain, (oh, ow))
            out_depth_measurement = _resize_mask(depth_measurement, (oh, ow))
            out_depth = torch.where(out_depth_measurement, out_depth, torch.zeros_like(out_depth))

    out_intrinsics = intrinsics
    if intrinsics is not None:
        out_intrinsics = intrinsics.clone()
        out_intrinsics[..., 0, 0] *= sx
        out_intrinsics[..., 0, 2] *= sx
        out_intrinsics[..., 1, 1] *= sy
        out_intrinsics[..., 1, 2] *= sy

    result = {
        "rgb": out_rgb,
        "rgb_valid": out_rgb_valid,
        "resolution_meta": torch.tensor(
            [h, w, oh, ow, sy, sx], device=rgb.device, dtype=torch.float32
        ),
    }
    if out_depth is not None:
        result["depth"] = out_depth
        result["depth_valid"] = out_depth_valid
        result["depth_measurement_valid"] = out_depth_measurement
    if out_intrinsics is not None:
        result["intrinsics"] = out_intrinsics
    return result
