# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Reference exposure and RGB-gain image-formation transform."""

from __future__ import annotations

import torch


def photometric_isp(
    rgb_linear: torch.Tensor,
    *,
    exposure_ev: torch.Tensor | None = None,
    rgb_gain: torch.Tensor | None = None,
    color_matrix: torch.Tensor | None = None,
    vignette: torch.Tensor | None = None,
    response_gamma: torch.Tensor | None = None,
    validate: bool = True,
    evidence: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply optional exposure, colour, vignetting, and response transforms.

    ``validate=False`` is the statically-specialized fast path for already
    validated tensor pipelines and ``torch.compile``. ``evidence=False`` omits
    image-wide diagnostics while retaining the ``rgb_observed`` dictionary
    contract. Public callers keep validation and evidence by default.
    """
    if (
        not isinstance(rgb_linear, torch.Tensor)
        or not rgb_linear.is_floating_point()
        or rgb_linear.ndim != 4
        or rgb_linear.shape[1] != 3
    ):
        raise ValueError("rgb_linear must be a floating [B,3,H,W] tensor")
    if validate and (not torch.isfinite(rgb_linear).all() or not ((rgb_linear >= 0) & (rgb_linear <= 1)).all()):
        raise ValueError("rgb_linear must contain finite values in [0, 1]")
    batch = rgb_linear.shape[0]

    if exposure_ev is not None and (
        not isinstance(exposure_ev, torch.Tensor)
        or not exposure_ev.is_floating_point()
        or exposure_ev.shape != (batch, 1, 1, 1)
        or exposure_ev.device != rgb_linear.device
        or exposure_ev.dtype != rgb_linear.dtype
    ):
        raise ValueError("exposure_ev must be floating [B,1,1,1] on rgb_linear device and dtype")
    if validate and exposure_ev is not None and not torch.isfinite(exposure_ev).all():
        raise ValueError("exposure_ev must contain finite values")
    if rgb_gain is not None and (
        not isinstance(rgb_gain, torch.Tensor)
        or not rgb_gain.is_floating_point()
        or rgb_gain.shape != (batch, 3, 1, 1)
        or rgb_gain.device != rgb_linear.device
        or rgb_gain.dtype != rgb_linear.dtype
    ):
        raise ValueError("rgb_gain must be floating [B,3,1,1] on rgb_linear device and dtype")
    if validate and rgb_gain is not None and (not torch.isfinite(rgb_gain).all() or not (rgb_gain >= 0).all()):
        raise ValueError("rgb_gain must contain finite non-negative values")
    if color_matrix is not None and (
        not isinstance(color_matrix, torch.Tensor)
        or not color_matrix.is_floating_point()
        or color_matrix.shape != (batch, 3, 3)
        or color_matrix.device != rgb_linear.device
        or color_matrix.dtype != rgb_linear.dtype
    ):
        raise ValueError("color_matrix must be finite [B,3,3] on rgb_linear device and dtype")
    if validate and color_matrix is not None and not torch.isfinite(color_matrix).all():
        raise ValueError("color_matrix must be finite [B,3,3] on rgb_linear device and dtype")
    if vignette is not None and (
        not isinstance(vignette, torch.Tensor)
        or not vignette.is_floating_point()
        or vignette.shape != (batch, 3, 3)
        or vignette.device != rgb_linear.device
        or vignette.dtype != rgb_linear.dtype
    ):
        raise ValueError("vignette must be finite [B,3,3] on rgb_linear device and dtype")
    if validate and vignette is not None and (
        not torch.isfinite(vignette).all() or not (vignette >= 0).all()
    ):
        raise ValueError("vignette must contain finite non-negative attenuation coefficients")
    if response_gamma is not None and (
        not isinstance(response_gamma, torch.Tensor)
        or not response_gamma.is_floating_point()
        or response_gamma.shape != (batch, 1, 1, 1)
        or response_gamma.device != rgb_linear.device
        or response_gamma.dtype != rgb_linear.dtype
    ):
        raise ValueError("response_gamma must be finite positive [B,1,1,1] on rgb_linear device and dtype")
    if validate and response_gamma is not None and (
        not torch.isfinite(response_gamma).all() or not (response_gamma > 0).all()
    ):
        raise ValueError("response_gamma must be finite positive [B,1,1,1] on rgb_linear device and dtype")

    raw = rgb_linear
    exposure_gain = torch.ones((batch, 1, 1, 1), device=rgb_linear.device, dtype=rgb_linear.dtype)
    if exposure_ev is not None:
        exposure_gain = torch.exp2(exposure_ev)
        raw = raw * exposure_gain
    rgb_gain_applied = torch.ones((batch, 3, 1, 1), device=rgb_linear.device, dtype=rgb_linear.dtype)
    if rgb_gain is not None:
        rgb_gain_applied = rgb_gain
        raw = raw * rgb_gain_applied
    vignette_gain = None
    if vignette is not None:
        height, width = rgb_linear.shape[-2:]
        y = torch.linspace(-1, 1, height, device=raw.device, dtype=raw.dtype)[:, None]
        x = torch.linspace(-1, 1, width, device=raw.device, dtype=raw.dtype)[None, :]
        radius_squared = x.square() + y.square()
        powers = torch.stack((radius_squared, radius_squared.square(), radius_squared.pow(3)))
        vignette_gain = torch.exp(-torch.einsum("bck,khw->bchw", vignette, powers))
        raw = raw * vignette_gain
    if color_matrix is not None:
        raw = torch.einsum("bij,bjhw->bihw", color_matrix, raw)
    lower_clipped = raw < 0
    # A positive floor keeps power-response gradients finite at black for
    # 0 < gamma < 1 while subtracting the same offset keeps black exact.
    if response_gamma is not None:
        epsilon = torch.as_tensor(torch.finfo(raw.dtype).eps, device=raw.device, dtype=raw.dtype)
        response = (raw.clamp_min(0) + epsilon).pow(response_gamma) - epsilon.pow(response_gamma)
    else:
        response = raw.clamp_min(0)
    upper_clipped = response > 1
    rgb_observed = response.clamp(0, 1)
    if not evidence:
        return {"rgb_observed": rgb_observed}

    color_matrix_applied = (
        torch.eye(3, device=raw.device, dtype=raw.dtype).expand(batch, -1, -1)
        if color_matrix is None
        else color_matrix
    )
    clipping_ratio = torch.stack(
        [lower_clipped.to(rgb_linear.dtype).mean(dim=(1, 2, 3)), upper_clipped.to(rgb_linear.dtype).mean(dim=(1, 2, 3))],
        dim=1,
    )
    correction_magnitude = (rgb_observed - rgb_linear).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    color_deviation = (color_matrix_applied - torch.eye(3, device=raw.device, dtype=raw.dtype)).abs().mean(dim=(1, 2)).unsqueeze(1)
    if vignette_gain is None:
        vignette_gain_range = torch.ones((batch, 2), device=raw.device, dtype=raw.dtype)
    else:
        vignette_gain_range = torch.stack((vignette_gain.amin(dim=(1, 2, 3)), vignette_gain.amax(dim=(1, 2, 3))), dim=1)
    crf_response_range = torch.stack((response.amin(dim=(1, 2, 3)), response.amax(dim=(1, 2, 3))), dim=1)
    return {
        "rgb_observed": rgb_observed,
        "exposure_gain": exposure_gain,
        "rgb_gain_applied": rgb_gain_applied,
        "clipping_ratio": clipping_ratio,
        "correction_magnitude": correction_magnitude,
        "color_deviation": color_deviation,
        "vignette_gain_range": vignette_gain_range,
        "crf_response_range": crf_response_range,
    }
