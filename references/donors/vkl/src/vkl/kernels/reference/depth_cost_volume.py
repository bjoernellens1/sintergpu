"""Edge-aware semi-global aggregation for dense depth cost volumes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[:, :1] if rgb.shape[1] == 1 else (rgb[:, :3] * torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)[None, :, None, None]).sum(1, keepdim=True)


def _scan(cost: torch.Tensor, gray: torch.Tensor, *, dim: int, reverse: bool, smoothness_penalty: float, edge_scale: float) -> torch.Tensor:
    """Aggregate one scan direction without crossing strong RGB edges."""
    length = cost.shape[dim]
    output = torch.empty_like(cost)
    indices = range(length - 1, -1, -1) if reverse else range(length)
    previous = None
    previous_gray = None
    for index in indices:
        current = cost.select(dim, index)
        current_gray = gray.select(2 if dim == 2 else 3, index)
        if previous is None:
            value = current
        else:
            p2 = smoothness_penalty * torch.exp(-(current_gray - previous_gray).abs() / edge_scale)
            p1 = p2
            same = previous
            lower = torch.cat((torch.full_like(previous[:, :1], torch.inf), previous[:, :-1] + p1), 1)
            upper = torch.cat((previous[:, 1:] + p1, torch.full_like(previous[:, :1], torch.inf)), 1)
            any_label = previous.amin(1, keepdim=True) + p2
            value = current + torch.minimum(torch.minimum(same, lower), torch.minimum(upper, any_label)) - previous.amin(1, keepdim=True)
        output.select(dim, index).copy_(value)
        previous, previous_gray = value, current_gray
    return output


def aggregate_depth_cost_volume(cost: torch.Tensor, rgb_reference: torch.Tensor, *, smoothness_penalty: float = 0.05, edge_scale: float = 0.10, aggregation_scale: int = 1) -> torch.Tensor:
    """Return four-direction SGM-style aggregation of ``[B,D,H,W]`` costs.

    Large/non-finite candidate costs remain unsupported; they are converted to a
    finite sentinel only for the dynamic-programming recurrence.
    """
    if cost.ndim != 4 or rgb_reference.ndim != 4 or rgb_reference.shape[:1] != cost.shape[:1] or rgb_reference.shape[-2:] != cost.shape[-2:] or rgb_reference.shape[1] not in {1, 3}:
        raise ValueError("cost must be [B,D,H,W] and RGB must match [B,C,H,W]")
    if not cost.is_floating_point() or rgb_reference.dtype != cost.dtype or rgb_reference.device != cost.device or smoothness_penalty < 0 or edge_scale <= 0 or aggregation_scale < 1 or aggregation_scale > min(cost.shape[-2:]):
        raise ValueError("cost/RGB device-dtype and aggregation parameters must be valid")
    if aggregation_scale > 1:
        finite = torch.isfinite(cost)
        valid = F.avg_pool2d(finite.to(cost.dtype), aggregation_scale, aggregation_scale)
        reduced = F.avg_pool2d(torch.where(finite, cost, torch.zeros_like(cost)), aggregation_scale, aggregation_scale) / valid.clamp_min(torch.finfo(cost.dtype).eps)
        reduced = reduced.masked_fill(valid == 0, torch.inf)
        aggregated = aggregate_depth_cost_volume(
            reduced,
            F.interpolate(rgb_reference, size=reduced.shape[-2:], mode="area"),
            smoothness_penalty=smoothness_penalty,
            edge_scale=edge_scale,
        )
        return F.interpolate(aggregated, size=cost.shape[-2:], mode="bilinear", align_corners=False)
    finite = torch.isfinite(cost)
    fallback = torch.where(finite, cost, torch.zeros_like(cost)).amax().detach() + max(1.0, smoothness_penalty)
    stable_cost = torch.where(finite, cost, torch.full_like(cost, fallback))
    gray = _gray(rgb_reference)
    paths = (
        _scan(stable_cost, gray, dim=2, reverse=False, smoothness_penalty=smoothness_penalty, edge_scale=edge_scale),
        _scan(stable_cost, gray, dim=2, reverse=True, smoothness_penalty=smoothness_penalty, edge_scale=edge_scale),
        _scan(stable_cost, gray, dim=3, reverse=False, smoothness_penalty=smoothness_penalty, edge_scale=edge_scale),
        _scan(stable_cost, gray, dim=3, reverse=True, smoothness_penalty=smoothness_penalty, edge_scale=edge_scale),
    )
    return sum(paths) / len(paths)
