# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

from collections.abc import Sequence

import torch


def fuse_depth(
    depths: Sequence[torch.Tensor],
    valids: Sequence[torch.Tensor],
    confidences: Sequence[torch.Tensor],
    *,
    metric_status: Sequence[str] | None = None,
    consistency_threshold: float = 0.05,
) -> dict[str, torch.Tensor | str]:
    """Fuse same-grid depth fields, weighting consistent values and rejecting conflicts."""
    if not depths or len(depths) != len(valids) or len(depths) != len(confidences):
        raise ValueError("depths, valids, and confidences must be non-empty and same length")
    if consistency_threshold <= 0:
        raise ValueError("consistency_threshold must be positive")
    shape, device, dtype = depths[0].shape, depths[0].device, depths[0].dtype
    if len(shape) != 4 or shape[1] != 1 or not depths[0].is_floating_point():
        raise ValueError("depth fields must be floating point [B,1,H,W]")
    for depth, valid, conf in zip(depths, valids, confidences):
        if depth.shape != shape or depth.device != device or depth.dtype != dtype or not depth.is_floating_point():
            raise ValueError("all depths must share shape, device, and dtype")
        if valid.shape != shape or valid.dtype != torch.bool or valid.device != device:
            raise ValueError("valid masks must be bool and match depth")
        if conf.shape != shape or conf.dtype != dtype or conf.device != device or not torch.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
            raise ValueError("confidences must be finite [0,1] and match depth")
    if metric_status is not None and len(metric_status) != len(depths):
        raise ValueError("metric_status must match fields")
    status = tuple(metric_status or ("METRIC_ANCHORED",) * len(depths))
    allowed = {"RELATIVE", "METRIC_ESTIMATED", "METRIC_ANCHORED"}
    if any(item not in allowed for item in status):
        raise ValueError("unsupported depth metric status")
    relative = [item == "RELATIVE" for item in status]
    if any(relative) and not all(relative):
        raise ValueError("relative and metric depth cannot be fused without alignment")
    output_status = "RELATIVE" if all(relative) else ("METRIC_ANCHORED" if "METRIC_ANCHORED" in status else "METRIC_ESTIMATED")
    d = torch.stack(tuple(depths), 0)
    c = torch.stack(tuple(confidences), 0)
    v = torch.stack(tuple(valids), 0) & torch.isfinite(d) & (d > 0) & (c > 0)
    safe_d = torch.where(v, d, torch.zeros_like(d))
    w = torch.where(v, c, torch.zeros_like(c))
    total = w.sum(0)
    mean = (w * safe_d).sum(0) / total.clamp_min(torch.finfo(dtype).eps)
    consistent = v & ((d - mean.unsqueeze(0)).abs() <= consistency_threshold * mean.unsqueeze(0).clamp_min(torch.finfo(dtype).eps))
    all_consistent = consistent.any(0) & (consistent.sum(0) == v.sum(0))
    consistent_w = torch.where(consistent, w, torch.zeros_like(w))
    consistent_d = torch.where(consistent, d, torch.zeros_like(d))
    fused_consistent = (consistent_w * consistent_d).sum(0) / consistent_w.sum(0).clamp_min(torch.finfo(dtype).eps)
    trusted = torch.tensor([s in {"METRIC_ESTIMATED", "METRIC_ANCHORED"} for s in status], device=device, dtype=torch.bool)[:, None, None, None, None]
    pick_score = torch.where(v & trusted, c, torch.where(v, c * 0.5, torch.zeros_like(c)))
    winner = pick_score.argmax(0)
    fallback = torch.gather(safe_d, 0, winner.unsqueeze(0)).squeeze(0)
    out_valid = v.any(0)
    out_depth = torch.where(all_consistent, fused_consistent, fallback)
    out_conf = torch.where(all_consistent, consistent_w.sum(0).clamp_max(1), pick_score.max(0).values)
    out_depth = torch.where(out_valid, out_depth, torch.zeros_like(out_depth))
    out_conf = torch.where(out_valid, out_conf, torch.zeros_like(out_conf))
    winner = torch.where(out_valid, winner, torch.full_like(winner, -1))
    return {"depth": out_depth, "valid": out_valid, "confidence": out_conf, "winning_source": winner, "metric_status": output_status}
