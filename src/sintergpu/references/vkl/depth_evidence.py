# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_evidence(depth: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor | None = None, *, grid_size: tuple[int, int] = (8, 8), near: float = 0.1, far: float = 8.0) -> dict[str, torch.Tensor]:
    """Return support-normalized evidence; local variance is per-pair population variance."""
    if depth.ndim != 4 or depth.shape[1] != 1 or not depth.is_floating_point() or valid.shape != depth.shape or valid.dtype != torch.bool or valid.device != depth.device:
        raise ValueError("depth and valid must be [B,1,H,W] on one device")
    if confidence is not None and (confidence.shape != depth.shape or confidence.dtype != depth.dtype or confidence.device != depth.device or not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError("confidence must be finite in [0,1] and match depth")
    if near < 0 or far <= near or not isinstance(grid_size, tuple) or len(grid_size) != 2 or not all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in grid_size): raise ValueError("invalid bounds or grid size")
    _, _, h, w = depth.shape; usable = valid & torch.isfinite(depth); d = torch.where(usable, depth, torch.zeros_like(depth)); support = usable.to(depth.dtype); n = support.sum((-2, -1)); safe = n.clamp_min(1)
    vf = n[:, 0] / (h * w); near_f = ((d < near) & usable).sum((-2, -1))[:, 0] / safe[:, 0]; far_f = ((d > far) & usable).sum((-2, -1))[:, 0] / safe[:, 0]
    conf = support if confidence is None else torch.where(usable, confidence, torch.zeros_like(confidence)); cm = conf.sum((-2, -1))[:, 0] / safe[:, 0]
    rp = usable[..., :, 1:] & usable[..., :, :-1]; dp = usable[..., 1:, :] & usable[..., :-1, :]; rd = d[..., :, 1:] - d[..., :, :-1]; dd = d[..., 1:, :] - d[..., :-1, :]
    edge = F.pad((rd.abs() > d[..., :, :-1].abs().clamp_min(1e-6) * .25) & rp, (0, 1, 0, 0)) | F.pad((dd.abs() > d[..., :-1, :].abs().clamp_min(1e-6) * .25) & dp, (0, 0, 0, 1))
    edge_fraction = edge.sum((-2, -1))[:, 0] / safe[:, 0]; pairs = rp.sum((-2, -1)) + dp.sum((-2, -1)); local_variance = ((rd.square() * rp).sum((-2, -1)) + (dd.square() * dp).sum((-2, -1)))[:, 0] / (4 * pairs[:, 0].clamp_min(1)); stable = usable & ~edge & (conf > 0); stable_fraction = stable.sum((-2, -1))[:, 0] / safe[:, 0]
    gh, gw = min(grid_size[0], h), min(grid_size[1], w); fields = torch.cat((support, ((d < near) & usable).to(depth.dtype), ((d > far) & usable).to(depth.dtype), d, d.square(), edge.to(depth.dtype), conf, stable.to(depth.dtype)), 1)
    rows = torch.arange(h, device=depth.device) * gh // h; cols = torch.arange(w, device=depth.device) * gw // w; bins = (rows[:, None] * gw + cols).reshape(1, 1, -1); flat = fields.flatten(2); index = bins.expand(depth.shape[0], flat.shape[1], -1); sums = torch.zeros((*flat.shape[:2], gh * gw), device=depth.device, dtype=depth.dtype).scatter_add(2, index, flat); support_sum = sums[:, :1]; area = torch.bincount(bins.flatten(), minlength=gh * gw).to(depth.dtype)[None, None]; grid = sums / support_sum.clamp_min(1); grid[:, :1] = support_sum / area
    return {"valid_fraction": vf, "hole_fraction": 1 - vf, "near_fraction": near_f, "far_fraction": far_f, "local_variance": local_variance, "edge_fraction": edge_fraction, "confidence_mean": cm, "stable_fraction": stable_fraction, "grid_evidence": grid.reshape(depth.shape[0], -1, gh, gw).permute(0, 2, 3, 1)}
