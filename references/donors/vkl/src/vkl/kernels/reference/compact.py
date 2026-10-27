"""Stable fixed-capacity tensor compaction."""

from __future__ import annotations

import torch


def select_compact(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    action_codes: torch.Tensor | None = None,
    capacity: int,
) -> dict[str, torch.Tensor]:
    """Compact ``[B,N,...]`` values in stable order into ``[B,capacity,...]``."""
    if values.ndim < 2 or mask.shape != values.shape[:2] or mask.dtype is not torch.bool or mask.device != values.device or capacity < 0:
        raise ValueError("values [B,N,...], matching bool mask, and non-negative capacity are required")
    if action_codes is not None and (action_codes.shape != mask.shape or action_codes.device != values.device or action_codes.dtype not in (torch.int32, torch.int64)):
        raise ValueError("action_codes must be matching int32 or int64")
    b, n = mask.shape
    rank = mask.to(torch.int64).cumsum(1) - 1
    keep = mask & (rank < capacity)
    batch = torch.arange(b, device=values.device)[:, None].expand(b, n)[keep]
    destination = rank[keep]
    compact_values = torch.zeros((b, capacity, *values.shape[2:]), device=values.device, dtype=values.dtype)
    compact_indices = torch.zeros((b, capacity), device=values.device, dtype=torch.int64)
    compact_valid = torch.zeros((b, capacity), device=values.device, dtype=torch.bool)
    compact_codes = torch.zeros((b, capacity), device=values.device, dtype=action_codes.dtype if action_codes is not None else torch.int32)
    compact_values[batch, destination] = values[keep]
    compact_indices[batch, destination] = torch.arange(n, device=values.device)[None].expand(b, n)[keep]
    compact_valid[batch, destination] = True
    if action_codes is not None:
        compact_codes[batch, destination] = action_codes[keep]
    return {"values": compact_values, "indices": compact_indices, "action_codes": compact_codes, "valid": compact_valid, "count": keep.sum(1, dtype=torch.int64)}
