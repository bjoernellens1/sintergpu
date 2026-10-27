"""Fixed-shape composition of dense and sparse tracking normal equations."""

from __future__ import annotations

import torch

from .tracking_solve import deterministic_reduce_solve


def compose_tracking_solve(
    *,
    dense_partial_H: torch.Tensor,
    dense_partial_b: torch.Tensor,
    dense_partial_valid: torch.Tensor,
    sparse_partial_H: torch.Tensor,
    sparse_partial_b: torch.Tensor,
    sparse_partial_valid: torch.Tensor,
    damping: float = 1e-4,
    condition_max: float = 1e6,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Reduce fixed dense and sparse evidence together through one solve tree.

    Each source owns its own partial construction.  This boundary preserves
    their evidence provenance while giving the solver one data-independent
    reduction order.
    """
    if validate and (
        dense_partial_H.ndim != 4
        or dense_partial_H.shape[-2:] != (6, 6)
        or dense_partial_b.shape != dense_partial_H.shape[:-1]
        or dense_partial_valid.shape != dense_partial_H.shape[:2]
        or sparse_partial_H.ndim != 4
        or sparse_partial_H.shape[-2:] != (6, 6)
        or sparse_partial_b.shape != sparse_partial_H.shape[:-1]
        or sparse_partial_valid.shape != sparse_partial_H.shape[:2]
        or sparse_partial_H.shape[0] != dense_partial_H.shape[0]
        or any(
            value.device != dense_partial_H.device
            or value.dtype != dense_partial_H.dtype
            for value in (dense_partial_b, sparse_partial_H, sparse_partial_b)
        )
        or dense_partial_valid.device != dense_partial_H.device
        or sparse_partial_valid.device != dense_partial_H.device
        or dense_partial_valid.dtype != torch.bool
        or sparse_partial_valid.dtype != torch.bool
    ):
        raise ValueError("dense and sparse partials must be compatible fixed [B,K,6,6]/[B,K,6] tensors")

    partial_H = torch.cat((dense_partial_H, sparse_partial_H), dim=1)
    partial_b = torch.cat((dense_partial_b, sparse_partial_b), dim=1)
    partial_valid = torch.cat((dense_partial_valid, sparse_partial_valid), dim=1)
    solved = deterministic_reduce_solve(
        partial_H,
        partial_b,
        partial_valid,
        damping=damping,
        condition_max=condition_max,
        validate=validate,
    )
    return {
        **solved,
        "dense_partial_count": dense_partial_valid.sum(1).to(torch.int64),
        "sparse_partial_count": sparse_partial_valid.sum(1).to(torch.int64),
    }
