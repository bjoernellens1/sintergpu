"""Fixed-order tracking normal-equation reduction and 6x6 solve."""

from __future__ import annotations

import torch


def _fixed_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce axis 1 through a fixed left-to-right binary tree."""
    while values.shape[1] > 1:
        if values.shape[1] % 2:
            values = torch.cat((values, torch.zeros_like(values[:, :1])), dim=1)
        values = values[:, 0::2] + values[:, 1::2]
    return values[:, 0]


def deterministic_reduce_solve(
    partial_H: torch.Tensor,
    partial_b: torch.Tensor,
    partial_valid: torch.Tensor,
    *,
    damping: float = 1e-4,
    condition_max: float = 1e6,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Reduce fixed partials and solve ``H delta = -b`` with explicit health.

    The reduction order never depends on data values. Invalid partials become
    exact zero contributions, and an ill-conditioned system returns a zero
    update rather than an untrusted pose increment.
    """
    if validate and (
        partial_H.ndim != 4
        or partial_H.shape[-2:] != (6, 6)
        or partial_b.shape != partial_H.shape[:-1]
        or partial_valid.shape != partial_H.shape[:2]
        or not partial_H.is_floating_point()
        or partial_b.device != partial_H.device
        or partial_b.dtype != partial_H.dtype
        or partial_valid.device != partial_H.device
        or partial_valid.dtype != torch.bool
        or not torch.isfinite(partial_H).all()
        or not torch.isfinite(partial_b).all()
        or not torch.isfinite(torch.tensor((damping, condition_max))).all()
        or damping < 0
        or condition_max <= 1
    ):
        raise ValueError("partials must be finite [B,K,6,6]/[B,K,6], with finite solve parameters")
    zeros_H = torch.zeros_like(partial_H)
    zeros_b = torch.zeros_like(partial_b)
    H = _fixed_tree(torch.where(partial_valid[:, :, None, None], partial_H, zeros_H))
    b = _fixed_tree(torch.where(partial_valid[:, :, None], partial_b, zeros_b))
    H = (H + H.transpose(-1, -2)) * 0.5
    eigenvalues = torch.linalg.eigvalsh(H)
    minimum = eigenvalues[:, 0]
    maximum = eigenvalues[:, -1]
    epsilon = torch.finfo(H.dtype).eps
    condition = torch.where(minimum > epsilon, maximum / minimum, torch.full_like(minimum, torch.inf))
    factor, info = torch.linalg.cholesky_ex(H + damping * torch.eye(6, device=H.device, dtype=H.dtype))
    candidate = torch.cholesky_solve(-b.unsqueeze(-1), factor).squeeze(-1)
    valid = (partial_valid.sum(1) > 0) & (minimum > epsilon) & (condition <= condition_max) & (info == 0)
    delta = torch.where(valid[:, None], candidate, torch.zeros_like(candidate))
    return {
        "H": H,
        "b": b,
        "delta": delta,
        "valid": valid,
        "damping": torch.full_like(minimum, damping),
        "condition_proxy": condition,
        "minimum_eigenvalue": minimum,
        "partial_count": partial_valid.sum(1).to(torch.int64),
    }
