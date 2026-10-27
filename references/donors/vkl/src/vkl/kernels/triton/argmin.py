"""Triton-backed inverse-depth hypothesis selection."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the eager fallback.
    triton = tl = None


if triton is not None:
    @triton.jit
    def _argmin2(cost, rho, best_out, second_out, index_out, depth_out, rows, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        values = tl.load(cost + row * D + offsets, mask=(row < rows) & (offsets < D), other=float("inf"))
        best, index = tl.min(values, axis=0, return_indices=True)
        second = tl.min(tl.where(offsets == index, float("inf"), values), axis=0)
        rho_best = tl.load(rho + index)
        tl.store(best_out + row, best, mask=row < rows)
        tl.store(second_out + row, second, mask=row < rows)
        tl.store(index_out + row, index, mask=row < rows)
        tl.store(depth_out + row, rho_best, mask=row < rows)


def select_best_inverse_depth(cost: torch.Tensor, inverse_depths: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return two best costs and selected inverse depth for ``[B,P,D]`` costs.

    CUDA/HIP tensors use one Triton program per point; other devices use the
    exact eager equivalent so the public tensor contract stays portable.
    """
    if cost.ndim != 3 or not cost.is_floating_point() or inverse_depths.ndim != 1 or inverse_depths.numel() < 2:
        raise ValueError("cost must be floating [B,P,D] and inverse_depths [D>=2]")
    if cost.shape[-1] != inverse_depths.numel() or cost.device != inverse_depths.device or cost.dtype != inverse_depths.dtype:
        raise ValueError("cost and inverse_depths must share dtype/device with matching D")
    if cost.device.type != "cuda" or triton is None:
        best_two, indices = cost.topk(2, dim=-1, largest=False)
        index = indices[..., 0]
        return {"best_cost": best_two[..., 0], "second_cost": best_two[..., 1], "best_index": index, "inverse_depth": inverse_depths[index]}
    b, p, d = cost.shape
    rows = b * p
    best = torch.empty((rows,), device=cost.device, dtype=cost.dtype)
    second = torch.empty_like(best)
    index = torch.empty((rows,), device=cost.device, dtype=torch.int64)
    depth = torch.empty_like(best)
    _argmin2[(rows,)](cost, inverse_depths, best, second, index, depth, rows, D=d, BLOCK=triton.next_power_of_2(d))
    return {"best_cost": best.reshape(b, p), "second_cost": second.reshape(b, p), "best_index": index.reshape(b, p), "inverse_depth": depth.reshape(b, p)}
