"""Best-observation reduction for temporal evidence."""

from __future__ import annotations

import torch


def temporal_best_observation(
    pairwise_equivalence: torch.Tensor,
    target_quality: torch.Tensor,
    neighbor_quality: torch.Tensor,
    *,
    pairwise_quality: torch.Tensor | None = None,
    pairwise_confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Select the highest-quality equivalent observation, retaining target fallback.

    Index zero denotes the target; neighbor ``i`` is returned as index ``i+1``.
    No mapping novelty or admission policy is applied here.
    """
    if pairwise_equivalence.ndim != 2 or pairwise_equivalence.dtype != torch.bool:
        raise ValueError("pairwise_equivalence must be bool [B,N]")
    if target_quality.ndim != 1 or neighbor_quality.ndim != 2 or neighbor_quality.shape != pairwise_equivalence.shape or target_quality.shape[0] != pairwise_equivalence.shape[0]:
        raise ValueError("quality tensors must have shapes [B] and [B,N]")
    if not target_quality.is_floating_point() or neighbor_quality.dtype != target_quality.dtype or neighbor_quality.device != target_quality.device or pairwise_equivalence.device != target_quality.device:
        raise ValueError("quality and equivalence tensors must share floating device/dtype")
    quality = neighbor_quality if pairwise_quality is None else pairwise_quality
    if quality.shape != neighbor_quality.shape or quality.dtype != target_quality.dtype or quality.device != target_quality.device:
        raise ValueError("pairwise_quality must match neighbor_quality")
    confidence = pairwise_equivalence.to(target_quality.dtype) if pairwise_confidence is None else pairwise_confidence
    if confidence.shape != neighbor_quality.shape or confidence.dtype != target_quality.dtype or confidence.device != target_quality.device or not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("pairwise_confidence must be finite [0,1] [B,N]")
    candidate_quality = torch.where(pairwise_equivalence, quality, torch.full_like(quality, -torch.inf))
    best_neighbor_quality, best_neighbor = candidate_quality.max(1)
    target_wins = (~torch.isfinite(best_neighbor_quality)) | (target_quality >= best_neighbor_quality)
    best_index = torch.where(target_wins, torch.zeros_like(best_neighbor), best_neighbor + 1).to(torch.int64)
    best_quality = torch.where(target_wins, target_quality, best_neighbor_quality)
    advantage = (best_neighbor_quality - target_quality).clamp_min(0).where(~target_wins, torch.zeros_like(target_quality))
    replaceability = torch.where(target_wins, torch.zeros_like(target_quality), advantage)
    replacement_confidence = torch.where(target_wins, torch.zeros_like(target_quality), confidence.gather(1, best_neighbor[:, None]).squeeze(1) * replaceability)
    return {
        "best_index": best_index,
        "best_quality": best_quality,
        "replaceability": replaceability,
        "replacement_confidence": replacement_confidence,
    }
