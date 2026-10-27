from __future__ import annotations

from collections.abc import Sequence

import torch


def temporal_depth_fuse(
    target_depth: torch.Tensor,
    temporal_depths: Sequence[torch.Tensor],
    target_valid: torch.Tensor,
    temporal_valids: Sequence[torch.Tensor],
    target_confidence: torch.Tensor,
    temporal_confidences: Sequence[torch.Tensor],
    *,
    visible: Sequence[torch.Tensor] | None = None,
    consistent: Sequence[torch.Tensor] | None = None,
    trusted: Sequence[torch.Tensor] | None = None,
    target_trusted: torch.Tensor | None = None,
    consistency_threshold: float = 0.05,
    min_support: int = 2,
    min_temporal_support: int = 1,
) -> dict[str, torch.Tensor]:
    """Fuse target and already-reprojected temporal depth observations."""
    if target_depth.ndim != 4 or target_depth.shape[1] != 1 or not target_depth.is_floating_point():
        raise ValueError("target_depth must be floating point [B,1,H,W]")
    n = len(temporal_depths)
    if n == 0 or len(temporal_valids) != n or len(temporal_confidences) != n:
        raise ValueError("temporal fields must be non-empty and have matching lengths")
    if visible is not None and len(visible) != n:
        raise ValueError("visible must match temporal_depths")
    if consistent is not None and len(consistent) != n:
        raise ValueError("consistent must match temporal_depths")
    if trusted is not None and len(trusted) != n:
        raise ValueError("trusted must match temporal_depths")
    if (
        consistency_threshold <= 0
        or not isinstance(min_support, int)
        or isinstance(min_support, bool)
        or min_support < 1
        or not isinstance(min_temporal_support, int)
        or isinstance(min_temporal_support, bool)
        or min_temporal_support < 0
    ):
        raise ValueError("positive consistency_threshold and non-negative integer support requirements are required")
    shape, device, dtype = target_depth.shape, target_depth.device, target_depth.dtype

    def check(depth: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor) -> None:
        if depth.shape != shape or depth.device != device or depth.dtype != dtype:
            raise ValueError("all depth fields must share shape, device, and dtype")
        if valid.shape != shape or valid.dtype != torch.bool or valid.device != device:
            raise ValueError("valid masks must be bool and match depth")
        if (
            confidence.shape != shape
            or confidence.dtype != dtype
            or confidence.device != device
            or not torch.isfinite(confidence).all()
            or (confidence < 0).any()
            or (confidence > 1).any()
        ):
            raise ValueError("confidences must be finite [0,1] and match depth")

    check(target_depth, target_valid, target_confidence)
    for depth, valid, confidence in zip(temporal_depths, temporal_valids, temporal_confidences):
        check(depth, valid, confidence)

    def check_evidence(items: Sequence[torch.Tensor] | None, name: str) -> list[torch.Tensor]:
        values = list(items) if items is not None else [torch.ones(shape, dtype=torch.bool, device=device)] * n
        for item in values:
            if item.shape != shape or item.dtype != torch.bool or item.device != device:
                raise ValueError(f"{name} masks must be bool and match depth")
        return values

    visibility = check_evidence(visible, "visible")
    consistency = check_evidence(consistent, "consistent")
    trust = list(trusted) if trusted is not None else [torch.zeros(shape, dtype=torch.bool, device=device)] * n
    for item in trust:
        if item.shape != shape or item.dtype != torch.bool or item.device != device:
            raise ValueError("trusted masks must be bool and match depth")
    target_trust = torch.zeros_like(target_valid) if target_trusted is None else target_trusted
    if target_trust.shape != shape or target_trust.dtype != torch.bool or target_trust.device != device:
        raise ValueError("target_trusted must be bool and match depth")

    depths = torch.stack((target_depth, *temporal_depths), dim=0)
    valids = torch.stack((target_valid, *temporal_valids), dim=0)
    confidences = torch.stack((target_confidence, *temporal_confidences), dim=0)
    evidence = torch.stack(
        (torch.ones_like(target_valid), *(vis & con for vis, con in zip(visibility, consistency))),
        dim=0,
    )
    usable = valids & evidence & torch.isfinite(depths) & (depths > 0) & (confidences > 0)
    trusts = torch.stack((target_trust, *trust), dim=0)
    score = torch.where(trusts, 1 + confidences, confidences * 0.5)
    score = torch.where(usable, score, torch.zeros_like(score))
    trusted_winner = score.argmax(0)
    pairwise_agreement = usable[:, None] & usable[None] & (
        (depths[:, None] - depths[None]).abs()
        <= consistency_threshold * depths[:, None].clamp_min(torch.finfo(dtype).eps)
    )
    cluster_score = (pairwise_agreement * confidences[None]).sum(1)
    consensus_winner = cluster_score.argmax(0)
    winner = torch.where((trusts & usable).any(0), trusted_winner, consensus_winner)
    reference = torch.gather(depths, 0, winner.unsqueeze(0)).squeeze(0)
    agrees = usable & (
        (depths - reference.unsqueeze(0)).abs()
        <= consistency_threshold * reference.unsqueeze(0).clamp_min(torch.finfo(dtype).eps)
    )
    support_count = agrees.sum(0).to(torch.int64)
    temporal_support_count = agrees[1:].sum(0).to(torch.int64)
    weights = torch.where(agrees, confidences, torch.zeros_like(confidences))
    weight_sum = weights.sum(0)
    dominant = weights.argmax(0).to(torch.int64)
    temporal_weight_fraction = weights[1:].sum(0) / weight_sum.clamp_min(torch.finfo(dtype).eps)
    safe_depths = torch.where(agrees, depths, torch.zeros_like(depths))
    fused = (weights * safe_depths).sum(0) / weight_sum.clamp_min(torch.finfo(dtype).eps)
    residual = torch.where(agrees, depths - fused.unsqueeze(0), torch.zeros_like(depths))
    variance = (weights * residual.square()).sum(0) / weight_sum.clamp_min(torch.finfo(dtype).eps)
    disagreement = torch.where(
        usable,
        (depths - reference.unsqueeze(0)).abs(),
        torch.zeros_like(depths),
    ).amax(0)
    trusted_support = (agrees & trusts).any(0)
    consensus_support = (support_count >= min_support) & (temporal_support_count >= min_temporal_support)
    out_valid = trusted_support | consensus_support
    out_depth = torch.where(out_valid, fused, torch.zeros_like(fused))
    out_confidence = torch.where(out_valid, weight_sum.clamp_max(1), torch.zeros_like(weight_sum))
    variance = torch.where(out_valid, variance, torch.zeros_like(variance))
    invalid_source = torch.full_like(dominant, -1)
    dominant = torch.where(out_valid, dominant, invalid_source)
    conflict_owner = torch.where(out_valid, winner.to(torch.int64), invalid_source)
    return {
        "depth": out_depth,
        "valid": out_valid,
        "confidence": out_confidence,
        "support_count": support_count,
        "temporal_support_count": temporal_support_count,
        "depth_variance": variance,
        "winning_source": dominant,
        "conflict_owner_source": conflict_owner,
        "dominant_contributor": dominant,
        "temporal_weight_fraction": torch.where(out_valid, temporal_weight_fraction, torch.zeros_like(temporal_weight_fraction)),
        "source_disagreement": torch.where(out_valid, disagreement, torch.zeros_like(disagreement)),
    }
