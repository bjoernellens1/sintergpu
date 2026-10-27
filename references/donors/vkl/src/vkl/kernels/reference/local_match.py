"""Pose-gated local correspondence references."""

from __future__ import annotations

import math

import torch


def binary_local_match(
    landmark_descriptor: torch.Tensor,
    candidate_descriptor: torch.Tensor,
    projected_xy: torch.Tensor,
    candidate_xy: torch.Tensor,
    *,
    radius_px: float,
    max_hamming_distance: int,
    mutual: bool = True,
    landmark_valid: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
    candidate_chunk_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Choose descriptor-qualified candidates inside projected pixel gates.

    Matching is mutual by default so one candidate cannot silently own multiple
    persistent landmark tracks. ``max_hamming_distance`` is explicit rather
    than hidden policy. ``candidate_chunk_size`` is an exact memory-control knob:
    chunking changes only temporary allocation, not matching semantics.
    """
    if (
        landmark_descriptor.ndim != 3
        or landmark_descriptor.dtype != torch.bool
        or candidate_descriptor.ndim != 3
        or candidate_descriptor.dtype != torch.bool
    ):
        raise ValueError("descriptors must be bool [B,M|P,D]")
    batch, landmarks, bits = landmark_descriptor.shape
    candidates = candidate_descriptor.shape[1]
    if batch < 1 or landmarks < 1 or bits < 1 or candidates < 1:
        raise ValueError("descriptors must contain non-empty batch, landmark, candidate, and bit axes")
    if (
        candidate_descriptor.shape[0] != batch
        or candidate_descriptor.shape[2] != bits
        or projected_xy.shape != (batch, landmarks, 2)
        or candidate_xy.shape != (batch, candidates, 2)
        or not projected_xy.is_floating_point()
        or candidate_xy.dtype != torch.int64
    ):
        raise ValueError("descriptor batches, projected_xy [B,M,2], and candidate_xy [B,P,2] must match")
    if any(
        item.device != landmark_descriptor.device
        for item in (candidate_descriptor, projected_xy, candidate_xy)
    ):
        raise ValueError("all tensors must share device")
    if (
        isinstance(radius_px, bool)
        or not isinstance(radius_px, (int, float))
        or not math.isfinite(float(radius_px))
        or radius_px <= 0
    ):
        raise ValueError("radius_px must be finite and positive")
    if (
        isinstance(max_hamming_distance, bool)
        or not isinstance(max_hamming_distance, int)
        or not 0 <= max_hamming_distance <= bits
    ):
        raise ValueError("max_hamming_distance must be an integer in [0,D]")
    if not isinstance(mutual, bool):
        raise ValueError("mutual must be bool")
    if candidate_chunk_size is not None and (
        isinstance(candidate_chunk_size, bool)
        or not isinstance(candidate_chunk_size, int)
        or candidate_chunk_size < 1
    ):
        raise ValueError("candidate_chunk_size must be a positive integer or None")

    if landmark_valid is None:
        landmark_valid = torch.ones(
            (batch, landmarks), dtype=torch.bool, device=landmark_descriptor.device
        )
    elif (
        landmark_valid.shape != (batch, landmarks)
        or landmark_valid.dtype != torch.bool
        or landmark_valid.device != landmark_descriptor.device
    ):
        raise ValueError("landmark_valid must be bool [B,M] on descriptor device")
    if candidate_valid is None:
        candidate_valid = torch.ones(
            (batch, candidates), dtype=torch.bool, device=landmark_descriptor.device
        )
    elif (
        candidate_valid.shape != (batch, candidates)
        or candidate_valid.dtype != torch.bool
        or candidate_valid.device != landmark_descriptor.device
    ):
        raise ValueError("candidate_valid must be bool [B,P] on descriptor device")

    landmark_valid = landmark_valid & torch.isfinite(projected_xy).all(-1)
    chunk_size = candidates if candidate_chunk_size is None else min(candidate_chunk_size, candidates)
    best_distance = torch.full(
        (batch, landmarks), bits + 1, dtype=torch.int64, device=landmark_descriptor.device
    )
    best_index = torch.zeros(
        (batch, landmarks), dtype=torch.int64, device=landmark_descriptor.device
    )
    reverse_chunks: list[torch.Tensor] = []

    for start in range(0, candidates, chunk_size):
        end = min(start + chunk_size, candidates)
        candidate_xy_chunk = candidate_xy[:, start:end]
        local = (
            (
                projected_xy[:, :, None]
                - candidate_xy_chunk[:, None].to(projected_xy.dtype)
            )
            .square()
            .sum(-1)
            <= radius_px * radius_px
        )
        local &= (
            landmark_valid[:, :, None]
            & candidate_valid[:, None, start:end]
        )
        distance = (
            landmark_descriptor[:, :, None]
            ^ candidate_descriptor[:, None, start:end]
        ).sum(-1)
        gated_distance = distance.masked_fill(~local, bits + 1)
        chunk_distance, chunk_offset = gated_distance.min(-1)
        chunk_index = chunk_offset + start
        improve = chunk_distance < best_distance
        best_distance = torch.where(improve, chunk_distance, best_distance)
        best_index = torch.where(improve, chunk_index, best_index)
        if mutual:
            reverse_chunks.append(gated_distance.min(1).indices)

    valid = landmark_valid & (best_distance <= max_hamming_distance)
    if mutual:
        reverse_best = torch.cat(reverse_chunks, dim=1)
        landmark_index = torch.arange(
            landmarks, device=landmark_descriptor.device
        )[None].expand(batch, -1)
        valid &= reverse_best.gather(1, best_index) == landmark_index

    return {
        "matched_index": torch.where(valid, best_index, torch.full_like(best_index, -1)),
        "distance": torch.where(valid, best_distance, torch.zeros_like(best_distance)),
        "valid": valid,
    }
