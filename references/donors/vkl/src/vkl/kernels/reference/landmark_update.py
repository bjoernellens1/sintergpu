"""Stateless fixed-capacity update for already-associated landmarks."""

from __future__ import annotations

import torch


def landmark_update(
    xyz_world: torch.Tensor,
    support_count: torch.Tensor,
    age: torch.Tensor,
    active: torch.Tensor,
    observation_xyz_world: torch.Tensor,
    matched: torch.Tensor,
    *,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Fuse matched world observations; lifetime and ID policy stay external."""
    if xyz_world.ndim != 3 or xyz_world.shape[-1] != 3 or not xyz_world.is_floating_point():
        raise ValueError("xyz_world must be floating [B,M,3]")
    batch, landmarks, _ = xyz_world.shape
    if (
        observation_xyz_world.shape != xyz_world.shape
        or observation_xyz_world.device != xyz_world.device
        or observation_xyz_world.dtype != xyz_world.dtype
        or support_count.shape != (batch, landmarks)
        or age.shape != (batch, landmarks)
        or support_count.dtype != torch.int64
        or age.dtype != torch.int64
        or any(value.device != xyz_world.device for value in (support_count, age, active, matched))
        or active.shape != (batch, landmarks)
        or matched.shape != (batch, landmarks)
        or active.dtype != torch.bool
        or matched.dtype != torch.bool
    ):
        raise ValueError("landmark state must have matching [B,M] tensors on the XYZ device")
    if not isinstance(validate, bool):
        raise ValueError("validate must be bool")
    if validate and ((support_count < 0).any() or (age < 0).any() or not torch.isfinite(xyz_world).all()):
        raise ValueError("landmark XYZ must be finite and support/age non-negative")
    actual_match = active & matched & torch.isfinite(observation_xyz_world).all(-1)
    weight = support_count.to(xyz_world.dtype)
    fused = (xyz_world * weight[..., None] + observation_xyz_world) / (weight + 1)[..., None]
    return {
        "xyz_world": torch.where(actual_match[..., None], fused, xyz_world),
        "support_count": support_count + actual_match.to(torch.int64),
        "age": age + active.to(torch.int64),
        "active": active,
        "matched": actual_match,
        "unmatched": active & ~actual_match,
    }
