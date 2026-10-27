from __future__ import annotations

import math

import torch

from .sparse_geometry import _pose

REASON_TRANSLATION = 1 << 0
REASON_ROTATION = 1 << 1
REASON_FRAME_QUALITY = 1 << 2
REASON_LOCAL_TILE_VALUE = 1 << 3
REASON_MAX_GAP = 1 << 4
REASON_NOVELTY = 1 << 5


def keyframe_filter(
    *,
    frame_score: torch.Tensor,
    tile_score: torch.Tensor,
    pose: torch.Tensor | None = None,
    reference_pose: torch.Tensor | None = None,
    frame_gap: torch.Tensor | None = None,
    rgb_novelty: torch.Tensor | None = None,
    depth_novelty: torch.Tensor | None = None,
    geometric_novelty: torch.Tensor | None = None,
    has_rgb: torch.Tensor | None = None,
    has_depth: torch.Tensor | None = None,
    has_pose: torch.Tensor | None = None,
    min_translation_m: float = 0.08,
    min_rotation_deg: float = 6.0,
    min_frame_score: float = 0.35,
    min_tile_score: float = 0.65,
    min_novelty: float = 0.10,
    topk_tile_fraction: float = 0.10,
    max_gap: int = 30,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Compose modality evidence into a stateless admission decision.

    RGB quality gates RGB/motion-driven admission when RGB is present. Depth or
    geometric novelty remains independently admissible. High stationary quality
    alone never admits, and ``max_gap`` remains the starvation backstop.

    ``validate=False`` is the prevalidated hot path: static shape/device/dtype
    contracts remain enforced, while value-dependent tensor checks are skipped
    so K4 does not introduce GPU-to-CPU synchronization or compile graph breaks.
    """
    if (
        frame_score.ndim != 1
        or frame_score.shape[0] < 1
        or tile_score.ndim < 2
        or tile_score.shape[0] != frame_score.shape[0]
        or tile_score[0].numel() < 1
        or not frame_score.is_floating_point()
        or not tile_score.is_floating_point()
        or tile_score.device != frame_score.device
        or tile_score.dtype != frame_score.dtype
    ):
        raise ValueError("non-empty frame_score [B] and tile_score [B,...] must share floating dtype/device")
    thresholds = (
        min_translation_m,
        min_rotation_deg,
        min_frame_score,
        min_tile_score,
        min_novelty,
        topk_tile_fraction,
    )
    if (
        any(not math.isfinite(value) for value in thresholds)
        or min_translation_m < 0
        or min_rotation_deg < 0
        or not 0 <= min_frame_score <= 1
        or not 0 <= min_tile_score <= 1
        or not 0 <= min_novelty <= 1
        or not 0 < topk_tile_fraction <= 1
        or isinstance(max_gap, bool)
        or not isinstance(max_gap, int)
        or max_gap < 1
        or not isinstance(validate, bool)
    ):
        raise ValueError("keyframe thresholds, max_gap, and top-k fraction must be valid")
    if validate and (
        not torch.isfinite(frame_score).all() or not torch.isfinite(tile_score).all()
    ):
        raise ValueError("frame and tile scores must be finite")

    batch = frame_score.shape[0]

    def availability(value: torch.Tensor | None, default: bool) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), default, device=frame_score.device, dtype=torch.bool)
        if value.shape != (batch,) or value.dtype != torch.bool or value.device != frame_score.device:
            raise ValueError("availability must be matching bool [B]")
        return value

    has_rgb = availability(has_rgb, True)
    has_depth = availability(has_depth, False)
    has_pose = availability(has_pose, pose is not None and reference_pose is not None)
    if (pose is None) != (reference_pose is None) or (
        pose is not None
        and (
            pose.shape != (batch, 4, 4)
            or reference_pose.shape != pose.shape
            or pose.device != frame_score.device
            or reference_pose.device != frame_score.device
            or pose.dtype != frame_score.dtype
            or reference_pose.dtype != frame_score.dtype
        )
    ):
        raise ValueError("pose and reference_pose must be matching [B,4,4] or both absent")
    if validate and pose is not None:
        _pose(pose, batch, frame_score.device, frame_score.dtype)
        _pose(reference_pose, batch, frame_score.device, frame_score.dtype)

    if pose is None:
        translation_m = torch.zeros_like(frame_score)
        rotation_deg = torch.zeros_like(frame_score)
    else:
        translation_m = torch.linalg.vector_norm(pose[:, :3, 3] - reference_pose[:, :3, 3], dim=-1)
        r_rel = reference_pose[:, :3, :3].transpose(-1, -2) @ pose[:, :3, :3]
        cos_theta = ((torch.diagonal(r_rel, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        rotation_deg = torch.rad2deg(torch.acos(cos_theta))

    flat = tile_score.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * topk_tile_fraction)))
    topk_tile_mean = flat.topk(k, dim=1, largest=True, sorted=False).values.mean(dim=1)
    translation_hit = has_pose & (translation_m >= min_translation_m)
    rotation_hit = has_pose & (rotation_deg >= min_rotation_deg)
    frame_quality_hit = has_rgb & (frame_score >= min_frame_score)
    local_tile_hit = has_rgb & (topk_tile_mean >= min_tile_score)
    quality_gate = (~has_rgb) | frame_quality_hit | local_tile_hit

    def novelty(value: torch.Tensor | None, available: torch.Tensor) -> torch.Tensor:
        if value is None:
            return torch.zeros_like(frame_score)
        if value.shape != (batch,) or value.device != frame_score.device or value.dtype != frame_score.dtype:
            raise ValueError("novelty must be matching [B] score")
        if validate and not torch.isfinite(value).all():
            raise ValueError("novelty must be finite")
        return torch.where(available, value.clamp(0, 1), torch.zeros_like(value))

    rgb_novelty_score = novelty(rgb_novelty, has_rgb)
    depth_novelty_score = novelty(depth_novelty, has_depth)
    geometric_novelty_score = novelty(geometric_novelty, has_pose)
    novelty_score = torch.maximum(
        rgb_novelty_score,
        torch.maximum(depth_novelty_score, geometric_novelty_score),
    )
    rgb_novelty_hit = has_rgb & (rgb_novelty_score >= min_novelty)
    depth_novelty_hit = has_depth & (depth_novelty_score >= min_novelty)
    geometric_novelty_hit = has_pose & (geometric_novelty_score >= min_novelty)
    novelty_hit = rgb_novelty_hit | depth_novelty_hit | geometric_novelty_hit

    if frame_gap is None:
        gap_hit = torch.zeros_like(translation_hit)
        gap_pressure = torch.zeros_like(frame_score)
    else:
        if (
            frame_gap.shape != (batch,)
            or frame_gap.device != frame_score.device
            or frame_gap.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("frame_gap must be int32/int64 [B] on the score device")
        if validate and (frame_gap < 0).any():
            raise ValueError("frame_gap must be non-negative")
        gap_hit = frame_gap >= max_gap
        gap_pressure = (frame_gap.to(frame_score.dtype) / float(max_gap)).clamp(0.0, 1.0)

    motion_score = torch.maximum(
        (translation_m / max(min_translation_m, 1e-6)).clamp(0.0, 1.0),
        (rotation_deg / max(min_rotation_deg, 1e-6)).clamp(0.0, 1.0),
    )
    admit_score = (
        0.35 * motion_score
        + 0.25 * frame_score.clamp(0.0, 1.0)
        + 0.15 * topk_tile_mean.clamp(0.0, 1.0)
        + 0.20 * novelty_score
        + 0.05 * gap_pressure
    ).clamp(0.0, 1.0)

    rgb_trigger = translation_hit | rotation_hit | rgb_novelty_hit
    non_rgb_trigger = depth_novelty_hit | geometric_novelty_hit
    admission_trigger = rgb_trigger | non_rgb_trigger
    admit_mask = gap_hit | non_rgb_trigger | (rgb_trigger & quality_gate)

    bits = torch.zeros_like(frame_score, dtype=torch.int32)
    bits = (
        bits
        | translation_hit.to(torch.int32) * REASON_TRANSLATION
        | rotation_hit.to(torch.int32) * REASON_ROTATION
        | frame_quality_hit.to(torch.int32) * REASON_FRAME_QUALITY
        | local_tile_hit.to(torch.int32) * REASON_LOCAL_TILE_VALUE
        | gap_hit.to(torch.int32) * REASON_MAX_GAP
        | novelty_hit.to(torch.int32) * REASON_NOVELTY
    )
    return {
        "admit_mask": admit_mask,
        "reason_bits": bits,
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
        "topk_tile_mean": topk_tile_mean,
        "novelty_score": novelty_score,
        "rgb_novelty_score": rgb_novelty_score,
        "depth_novelty_score": depth_novelty_score,
        "geometric_novelty_score": geometric_novelty_score,
        "quality_gate": quality_gate,
        "rgb_trigger": rgb_trigger,
        "non_rgb_trigger": non_rgb_trigger,
        "admission_trigger": admission_trigger,
        "admit_score": admit_score,
    }
