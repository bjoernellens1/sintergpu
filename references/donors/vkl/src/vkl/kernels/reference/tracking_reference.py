"""Fixed-shape F2M/F2F reference selection and camera-local pose updates."""

from __future__ import annotations

import torch


def select_tracking_reference(
    *,
    f2m_partial_H: torch.Tensor,
    f2m_partial_b: torch.Tensor,
    f2m_partial_valid: torch.Tensor,
    f2m_usable: torch.Tensor,
    f2f_partial_H: torch.Tensor,
    f2f_partial_b: torch.Tensor,
    f2f_partial_valid: torch.Tensor,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Choose fixed F2M evidence when usable, otherwise fixed F2F evidence."""
    if validate and (
        f2m_partial_H.ndim != 4
        or f2m_partial_H.shape[-2:] != (6, 6)
        or f2m_partial_b.shape != f2m_partial_H.shape[:-1]
        or f2m_partial_valid.shape != f2m_partial_H.shape[:2]
        or f2f_partial_H.shape != f2m_partial_H.shape
        or f2f_partial_b.shape != f2m_partial_b.shape
        or f2f_partial_valid.shape != f2m_partial_valid.shape
        or f2m_usable.shape != f2m_partial_H.shape[:1]
        or f2m_partial_valid.dtype != torch.bool
        or f2f_partial_valid.dtype != torch.bool
        or f2m_usable.dtype != torch.bool
        or any(
            value.device != f2m_partial_H.device
            for value in (f2m_partial_b, f2m_partial_valid, f2m_usable, f2f_partial_H, f2f_partial_b, f2f_partial_valid)
        )
        or any(value.dtype != f2m_partial_H.dtype for value in (f2m_partial_b, f2f_partial_H, f2f_partial_b))
    ):
        raise ValueError("F2M/F2F partial tensors and f2m_usable must have matching fixed shapes")
    use_f2m = f2m_usable[:, None]
    return {
        "partial_H": torch.where(use_f2m[:, :, None, None], f2m_partial_H, f2f_partial_H),
        "partial_b": torch.where(use_f2m[:, :, None], f2m_partial_b, f2f_partial_b),
        "partial_valid": torch.where(use_f2m, f2m_partial_valid, f2f_partial_valid),
        "f2f_selected": ~f2m_usable,
    }


def apply_camera_local_tracking_update(
    pose_c2w: torch.Tensor,
    delta: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Apply a TR1/TR2 camera-local update to a camera-to-world pose.

    Dense and sparse reducers both emit the camera-to-world local convention:
    ``pose_c2w @ exp(delta^)``.
    """
    if validate and (
        pose_c2w.ndim != 3
        or pose_c2w.shape[-2:] != (4, 4)
        or delta.shape != (pose_c2w.shape[0], 6)
        or not pose_c2w.is_floating_point()
        or delta.device != pose_c2w.device
        or delta.dtype != pose_c2w.dtype
        or not torch.isfinite(pose_c2w).all()
        or not torch.isfinite(delta).all()
    ):
        raise ValueError("pose_c2w and delta must be finite compatible [B,4,4]/[B,6] tensors")
    batch = pose_c2w.shape[0]
    vx, vy, vz, wx, wy, wz = delta.unbind(-1)
    tangent = torch.zeros((batch, 4, 4), device=pose_c2w.device, dtype=pose_c2w.dtype)
    tangent[:, 0, 1], tangent[:, 0, 2] = -wz, wy
    tangent[:, 1, 0], tangent[:, 1, 2] = wz, -wx
    tangent[:, 2, 0], tangent[:, 2, 1] = -wy, wx
    tangent[:, :3, 3] = torch.stack((vx, vy, vz), -1)
    return pose_c2w @ torch.linalg.matrix_exp(tangent)
