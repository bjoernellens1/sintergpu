# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
"""Fixed-capacity sparse 3D landmark pose normal-equation reduction."""

from __future__ import annotations

import torch

from .sparse_geometry import _pose


def sparse_pose_reduce(
    xyz_world: torch.Tensor,
    observed_xyz_camera: torch.Tensor,
    pose_c2w: torch.Tensor,
    *,
    matched: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    partial_count: int = 1,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Reduce matched world-to-camera 3D residuals into fixed SE(3) partials.

    ``pose_c2w`` is the candidate camera pose.  Residuals are evaluated in the
    candidate camera frame as ``R.T @ (xyz_world - t) - observed_xyz_camera``.
    Updates use camera-local translation then rotation ordering.
    """
    if validate and (
        xyz_world.ndim != 3
        or xyz_world.shape[-1] != 3
        or not xyz_world.is_floating_point()
        or observed_xyz_camera.shape != xyz_world.shape
        or observed_xyz_camera.device != xyz_world.device
        or observed_xyz_camera.dtype != xyz_world.dtype
        or not isinstance(partial_count, int)
        or isinstance(partial_count, bool)
        or partial_count < 1
    ):
        raise ValueError("XYZ inputs must be matching floating [B,P,3] with positive partial_count")
    batch, points, _ = xyz_world.shape
    if validate and (batch < 1 or points < 1):
        raise ValueError("XYZ batch and point axes must be non-empty")
    if validate:
        _pose(pose_c2w, batch, xyz_world.device, xyz_world.dtype)
    if matched is None:
        matched = torch.ones((batch, points), dtype=torch.bool, device=xyz_world.device)
    elif validate and (
        matched.shape != (batch, points) or matched.dtype != torch.bool or matched.device != xyz_world.device
    ):
        raise ValueError("matched must be bool [B,P] on the XYZ device")
    if confidence is None:
        confidence = torch.ones((batch, points), dtype=xyz_world.dtype, device=xyz_world.device)
    elif validate and (
        confidence.shape != (batch, points)
        or confidence.device != xyz_world.device
        or confidence.dtype != xyz_world.dtype
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError("confidence must be finite [0,1] [B,P] on the XYZ device")

    point = torch.einsum(
        "bij,bpj->bpi",
        pose_c2w[:, :3, :3].transpose(1, 2),
        xyz_world - pose_c2w[:, None, :3, 3],
    )
    residual = point - observed_xyz_camera
    valid = (
        matched
        & torch.isfinite(xyz_world).all(-1)
        & torch.isfinite(observed_xyz_camera).all(-1)
        & torch.isfinite(point).all(-1)
        & (point[..., 2] > 0)
        & (observed_xyz_camera[..., 2] > 0)
    )
    weight = torch.where(valid, confidence, torch.zeros_like(confidence))
    partial_H: list[torch.Tensor] = []
    partial_b: list[torch.Tensor] = []
    partial_valid: list[torch.Tensor] = []
    for index in range(partial_count):
        start, end = index * points // partial_count, (index + 1) * points // partial_count
        p, r, w = point[:, start:end], residual[:, start:end], weight[:, start:end]
        x, y, z = p.unbind(-1)
        weighted_x, weighted_y, weighted_z = (w * x).sum(1), (w * y).sum(1), (w * z).sum(1)
        weighted_xx, weighted_yy, weighted_zz = (w * x.square()).sum(1), (w * y.square()).sum(1), (w * z.square()).sum(1)
        weighted_xy, weighted_xz, weighted_yz = (w * x * y).sum(1), (w * x * z).sum(1), (w * y * z).sum(1)
        H = torch.zeros((batch, 6, 6), device=xyz_world.device, dtype=xyz_world.dtype)
        H[:, 0, 0] = H[:, 1, 1] = H[:, 2, 2] = w.sum(1)
        H[:, :3, 3:] = torch.stack(
            (
                torch.stack((torch.zeros_like(weighted_z), weighted_z, -weighted_y), -1),
                torch.stack((-weighted_z, torch.zeros_like(weighted_x), weighted_x), -1),
                torch.stack((weighted_y, -weighted_x, torch.zeros_like(weighted_x)), -1),
            ),
            1,
        )
        H[:, 3:, :3] = H[:, :3, 3:].transpose(1, 2)
        H[:, 3:, 3:] = torch.stack(
            (
                torch.stack((weighted_yy + weighted_zz, -weighted_xy, -weighted_xz), -1),
                torch.stack((-weighted_xy, weighted_xx + weighted_zz, -weighted_yz), -1),
                torch.stack((-weighted_xz, -weighted_yz, weighted_xx + weighted_yy), -1),
            ),
            1,
        )
        weighted_residual = w[..., None] * r
        # ``pose_c2w`` is updated in its camera-local frame as
        # ``pose_c2w @ exp(delta^)``.  World-to-camera residuals therefore
        # have the opposite sign to source-camera-to-reference residuals in
        # dense_track_reduce; negate J^T r here so both reducers share the
        # same solved camera-to-world update convention.
        b = -torch.cat((weighted_residual.sum(1), (w[..., None] * torch.cross(p, r, dim=-1)).sum(1)), -1)
        partial_H.append(H)
        partial_b.append(b)
        partial_valid.append(valid[:, start:end].any(1))
    partial_H_out = torch.stack(partial_H, 1)
    partial_b_out = torch.stack(partial_b, 1)
    partial_valid_out = torch.stack(partial_valid, 1)
    count = valid.sum(1).to(torch.int64)
    residual_l2 = residual.square().sum(-1).sqrt()
    return {
        "H": partial_H_out.sum(1),
        "b": partial_b_out.sum(1),
        "inlier_count": count,
        "residual_l2_mean": (residual_l2 * valid).sum(1) / count.to(xyz_world.dtype).clamp_min(1),
        "partial_H": partial_H_out,
        "partial_b": partial_b_out,
        "partial_valid": partial_valid_out,
    }
