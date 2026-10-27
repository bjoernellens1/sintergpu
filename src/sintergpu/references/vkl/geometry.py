# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_geometry(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_c2w: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    *,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Camera axes are +X right, +Y image-down, +Z forward; normals face camera.

    Invalid depth never contributes a 3D coordinate. Normals and
    geometry_confidence are zero at invalid, discontinuity, and rightmost/bottom
    stencil-boundary pixels.
    """
    if validate and (
        depth.ndim != 4
        or depth.shape[1] != 1
        or not depth.is_floating_point()
        or intrinsics.shape != (depth.shape[0], 3, 3)
        or intrinsics.device != depth.device
        or intrinsics.dtype != depth.dtype
    ):
        raise ValueError("depth must be [B,1,H,W] and intrinsics [B,3,3]")
    if validate and (
        not torch.isfinite(intrinsics).all()
        or (intrinsics[:, 0, 0] <= 0).any()
        or (intrinsics[:, 1, 1] <= 0).any()
    ):
        raise ValueError("intrinsics must be finite with positive focal lengths")

    b, _, h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    rays = torch.stack(
        (
            (x[None] - intrinsics[:, None, None, 0, 2]) / fx[:, None, None],
            (y[None] - intrinsics[:, None, None, 1, 2]) / fy[:, None, None],
            torch.ones((b, h, w), device=depth.device, dtype=depth.dtype),
        ),
        dim=1,
    )
    valid = torch.isfinite(depth) & (depth > 0)
    xyz = rays * torch.where(valid, depth, torch.zeros_like(depth))
    dx = F.pad(xyz[..., :, 1:] - xyz[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(xyz[..., 1:, :] - xyz[..., :-1, :], (0, 0, 0, 1))
    stencil = (
        valid
        & F.pad(valid[..., :, 1:], (0, 1, 0, 0))
        & F.pad(valid[..., 1:, :], (0, 0, 0, 1))
    )
    stencil &= (F.pad(depth[..., :, 1:], (0, 1, 0, 0)) - depth).abs() <= depth.abs() * 0.25
    stencil &= (F.pad(depth[..., 1:, :], (0, 0, 0, 1)) - depth).abs() <= depth.abs() * 0.25
    normals = F.normalize(torch.cross(dx, dy, dim=1), dim=1, eps=1e-8)
    normals = torch.where((normals * rays).sum(1, keepdim=True) > 0, -normals, normals)
    normals = torch.where(stencil.expand_as(normals), normals, torch.zeros_like(normals))
    geometry_confidence = stencil.to(depth.dtype)
    if confidence is not None:
        if validate and (confidence.shape != depth.shape or confidence.device != depth.device or confidence.dtype != depth.dtype):
            raise ValueError("confidence must match depth")
        geometry_confidence = geometry_confidence * confidence

    out = {
        "valid": valid,
        "rays_camera": rays,
        "xyz_camera": xyz,
        "normals": normals,
        "geometry_confidence": geometry_confidence,
    }
    if pose_c2w is not None:
        xyz_world = world_transform(xyz, pose_c2w)["xyz_world"]
        out["xyz_world"] = torch.where(valid.expand_as(xyz_world), xyz_world, torch.zeros_like(xyz_world))
    return out


def world_transform(xyz_camera: torch.Tensor, pose_c2w: torch.Tensor) -> dict[str, torch.Tensor]:
    if (
        xyz_camera.ndim != 4
        or not xyz_camera.is_floating_point()
        or pose_c2w.shape != (xyz_camera.shape[0], 4, 4)
        or pose_c2w.device != xyz_camera.device
        or pose_c2w.dtype != xyz_camera.dtype
    ):
        raise ValueError("xyz_camera must be [B,3,H,W] and pose_c2w [B,4,4]")
    if not torch.isfinite(pose_c2w).all():
        raise ValueError("pose_c2w must be finite")
    xyz_world = (
        torch.einsum("bij,bjhw->bihw", pose_c2w[:, :3, :3], xyz_camera)
        + pose_c2w[:, :3, 3, None, None]
    )
    return {"xyz_world": xyz_world}
