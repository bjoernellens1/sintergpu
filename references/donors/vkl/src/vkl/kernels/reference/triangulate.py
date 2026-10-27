"""Compact two-view geometric depth triangulation."""

from __future__ import annotations

import torch


def _check_camera(
    intrinsics: torch.Tensor,
    pose: torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if (
        intrinsics.shape != (batch, 3, 3)
        or pose.shape != (batch, 4, 4)
        or intrinsics.device != device
        or pose.device != device
        or intrinsics.dtype != dtype
        or pose.dtype != dtype
    ):
        raise ValueError("intrinsics and poses must be [B,3,3]/[B,4,4] on the correspondence device and dtype")
    if (
        not torch.isfinite(intrinsics).all()
        or (intrinsics[:, 0, 0] <= 0).any()
        or (intrinsics[:, 1, 1] <= 0).any()
    ):
        raise ValueError("intrinsics must be finite with positive focal lengths")
    if not torch.isfinite(pose).all():
        raise ValueError("poses must be finite")
    rotation = pose[:, :3, :3]
    identity = torch.eye(3, device=device, dtype=dtype).expand(batch, -1, -1)
    if (
        not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=1e-4)
        or (torch.linalg.det(rotation) <= 0).any()
        or not torch.allclose(
            pose[:, 3],
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(batch, -1),
            atol=1e-6,
            rtol=0,
        )
    ):
        raise ValueError("poses must be rigid homogeneous camera-to-world transforms")


def _rays(
    pixels: torch.Tensor,
    intrinsics: torch.Tensor,
    pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    camera = torch.einsum("bij,bnj->bni", torch.linalg.inv(intrinsics), homogeneous)
    camera = torch.nn.functional.normalize(camera, dim=-1)
    return torch.einsum("bij,bnj->bni", pose[:, :3, :3], camera), pose[:, None, :3, 3]


def _project(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = torch.einsum(
        "bij,bnj->bni",
        pose[:, :3, :3].transpose(1, 2),
        points_world - pose[:, None, :3, 3],
    )
    z = camera[..., 2]
    safe_z = z.clamp_min(torch.finfo(z.dtype).eps)
    pixel = torch.stack(
        (
            intrinsics[:, None, 0, 0] * camera[..., 0] / safe_z + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * camera[..., 1] / safe_z + intrinsics[:, None, 1, 2],
        ),
        dim=-1,
    )
    return pixel, z


def triangulate_depth(
    pixels_a: torch.Tensor,
    pixels_b: torch.Tensor,
    intrinsics_a: torch.Tensor,
    intrinsics_b: torch.Tensor,
    pose_a_c2w: torch.Tensor,
    pose_b_c2w: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    min_angle_deg: float = 1.0,
    metric_status: str = "METRIC_ANCHORED",
) -> dict[str, torch.Tensor | str]:
    """Triangulate compact ``[B,N,2]`` pixel correspondences from c2w poses."""
    if (
        not isinstance(pixels_a, torch.Tensor)
        or pixels_a.ndim != 3
        or pixels_a.shape[-1] != 2
        or not pixels_a.is_floating_point()
        or pixels_a.shape != pixels_b.shape
        or not pixels_b.is_floating_point()
    ):
        raise ValueError("pixels_a and pixels_b must be floating [B,N,2] tensors")
    batch, count, _ = pixels_a.shape
    if pixels_b.device != pixels_a.device or pixels_b.dtype != pixels_a.dtype:
        raise ValueError("correspondence tensors must share device and dtype")
    if not torch.isfinite(pixels_a).all() or not torch.isfinite(pixels_b).all():
        raise ValueError("correspondence tensors must be finite")
    _check_camera(intrinsics_a, pose_a_c2w, batch, pixels_a.device, pixels_a.dtype)
    _check_camera(intrinsics_b, pose_b_c2w, batch, pixels_a.device, pixels_a.dtype)
    if confidence is None:
        confidence = torch.ones((batch, count), device=pixels_a.device, dtype=pixels_a.dtype)
    if (
        confidence.shape != (batch, count)
        or confidence.device != pixels_a.device
        or confidence.dtype != pixels_a.dtype
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError("confidence must be finite [B,N] in [0,1]")
    if min_angle_deg <= 0 or metric_status not in {"RELATIVE", "METRIC_ANCHORED"}:
        raise ValueError("min_angle_deg must be positive and metric_status must be RELATIVE or METRIC_ANCHORED")

    direction_a, center_a = _rays(pixels_a, intrinsics_a, pose_a_c2w)
    direction_b, center_b = _rays(pixels_b, intrinsics_b, pose_b_c2w)
    between = center_b - center_a
    dot = (direction_a * direction_b).sum(-1).clamp(-1, 1)
    denom = 1 - dot.square()
    da_between = (direction_a * between).sum(-1)
    db_between = (direction_b * between).sum(-1)
    ta = (da_between - dot * db_between) / denom.clamp_min(torch.finfo(pixels_a.dtype).eps)
    tb = (dot * da_between - db_between) / denom.clamp_min(torch.finfo(pixels_a.dtype).eps)
    point_a = center_a + ta[..., None] * direction_a
    point_b = center_b + tb[..., None] * direction_b
    point = (point_a + point_b) * 0.5
    reproject_a, depth = _project(point, intrinsics_a, pose_a_c2w)
    reproject_b, depth_b = _project(point, intrinsics_b, pose_b_c2w)
    reprojection_error = (
        (reproject_a - pixels_a).square().sum(-1)
        + (reproject_b - pixels_b).square().sum(-1)
    ).sqrt()
    angle = torch.rad2deg(torch.acos(dot.abs()))
    valid = (
        torch.isfinite(point).all(-1)
        & (ta > 0)
        & (tb > 0)
        & (depth > 0)
        & (depth_b > 0)
        & (angle >= min_angle_deg)
    )
    output_confidence = (
        confidence
        * torch.sin(torch.deg2rad(angle)).clamp(0, 1)
        * torch.exp(-reprojection_error)
    ).clamp(0, 1)
    return {
        "depth": torch.where(valid, depth, torch.zeros_like(depth)),
        "inverse_depth": torch.where(valid, depth.reciprocal(), torch.zeros_like(depth)),
        "valid": valid,
        "confidence": torch.where(valid, output_confidence, torch.zeros_like(output_confidence)),
        "reprojection_error": torch.where(valid, reprojection_error, torch.zeros_like(reprojection_error)),
        "triangulation_angle_deg": torch.where(valid, angle, torch.zeros_like(angle)),
        "metric_status": metric_status,
    }
