# SPDX-License-Identifier: Apache-2.0
# Original VKL owner code, relicensed by explicit owner authorization 2026-10-27.
from __future__ import annotations

from collections.abc import Mapping

import torch


def _floating_batch(
    batch: Mapping[str, torch.Tensor],
    name: str,
    *,
    batch_size: int,
    tail: tuple[int, ...] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if name not in batch:
        return None
    value = batch[name]
    if (
        not isinstance(value, torch.Tensor)
        or value.shape[0] != batch_size
        or not value.is_floating_point()
        or value.device != device
        or value.dtype != dtype
        or (tail is not None and tuple(value.shape[1:]) != tail)
    ):
        raise ValueError(f"{name} has invalid shape/device/dtype")
    return value


def _pinhole(value: torch.Tensor | None, name: str) -> None:
    if value is None:
        return
    if (
        not torch.isfinite(value).all()
        or (value[:, 0, 0] <= 0).any()
        or (value[:, 1, 1] <= 0).any()
        or (value[:, 2, 2] != 1).any()
        or (value[:, 0, 1] != 0).any()
        or (value[:, 1, 0] != 0).any()
        or (value[:, 2, :2] != 0).any()
    ):
        raise ValueError(f"{name} must be finite canonical pinhole intrinsics")


def _rigid(value: torch.Tensor | None, name: str) -> None:
    if value is None:
        return
    batch_size = value.shape[0]
    rotation = value[:, :3, :3]
    identity = torch.eye(3, device=value.device, dtype=value.dtype).expand(batch_size, -1, -1)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=value.device, dtype=value.dtype).expand(batch_size, -1)
    if (
        not torch.isfinite(value).all()
        or not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=1e-4)
        or (torch.linalg.det(rotation) <= 0).any()
        or not torch.allclose(value[:, 3], bottom, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a rigid homogeneous transform")


def validate_frame_batch(batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Validate the capability-composed FrameBatch boundary without copying.

    Raw RGB and depth may have different spatial resolutions before camera
    conditioning. Image-space tensors/calibration share RGB device/dtype, while
    timestamps are float64 so absolute dataset clocks retain frame-scale
    precision independently of image precision.
    """
    if "rgb" not in batch:
        raise ValueError("frame batch requires rgb")
    rgb = batch["rgb"]
    if (
        not isinstance(rgb, torch.Tensor)
        or rgb.ndim != 4
        or rgb.shape[0] < 1
        or rgb.shape[1] != 3
        or min(rgb.shape[-2:]) < 1
        or not rgb.is_floating_point()
    ):
        raise ValueError("rgb must have non-empty floating shape [B,3,H,W]")
    batch_size = rgb.shape[0]
    device, dtype = rgb.device, rgb.dtype

    depth = None
    if "depth" in batch:
        depth = batch["depth"]
        if (
            not isinstance(depth, torch.Tensor)
            or depth.ndim != 4
            or depth.shape[0] != batch_size
            or depth.shape[1] != 1
            or min(depth.shape[-2:]) < 1
            or not depth.is_floating_point()
            or depth.device != device
            or depth.dtype != dtype
        ):
            raise ValueError("depth must be floating [B,1,Hd,Wd] on rgb device/dtype")

    cameras = {}
    for name in ("intrinsics", "depth_intrinsics", "target_intrinsics"):
        cameras[name] = _floating_batch(
            batch,
            name,
            batch_size=batch_size,
            tail=(3, 3),
            device=device,
            dtype=dtype,
        )
        _pinhole(cameras[name], name)

    transforms = {}
    for name in ("pose_c2w", "transform_rgb_depth"):
        transforms[name] = _floating_batch(
            batch,
            name,
            batch_size=batch_size,
            tail=(4, 4),
            device=device,
            dtype=dtype,
        )
        _rigid(transforms[name], name)

    if depth is None and (cameras["depth_intrinsics"] is not None or transforms["transform_rgb_depth"] is not None):
        raise ValueError("depth calibration requires depth")

    if "timestamp" in batch:
        timestamp = batch["timestamp"]
        if (
            not isinstance(timestamp, torch.Tensor)
            or timestamp.shape != (batch_size,)
            or timestamp.dtype != torch.float64
            or timestamp.device != device
            or not torch.isfinite(timestamp).all()
        ):
            raise ValueError("timestamp must be finite float64 [B] on the batch device")

    if "valid_pose" in batch:
        valid_pose = batch["valid_pose"]
        if (
            not isinstance(valid_pose, torch.Tensor)
            or valid_pose.shape != (batch_size,)
            or valid_pose.dtype != torch.bool
            or valid_pose.device != device
        ):
            raise ValueError("valid_pose must be bool [B] on the batch device")

    if "depth_scale" in batch and (
        not isinstance(batch["depth_scale"], (int, float)) or batch["depth_scale"] <= 0
    ):
        raise ValueError("depth_scale must be positive")
    return batch
