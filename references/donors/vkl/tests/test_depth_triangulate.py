from __future__ import annotations

import pytest
import torch

from vkl.kernels import REGISTRY, triangulate_depth


def test_triangulate_depth_recovers_metric_plane_depth():
    intrinsics = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    pose_a = torch.eye(4).unsqueeze(0)
    pose_b = pose_a.clone()
    pose_b[:, 0, 3] = 0.3
    pixels_a = torch.tensor([[[50.0, 40.0], [60.0, 40.0]]])
    pixels_b = torch.tensor([[[40.0, 40.0], [50.0, 40.0]]])

    out = triangulate_depth(pixels_a, pixels_b, intrinsics, intrinsics, pose_a, pose_b)

    assert torch.all(out["valid"])
    assert torch.allclose(out["depth"], torch.full((1, 2), 3.0), atol=1e-4)
    assert out["metric_status"] == "METRIC_ANCHORED"
    assert REGISTRY["triangulate_depth"] is triangulate_depth


def test_triangulate_depth_rejects_parallel_rays():
    intrinsics = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    points = torch.zeros(1, 1, 2)

    out = triangulate_depth(points, points, intrinsics, intrinsics, pose, pose)

    assert not out["valid"].item()
    assert out["depth"].item() == 0.0


def test_triangulate_depth_rejects_invalid_camera_geometry():
    pixels = torch.zeros(1, 1, 2)
    intrinsics = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)

    bad_intrinsics = intrinsics.clone()
    bad_intrinsics[:, 0, 0] = 0
    with pytest.raises(ValueError):
        triangulate_depth(pixels, pixels, bad_intrinsics, intrinsics, pose, pose)

    bad_pose = pose.clone()
    bad_pose[:, 0, 0] = 2
    with pytest.raises(ValueError):
        triangulate_depth(pixels, pixels, intrinsics, intrinsics, bad_pose, pose)


def test_triangulate_depth_rejects_nonfinite_correspondence():
    pixels = torch.tensor([[[float("nan"), 0.0]]])
    intrinsics = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError):
        triangulate_depth(pixels, pixels, intrinsics, intrinsics, pose, pose)
