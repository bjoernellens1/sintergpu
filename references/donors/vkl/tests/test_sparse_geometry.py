from __future__ import annotations

import pytest
import torch

from vkl.kernels import project_landmarks, sparse_depth_validate, sparse_rgbd_observation


def test_sparse_depth_validate_rejects_holes_and_depth_discontinuities():
    depth = torch.ones(1, 1, 5, 6)
    depth[0, 0, 2, 4] = 0
    depth[0, 0, 1:4, 1] = 4
    pixels = torch.tensor([[[3, 2], [4, 2], [1, 2]]])
    out = sparse_depth_validate(depth, pixels, max_relative_dispersion=0.1)
    assert out["valid"].tolist() == [[True, False, False]]
    assert torch.allclose(out["depth"][0, 0], torch.tensor(1.0))
    assert out["confidence"][0, 0] > 0


def test_sparse_depth_validate_preserves_upstream_point_validity():
    depth = torch.ones(1, 1, 5, 5)
    pixels = torch.tensor([[[2, 2], [2, 2]]])
    out = sparse_depth_validate(depth, pixels, pixels_valid=torch.tensor([[True, False]]))
    assert out["valid"].tolist() == [[True, False]]
    assert out["depth"].tolist() == [[1.0, 0.0]]


def test_sparse_depth_validate_does_not_duplicate_border_support():
    depth = torch.ones(1, 1, 4, 4)
    pixels = torch.tensor([[[0, 0], [1, 1]]])
    out = sparse_depth_validate(depth, pixels, min_neighbors=9)
    assert out["neighbor_count"].tolist() == [[4, 9]]
    assert out["valid"].tolist() == [[False, True]]


def test_sparse_rgbd_observation_projects_camera_and_world_coordinates():
    pixels = torch.tensor([[[2, 1], [4, 3]]])
    depth = torch.tensor([[2.0, 4.0]])
    intrinsics = torch.tensor([[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]]])
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 1
    out = sparse_rgbd_observation(pixels, depth, intrinsics, pose)
    assert torch.allclose(out["xyz_camera"], torch.tensor([[[2.0, 1.0, 2.0], [8.0, 6.0, 4.0]]]))
    assert torch.allclose(out["xyz_world"], torch.tensor([[[3.0, 1.0, 2.0], [9.0, 6.0, 4.0]]]))
    assert out["valid"].all()


def test_sparse_rgbd_observation_rejects_invalid_camera_geometry():
    pixels = torch.tensor([[[0, 0]]])
    depth = torch.ones(1, 1)
    intrinsics = torch.eye(3)[None]
    bad_intrinsics = intrinsics.clone(); bad_intrinsics[:, 0, 0] = 0
    with pytest.raises(ValueError): sparse_rgbd_observation(pixels, depth, bad_intrinsics)
    bad_pose = torch.eye(4)[None]; bad_pose[:, 0, 0] = 2
    with pytest.raises(ValueError): sparse_rgbd_observation(pixels, depth, intrinsics, bad_pose)


def test_project_landmarks_pose_gates_positive_depth_and_image_bounds():
    world = torch.tensor([[[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 0.0, -1.0]]])
    intrinsics = torch.tensor([[[2.0, 0, 2.0], [0, 2.0, 1.0], [0, 0, 1]]])
    out = project_landmarks(world, intrinsics, torch.eye(4)[None], image_hw=(3, 4))
    assert torch.allclose(out["pixels_xy"][0, 0], torch.tensor([2.0, 1.0]))
    assert out["valid"].tolist() == [[True, False, False]]


def test_project_landmarks_rejects_invalid_camera_geometry():
    world = torch.tensor([[[0.0, 0.0, 2.0]]])
    intrinsics = torch.eye(3)[None]
    pose = torch.eye(4)[None]
    bad_intrinsics = intrinsics.clone(); bad_intrinsics[:, 1, 1] = 0
    with pytest.raises(ValueError): project_landmarks(world, bad_intrinsics, pose, image_hw=(3, 4))
    bad_pose = pose.clone(); bad_pose[:, 3, 3] = 0
    with pytest.raises(ValueError): project_landmarks(world, intrinsics, bad_pose, image_hw=(3, 4))
