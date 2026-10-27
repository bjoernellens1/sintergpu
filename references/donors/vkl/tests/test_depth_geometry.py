import pytest
import torch

from vkl.kernels.reference.geometry import depth_geometry, world_transform


def test_depth_geometry_central_xyz_and_invalid_depth():
    depth = torch.tensor([[[[0.0, 2.0, 0.0], [2.0, 2.0, 2.0], [0.0, 2.0, 0.0]]]])
    intr = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    out = depth_geometry(depth, intr)
    assert out["valid"].dtype == torch.bool and not out["valid"][0, 0, 0, 0]
    assert torch.equal(out["xyz_camera"][0, :, 0, 0], torch.zeros(3))
    assert torch.allclose(out["xyz_camera"][0, :, 1, 1], torch.tensor([0.0, 0.0, 2.0]))
    assert torch.isfinite(out["normals"]).all()
    assert out["rays_camera"].shape == out["xyz_camera"].shape


def test_depth_geometry_invalid_world_xyz_stays_zero_after_translation():
    depth = torch.tensor([[[[0.0, 1.0]]]])
    intr = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor([4.0, 5.0, 6.0])

    out = depth_geometry(depth, intr, pose)

    assert torch.equal(out["xyz_world"][0, :, 0, 0], torch.zeros(3))
    assert torch.allclose(out["xyz_world"][0, :, 0, 1], torch.tensor([5.0, 5.0, 7.0]))


def test_world_transform_applies_camera_to_world_translation():
    xyz = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 0, 3] = 4.0
    out = world_transform(xyz, pose)
    assert torch.allclose(out["xyz_world"], torch.tensor([[[[5.0]], [[2.0]], [[3.0]]]]))


def test_depth_geometry_can_return_world_xyz():
    depth = torch.ones(1, 1, 1, 1)
    intr = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 2, 3] = 2
    assert torch.allclose(
        depth_geometry(depth, intr, pose)["xyz_world"][0, :, 0, 0],
        torch.tensor([0.0, 0.0, 3.0]),
    )


def test_world_transform_rejects_pose_dtype_mismatch():
    xyz = torch.ones(1, 3, 2, 2)
    with pytest.raises(ValueError):
        world_transform(xyz, torch.eye(4, dtype=torch.float64).unsqueeze(0))


def test_depth_geometry_rejects_nonpositive_focal_length():
    depth = torch.ones(1, 1, 2, 2)
    intr = torch.eye(3).unsqueeze(0)
    intr[:, 0, 0] = 0
    with pytest.raises(ValueError):
        depth_geometry(depth, intr)


def test_normals_are_zero_at_image_boundary():
    depth = torch.ones(1, 1, 3, 3)
    intr = torch.eye(3).unsqueeze(0)
    out = depth_geometry(depth, intr)
    assert torch.equal(out["normals"][0, :, -1, -1], torch.zeros(3))


def test_depth_geometry_invalid_neighbor_does_not_make_normal():
    depth = torch.ones(1, 1, 3, 3)
    depth[:, :, 1, 1] = 0
    intr = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    out = depth_geometry(depth, intr)
    assert not out["valid"][0, 0, 1, 1]
    assert torch.equal(out["normals"][0, :, 1, 0], torch.zeros(3))


def test_depth_geometry_rejects_depth_step_stencil():
    depth = torch.ones(1, 1, 3, 3)
    depth[:, :, :, 2] = 2
    intr = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    out = depth_geometry(depth, intr)
    assert out["geometry_confidence"][0, 0, 1, 1] == 0


def test_flat_normal_faces_camera():
    depth = torch.ones(1, 1, 5, 5)
    intr = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
    out = depth_geometry(depth, intr)
    assert out["normals"][0, 2, 2, 2] < -0.9
    assert out["geometry_confidence"][0, 0, 2, 2] == 1
