import pytest
import torch

from vkl.kernels import deterministic_reduce_solve, sparse_pose_reduce


def _inputs(batch: int, *, translation_z: float = 0.0):
    xyz_world = torch.tensor(
        [[[-0.2, -0.1, 2.0], [0.3, -0.2, 2.2], [-0.1, 0.4, 1.8], [0.25, 0.3, 2.4]]]
    ).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    pose[:, 2, 3] = translation_z
    return xyz_world, xyz_world.clone(), pose


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_pose_reduce_emits_fixed_partials_for_matched_landmarks(batch: int):
    xyz_world, observed_xyz_camera, pose = _inputs(batch)

    out = sparse_pose_reduce(
        xyz_world,
        observed_xyz_camera,
        pose,
        partial_count=2,
    )

    assert out["H"].shape == (batch, 6, 6)
    assert out["b"].shape == (batch, 6)
    assert out["partial_H"].shape == (batch, 2, 6, 6)
    assert out["partial_b"].shape == (batch, 2, 6)
    assert out["partial_valid"].shape == (batch, 2)
    assert torch.allclose(out["H"], out["partial_H"].sum(1))
    assert torch.allclose(out["b"], out["partial_b"].sum(1))
    assert torch.allclose(out["b"], torch.zeros_like(out["b"]))
    assert (out["inlier_count"] == 4).all()
    assert out["partial_valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_pose_reduce_reports_pose_error_without_synthetic_support(batch: int):
    xyz_world, observed_xyz_camera, pose = _inputs(batch, translation_z=0.2)
    matched = torch.tensor([[True, False, True, False]]).expand(batch, -1)

    out = sparse_pose_reduce(
        xyz_world,
        observed_xyz_camera,
        pose,
        matched=matched,
        partial_count=2,
    )

    assert (out["inlier_count"] == 2).all()
    assert torch.allclose(out["residual_l2_mean"], torch.full((batch,), 0.2), atol=1e-6)
    assert (out["b"].abs().sum(-1) > 0).all()
    assert out["partial_valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_pose_reduce_zeroes_unmatched_capacity(batch: int):
    xyz_world, observed_xyz_camera, pose = _inputs(batch)

    out = sparse_pose_reduce(
        xyz_world,
        observed_xyz_camera,
        pose,
        matched=torch.zeros((batch, 4), dtype=torch.bool),
        partial_count=2,
    )

    assert torch.equal(out["H"], torch.zeros_like(out["H"]))
    assert torch.equal(out["b"], torch.zeros_like(out["b"]))
    assert torch.equal(out["inlier_count"], torch.zeros(batch, dtype=torch.int64))
    assert not out["partial_valid"].any()


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_pose_reduce_partials_feed_deterministic_tracking_solve(batch: int):
    xyz_world, observed_xyz_camera, pose = _inputs(batch, translation_z=0.2)
    reduced = sparse_pose_reduce(xyz_world, observed_xyz_camera, pose, partial_count=2)

    solved = deterministic_reduce_solve(reduced["partial_H"], reduced["partial_b"], reduced["partial_valid"])

    assert torch.allclose(solved["H"], reduced["H"])
    assert torch.allclose(solved["b"], reduced["b"])
    assert solved["valid"].all()
