from __future__ import annotations

import pytest
import torch

from vkl.kernels import project_depth_evidence


def _camera_case():
    xyz = torch.tensor([[[0.0, 0.0, 2.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    w2c = torch.eye(4)[None]
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
    depth = torch.full((1, 1, 5, 5), 2.0)
    return xyz, mask, w2c, intrinsics, depth


@pytest.mark.parametrize("batch", [1, 8])
def test_project_depth_evidence_classifies_support_conflict_and_invalid_projection(batch: int):
    xyz = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [20.0, 0.0, 2.0]]]).expand(batch, -1, -1)
    visible = torch.tensor([[True, True, True]]).expand(batch, -1)
    provisional = torch.tensor([[True, False, True]]).expand(batch, -1)
    candidate = torch.tensor([[False, True, False]]).expand(batch, -1)
    w2c = torch.eye(4)[None].expand(batch, -1, -1)
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1)
    sensor_depth = torch.full((batch, 1, 5, 5), 2.0)
    out = project_depth_evidence(
        xyz, visible, provisional, candidate, w2c, intrinsics, sensor_depth,
        support_threshold=0.1, conflict_threshold=0.1,
    )
    assert out["projected_valid"].shape == (batch, 3)
    assert out["support"].tolist() == [[True, False, False]] * batch
    assert out["conflict"].tolist() == [[False, True, False]] * batch
    assert not out["projected_valid"][:, 2].any()
    assert out["depth_residual"][:, 0].eq(0).all()


def test_project_depth_evidence_masks_nonfinite_points_without_border_aliasing():
    xyz = torch.tensor([[[0.0, 0.0, 2.0], [float("nan"), 0.0, 2.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = project_depth_evidence(
        xyz, mask, mask, torch.zeros_like(mask), torch.eye(4)[None],
        torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]),
        torch.full((1, 1, 5, 5), 2.0), support_threshold=0.1, conflict_threshold=0.1,
    )
    assert out["eligible"].tolist() == [[True, False]]
    assert out["support"].tolist() == [[True, False]]
    assert out["depth_residual"].tolist() == [[0.0, 0.0]]


def test_project_depth_evidence_rejects_invalid_camera_models_by_default():
    xyz, mask, w2c, intrinsics, depth = _camera_case()
    bad_intrinsics = intrinsics.clone()
    bad_intrinsics[:, 0, 0] = 0
    with pytest.raises(ValueError):
        project_depth_evidence(
            xyz, mask, mask, mask, w2c, bad_intrinsics, depth,
            support_threshold=0.1, conflict_threshold=0.1,
        )
    bad_pose = w2c.clone()
    bad_pose[:, 0, 0] = 2
    with pytest.raises(ValueError):
        project_depth_evidence(
            xyz, mask, mask, mask, bad_pose, intrinsics, depth,
            support_threshold=0.1, conflict_threshold=0.1,
        )


def test_project_depth_evidence_prevalidated_path_skips_value_checks():
    xyz, mask, w2c, intrinsics, depth = _camera_case()
    out = project_depth_evidence(
        xyz, mask, mask, mask, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )
    assert out["support"].item()


def test_project_depth_evidence_rejects_nonfinite_thresholds():
    xyz, mask, w2c, intrinsics, depth = _camera_case()
    with pytest.raises(ValueError):
        project_depth_evidence(
            xyz, mask, mask, torch.zeros_like(mask), w2c, intrinsics, depth,
            support_threshold=float("nan"), conflict_threshold=0.1,
        )
