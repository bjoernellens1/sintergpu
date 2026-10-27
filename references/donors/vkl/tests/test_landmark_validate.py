from __future__ import annotations

import pytest
import torch

from vkl.kernels import landmark_multiview_validate


def _inputs():
    xyz = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])
    intrinsics = torch.tensor([[[[2.0, 0, 2.0], [0, 2.0, 1.0], [0, 0, 1]]]]).repeat(1, 2, 1, 1)
    poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    uv = torch.tensor([[[[2.0, 1.0], [3.4, 1.0]], [[2.0, 1.0], [2.8, 1.0]]]])
    depth = torch.tensor([[[2.0, 2.0], [2.0, 2.8]]])
    return xyz, uv, depth, intrinsics, poses


def test_landmark_multiview_validate_requires_consistent_reprojection_and_depth():
    xyz, uv, depth, intrinsics, poses = _inputs()

    out = landmark_multiview_validate(
        xyz, uv, depth, intrinsics, poses,
        min_support=2,
        max_reprojection_px=0.25,
        max_relative_depth_error=0.1,
    )

    assert out["support_count"].tolist() == [[2, 0]]
    assert out["valid"].tolist() == [[True, False]]
    assert torch.allclose(out["reprojection_rmse"][0, 0], torch.tensor(0.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_support": True},
        {"min_support": 0},
        {"min_support": 3},
        {"max_reprojection_px": float("nan")},
        {"max_reprojection_px": float("inf")},
        {"max_relative_depth_error": float("nan")},
        {"max_relative_depth_error": float("inf")},
    ],
)
def test_landmark_multiview_validate_rejects_invalid_thresholds(kwargs):
    xyz, uv, depth, intrinsics, poses = _inputs()
    with pytest.raises(ValueError):
        landmark_multiview_validate(xyz, uv, depth, intrinsics, poses, **kwargs)
