from __future__ import annotations

import torch

from vkl.kernels import align_depth, fuse_depth, landmark_multiview_validate, temporal_depth_fuse


def test_fuse_depth_ignores_invalid_nan_source():
    good = torch.tensor([[[[2.0]]]])
    bad = torch.tensor([[[[float("nan")]]]])
    out = fuse_depth(
        [good, bad],
        [torch.ones_like(good, dtype=torch.bool), torch.zeros_like(bad, dtype=torch.bool)],
        [torch.ones_like(good), torch.ones_like(bad)],
    )

    assert out["valid"].item()
    assert torch.isfinite(out["depth"]).all()
    assert out["depth"].item() == 2.0
    assert out["winning_source"].item() == 0


def test_temporal_depth_fuse_ignores_invalid_nan_source_in_depth_and_variance():
    target = torch.tensor([[[[2.0]]]])
    temporal = torch.tensor([[[[float("nan")]]]])
    out = temporal_depth_fuse(
        target,
        [temporal],
        torch.ones_like(target, dtype=torch.bool),
        [torch.zeros_like(temporal, dtype=torch.bool)],
        torch.ones_like(target),
        [torch.ones_like(temporal)],
        target_trusted=torch.ones_like(target, dtype=torch.bool),
    )

    assert out["valid"].item()
    assert out["depth"].item() == 2.0
    assert torch.isfinite(out["depth_variance"]).all()
    assert out["depth_variance"].item() == 0.0


def test_align_depth_zero_weight_outlier_does_not_change_robust_diagnostics():
    predicted = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    anchors = torch.tensor([[[[2.0, 4.0, 1000.0]]]])
    mask = torch.ones_like(predicted, dtype=torch.bool)
    confidence = torch.tensor([[[[1.0, 1.0, 0.0]]]])

    weighted = align_depth(predicted, anchors, mask, confidence, mode="scale_only")
    trimmed = align_depth(
        predicted,
        torch.tensor([[[[2.0, 4.0, 6.0]]]]),
        torch.tensor([[[[True, True, False]]]]),
        mode="scale_only",
    )

    assert torch.allclose(weighted["scale"], trimmed["scale"])
    assert torch.allclose(weighted["fit_error"], trimmed["fit_error"])
    assert torch.allclose(weighted["inlier_fraction"], trimmed["inlier_fraction"])


def test_landmark_multiview_validation_ignores_invalid_nan_view():
    xyz = torch.tensor([[[0.0, 0.0, 2.0]]])
    intrinsics = torch.tensor([[[[2.0, 0, 2.0], [0, 2.0, 1.0], [0, 0, 1]]]]).repeat(1, 3, 1, 1)
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    uv = torch.tensor([[[[2.0, 1.0]], [[2.0, 1.0]], [[float("nan"), float("nan")]]]])
    depth = torch.tensor([[[2.0], [2.0], [float("nan")]]])
    observed_valid = torch.tensor([[[True], [True], [False]]])

    out = landmark_multiview_validate(
        xyz,
        uv,
        depth,
        intrinsics,
        poses,
        observed_valid=observed_valid,
        min_support=2,
    )

    assert out["valid"].item()
    assert out["support_count"].item() == 2
    assert torch.isfinite(out["reprojection_rmse"]).all()
    assert torch.isfinite(out["relative_depth_mae"]).all()
