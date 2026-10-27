from __future__ import annotations

import pytest
import torch

from vkl.kernels import direct_pose_depth_dso, direct_pose_depth_tensorized, dso_select_points


def _inputs(dtype: torch.dtype = torch.float32):
    height, width = 24, 32
    rgb = torch.rand(1, 3, height, width, dtype=dtype)
    neighbors = rgb[:, None].clone()
    intrinsics = torch.tensor(
        [[[24.0, 0.0, 15.5], [0.0, 24.0, 11.5], [0.0, 0.0, 1.0]]],
        dtype=dtype,
    )
    pose = torch.eye(4, dtype=dtype)[None]
    neighbor_pose = torch.eye(4, dtype=dtype)[None, None]
    neighbor_pose[:, :, 0, 3] = 0.1
    return rgb, neighbors, pose, neighbor_pose, intrinsics


def test_dso_selector_does_not_fabricate_flat_image_points():
    out = dso_select_points(torch.zeros(1, 3, 16, 24), num_points=12, patch_radius=1)
    assert not out["valid"].any()
    assert not out["score"].any()
    assert not out["gradient"].any()


def test_dso_selector_does_not_treat_validity_border_as_gradient():
    rgb = torch.ones(1, 3, 24, 32)
    valid = torch.ones(1, 1, 24, 32, dtype=torch.bool)
    valid[..., :12, :] = False
    out = dso_select_points(rgb, rgb_valid=valid, num_points=16, patch_radius=1)
    assert not out["valid"].any()


def test_d6_preserves_float64_and_fixed_capacity_validity():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs(torch.float64)
    pixels = torch.tensor([[[12, 12], [-100, -100], [12, 12]]], dtype=torch.int64)
    pixels_valid = torch.tensor([[True, False, False]])
    out = direct_pose_depth_dso(
        rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
        pose_neighbors=neighbor_pose, intrinsics=intrinsics, pixels_xy=pixels,
        pixels_valid=pixels_valid, num_hypotheses=8, min_depth=1.0, max_depth=4.0,
        min_parallax_px=0.0, min_confidence=0.0,
    )
    assert out["depth"].dtype == torch.float64
    assert out["gradient"].dtype == torch.float64
    assert out["selected"].sum().item() == 1
    assert out["selected"][0, 0, 12, 12]


def test_d6_deduplicates_external_slots_before_dense_outputs():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs()
    pixels = torch.tensor([[[12, 12], [12, 12], [14, 12]]])
    out = direct_pose_depth_dso(
        rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
        pose_neighbors=neighbor_pose, intrinsics=intrinsics, pixels_xy=pixels,
        pixels_valid=torch.ones(1, 3, dtype=torch.bool), num_hypotheses=8,
        min_depth=1.0, max_depth=4.0, min_parallax_px=0.0, min_confidence=0.0,
    )
    assert out["selected"].sum().item() == 2
    both = out["valid"] & (out["depth"] > 0) & (out["inverse_depth"] > 0)
    assert torch.allclose(
        (out["depth"] * out["inverse_depth"])[both],
        torch.ones_like((out["depth"] * out["inverse_depth"])[both]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_d6_neighbor_validity_can_suppress_all_anchor_support():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs()
    pixels = torch.tensor([[[12, 12], [14, 12]]])
    out = direct_pose_depth_dso(
        rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
        pose_neighbors=neighbor_pose, intrinsics=intrinsics, pixels_xy=pixels,
        pixels_valid=torch.ones(1, 2, dtype=torch.bool),
        rgb_neighbors_valid=torch.zeros(1, 1, 1, 24, 32, dtype=torch.bool),
        num_hypotheses=8, min_depth=1.0, max_depth=4.0,
        min_parallax_px=0.0, min_confidence=0.0,
    )
    assert out["selected"].sum().item() == 2
    assert not out["valid"].any()


def test_streamed_and_tensorized_d6_share_hypothesis_contract():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs()
    for fn in (direct_pose_depth_dso, direct_pose_depth_tensorized):
        with pytest.raises(ValueError, match="parameters are invalid"):
            fn(
                rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
                pose_neighbors=neighbor_pose, intrinsics=intrinsics, num_points=8,
                num_hypotheses=1, min_depth=1.0, max_depth=4.0,
            )


def test_d6_requires_pixels_for_pixels_valid():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs()
    with pytest.raises(ValueError, match="pixels_valid requires pixels_xy"):
        direct_pose_depth_dso(
            rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
            pose_neighbors=neighbor_pose, intrinsics=intrinsics,
            pixels_valid=torch.ones(1, 4, dtype=torch.bool), num_points=4,
            num_hypotheses=8, min_depth=1.0, max_depth=4.0,
        )


def test_d6_rejects_invalid_intrinsics_and_nonrigid_pose():
    rgb, neighbors, pose, neighbor_pose, intrinsics = _inputs()
    bad_intrinsics = intrinsics.clone()
    bad_intrinsics[:, 0, 0] = 0
    with pytest.raises(ValueError):
        direct_pose_depth_dso(
            rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=pose,
            pose_neighbors=neighbor_pose, intrinsics=bad_intrinsics, num_points=8,
            num_hypotheses=8, min_depth=1.0, max_depth=4.0,
        )
    bad_pose = pose.clone()
    bad_pose[:, 0, 0] = 2
    with pytest.raises(ValueError):
        direct_pose_depth_tensorized(
            rgb_reference=rgb, rgb_neighbors=neighbors, pose_reference=bad_pose,
            pose_neighbors=neighbor_pose, intrinsics=intrinsics, num_points=8,
            num_hypotheses=8, min_depth=1.0, max_depth=4.0,
        )
