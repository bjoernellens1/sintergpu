import pytest
import torch

from vkl.data.consistent import wall_floor_boxes, wall_floor_boxes_trajectory
from vkl.kernels.reference.reproject import rasterize_reprojection, reproject_rgbd


def test_reproject_identity_is_visible_and_consistent():
    depth = torch.ones(1, 1, 3, 3)
    intr = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4).unsqueeze(0)
    out = reproject_rgbd(depth, intr, pose, intr, pose, depth)
    assert out["visible"].all()
    assert out["depth_consistency_valid"].all()
    assert torch.allclose(out["depth_consistency"], torch.zeros_like(out["depth_consistency"]))
    assert torch.allclose(out["displacement"], torch.zeros_like(out["displacement"]))


def test_reproject_invalid_source_never_becomes_valid_consistency_evidence():
    depth = torch.tensor([[[[0.0, 1.0]]]])
    target_depth = torch.ones_like(depth)
    intr = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)

    out = reproject_rgbd(depth, intr, pose, intr, pose, target_depth)

    assert not out["visible"][0, 0, 0, 0]
    assert not out["depth_consistency_valid"][0, 0, 0, 0]
    assert not out["occlusion_valid"][0, 0, 0, 0]
    assert not out["occluded"][0, 0, 0, 0]
    assert out["depth_consistency"][0, 0, 0, 0] == 0
    assert out["depth_consistency_valid"][0, 0, 0, 1]


def test_rasterize_reprojection_identity_preserves_depth_and_confidence():
    depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    confidence = torch.full_like(depth, 0.7)
    intr = torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4).unsqueeze(0)

    out = rasterize_reprojection(depth, confidence, intr, pose, intr, pose, target_hw=(2, 2))

    assert out["valid"].all()
    assert torch.allclose(out["depth"], depth)
    assert torch.allclose(out["confidence"], confidence)


def test_reproject_translation_reports_parallax():
    depth = torch.ones(1, 1, 5, 5)
    intr = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
    source = torch.eye(4).unsqueeze(0)
    target = source.clone()
    target[:, 0, 3] = 0.25
    out = reproject_rgbd(depth, intr, source, intr, target, target_hw=(5, 5))
    assert out["overlap"].item() < 1.0
    assert out["parallax"].item() > 0.0
    assert "consistency" not in out


def test_reproject_reports_batchwise_metrics_and_target_shape():
    depth = torch.ones(2, 1, 3, 3)
    intr = torch.tensor([[[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]]).repeat(2, 1, 1)
    source = torch.eye(4).repeat(2, 1, 1)
    target = source.clone()
    target[1, 0, 3] = .5
    out = reproject_rgbd(depth, intr, source, intr, target, torch.ones(2, 1, 4, 5))
    assert out["visible"].shape == (2, 1, 3, 3)
    assert out["overlap"].shape == (2,) and out["parallax"].shape == (2,)
    assert out["parallax"][1] > out["parallax"][0]


def test_reproject_rejects_target_shape_disagreement():
    depth = torch.ones(1, 1, 2, 2)
    intr = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError):
        reproject_rgbd(depth, intr, pose, intr, pose, depth, target_hw=(3, 3))


def test_consistent_wall_floor_boxes_trajectory_has_overlap_and_parallax():
    first, second = wall_floor_boxes_trajectory()
    out = reproject_rgbd(first["depth"], first["intrinsics"], first["pose_c2w"], second["intrinsics"], second["pose_c2w"], second["depth"])
    assert out["overlap"][0] > 0.5
    assert out["parallax"][0] > 0


def test_wall_floor_boxes_raycast_contains_each_surface_and_changes_with_view():
    first = wall_floor_boxes(camera_x=0.0)
    second = wall_floor_boxes(camera_x=0.08)
    expected_colors = torch.tensor(
        [[0.45, 0.45, 0.5], [0.35, 0.25, 0.15], [0.8, 0.2, 0.1], [0.1, 0.7, 0.2]]
    )

    pixels = first["rgb"][0].permute(1, 2, 0)
    for color in expected_colors:
        assert torch.any(torch.all(torch.isclose(pixels, color), dim=-1))
    assert not torch.equal(first["rgb"], second["rgb"])
    assert not torch.equal(first["depth"], second["depth"])


def test_reproject_rejects_mismatched_batch():
    d = torch.ones(2, 1, 2, 2)
    k = torch.eye(3).repeat(2, 1, 1)
    p = torch.eye(4).repeat(2, 1, 1)
    with pytest.raises(ValueError):
        reproject_rgbd(d, k[:1], p, k, p, target_hw=(2, 2))
    with pytest.raises(ValueError):
        reproject_rgbd(d, k, p, k[:1], p, target_hw=(2, 2))


def test_reproject_rejects_geometry_dtype_mismatch():
    d = torch.ones(2, 1, 2, 2)
    k = torch.eye(3).repeat(2, 1, 1)
    p = torch.eye(4).repeat(2, 1, 1)
    with pytest.raises(ValueError):
        reproject_rgbd(d, k, p, k.double(), p, target_hw=(2, 2))


def test_occlusion_is_directional():
    d = torch.ones(1, 1, 2, 2)
    k = torch.eye(3).unsqueeze(0)
    p = torch.eye(4).unsqueeze(0)
    near = reproject_rgbd(d, k, p, k, p, torch.full_like(d, 2.), target_hw=(2, 2))
    far = reproject_rgbd(torch.full_like(d, 2.), k, p, k, p, d, target_hw=(2, 2))
    assert not near["occluded"].any()
    assert far["occluded"].all()


def test_odd_resolution_center_row_contains_box_without_gap():
    # Re-render at odd dimensions; the center row must contain a contiguous
    # finite wall/box raycast, rather than a slab-division artifact.
    frame = wall_floor_boxes(height=33, width=49)
    row = frame["rgb"][0, :, 16]
    assert torch.isfinite(frame["depth"]).all()
    assert (row[0] > 0.7).any()
