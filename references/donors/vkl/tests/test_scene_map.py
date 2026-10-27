from __future__ import annotations

import struct

import pytest
import torch

from vkl.bench.scene_map import (
    backproject_world_points,
    densify_pseudodepth_evidence,
    fuse_pseudodepth_evidence,
    rotate_display_x,
    voxel_downsample,
    voxel_downsample_multiview,
    world_voxel_overlap,
    write_colored_ply,
    write_triangle_ply,
)


def test_backproject_world_points_applies_pose_and_keeps_rgb() -> None:
    depth = torch.tensor([[[[1.0, 0.0], [2.0, 1.0]]]])
    rgb = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]])
    intrinsics = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 0, 3] = 3.0

    xyz, colors = backproject_world_points(depth, rgb, intrinsics, pose, stride=1)

    torch.testing.assert_close(xyz, torch.tensor([[3.0, 0.0, 1.0], [3.0, 2.0, 2.0], [4.0, 1.0, 1.0]]))
    torch.testing.assert_close(colors, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))


def test_voxel_downsample_averages_each_occupied_cell() -> None:
    xyz = torch.tensor([[0.01, 0.01, 0.01], [0.09, 0.01, 0.01], [0.21, 0.01, 0.01]])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    reduced_xyz, reduced_colors = voxel_downsample(xyz, colors, voxel_size=0.1)

    torch.testing.assert_close(reduced_xyz, torch.tensor([[0.05, 0.01, 0.01], [0.21, 0.01, 0.01]]))
    torch.testing.assert_close(reduced_colors, torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]))


def test_world_voxel_overlap_reports_precision_recall_and_f1() -> None:
    estimate = torch.tensor([[0.05, 0.05, 0.05], [1.05, 0.05, 0.05]])
    reference = torch.tensor([[0.06, 0.05, 0.05], [2.05, 0.05, 0.05]])

    overlap = world_voxel_overlap(estimate, reference, voxel_size=1.0)

    assert overlap == {
        "estimate_cells": 2,
        "reference_cells": 2,
        "intersection_cells": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_rotate_display_x_uses_requested_negative_ninety_degree_world_up_view() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0]])

    rotated = rotate_display_x(points, degrees=-90.0)

    torch.testing.assert_close(rotated, torch.tensor([[1.0, 3.0, -2.0]]), atol=1e-6, rtol=0)


def test_write_colored_ply_keeps_raw_world_coordinates_and_rgb(tmp_path) -> None:
    path = tmp_path / "map.ply"

    write_colored_ply(path, torch.tensor([[1.5, -2.0, 3.25]]), torch.tensor([[1.0, 0.5, 0.0]]))

    header, record = path.read_bytes().split(b"end_header\n", maxsplit=1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 1" in header
    assert struct.unpack("<fffBBB", record) == (1.5, -2.0, 3.25, 255, 128, 0)


def test_write_triangle_ply_keeps_vertices_and_faces(tmp_path) -> None:
    path = tmp_path / "boundary.ply"
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    write_triangle_ply(path, vertices, faces)

    payload = path.read_bytes()
    header, record = payload.split(b"end_header\n", maxsplit=1)
    assert b"element vertex 3" in header
    assert b"element face 1" in header
    assert struct.unpack("<fff", record[:12]) == (0.0, 0.0, 0.0)
    assert record[36:] == bytes((3, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0))


@pytest.mark.parametrize("batch", [1, 8])
def test_fuse_pseudodepth_requires_temporal_agreement_without_sensor_depth(batch: int) -> None:
    dense = torch.full((batch, 1, 2, 2), 2.0)
    dso = torch.zeros_like(dense)
    dso[:, :, 0, 0] = 2.02
    dense_valid = torch.ones_like(dense, dtype=torch.bool)
    dso_valid = torch.zeros_like(dense_valid)
    dso_valid[:, :, 0, 0] = True
    confidence = torch.full_like(dense, 0.5)
    temporal = torch.full_like(dense, 2.01)

    out = fuse_pseudodepth_evidence(
        d10_depth=dense,
        d10_valid=dense_valid,
        d10_confidence=confidence,
        d6_depth=dso,
        d6_valid=dso_valid,
        d6_confidence=confidence,
        temporal_depths=[temporal],
        temporal_valids=[dense_valid],
        temporal_confidences=[confidence],
    )

    assert out["d10_d6"]["valid"].all()
    assert out["temporal"]["valid"].all()
    expected = torch.full_like(dense, 2.005)
    expected[:, :, 0, 0] = 2.01
    torch.testing.assert_close(out["temporal"]["depth"], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("batch", [1, 8])
def test_fuse_pseudodepth_can_require_two_temporal_observations(batch: int) -> None:
    depth = torch.full((batch, 1, 2, 2), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    confidence = torch.full_like(depth, 0.5)

    out = fuse_pseudodepth_evidence(
        d10_depth=depth,
        d10_valid=valid,
        d10_confidence=confidence,
        d6_depth=torch.zeros_like(depth),
        d6_valid=torch.zeros_like(valid),
        d6_confidence=torch.zeros_like(depth),
        temporal_depths=[torch.full_like(depth, 2.01), torch.full_like(depth, 1.99)],
        temporal_valids=[valid, valid],
        temporal_confidences=[confidence, confidence],
        consistency_threshold=0.03,
        min_support=3,
        min_temporal_support=2,
    )

    assert out["temporal"]["valid"].all()
    torch.testing.assert_close(out["temporal"]["depth"], torch.full_like(depth, 2.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("batch", [1, 8])
def test_densify_pseudodepth_preserves_direct_provenance(batch: int) -> None:
    depth = torch.full((batch, 1, 5, 5), 2.0)
    direct_valid = torch.ones_like(depth, dtype=torch.bool)
    direct_valid[:, :, 2, 2] = False
    depth[:, :, 2, 2] = 0

    out = densify_pseudodepth_evidence(
        depth=depth,
        valid=direct_valid,
        confidence=torch.ones_like(depth),
        rgb=torch.zeros(batch, 3, 5, 5),
    )

    assert out["valid"][:, :, 2, 2].all()
    assert not out["direct_valid"][:, :, 2, 2].any()
    assert out["propagated_valid"][:, :, 2, 2].all()
    assert (out["provenance"][:, :, 2, 2] == 2).all()


def test_voxel_downsample_multiview_requires_distinct_frame_support() -> None:
    xyz = torch.tensor([[0.05, 0.05, 0.05], [0.06, 0.05, 0.05], [1.05, 0.05, 0.05]])
    rgb = torch.ones_like(xyz)
    source_ids = torch.tensor([7, 7, 8], dtype=torch.int64)

    reduced_xyz, _, source_support = voxel_downsample_multiview(
        xyz,
        rgb,
        source_ids,
        voxel_size=1.0,
        min_source_support=2,
    )

    assert reduced_xyz.shape == (0, 3)
    assert source_support.shape == (0,)
