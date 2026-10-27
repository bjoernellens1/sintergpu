from __future__ import annotations

import pytest
import torch

from vkl.kernels import (
    aggregate_depth_cost_volume,
    align_depth,
    brief_describe,
    dense_pose_depth,
    depth_reproject,
    direct_pose_depth_dso,
    direct_pose_depth_tensorized,
    dso_select_points,
    fuse_depth,
    keypoint_evidence,
    photometric_isp,
    pose_observable_select,
    project_depth_evidence,
    rasterize_reprojection,
    regularize_dense_depth,
    select_best_inverse_depth,
    select_compact,
    spatial_bin_reduce,
    spatial_select_points,
    temporal_best_observation,
    temporal_depth_fuse,
    temporal_frame_compare,
    temporal_reproject_rgb,
    temporal_restore_rgb,
    temporal_restore_rgbd,
    triangulate_depth,
    validate_frame_batch,
    world_transform,
)


def _intrinsics(batch: int) -> torch.Tensor:
    return torch.tensor([[[12.0, 0, 7.5], [0, 12.0, 5.5], [0, 0, 1]]]).expand(batch, -1, -1).clone()


@pytest.mark.parametrize("batch", [1, 8])
def test_remaining_public_kernels_preserve_b1_b8(batch: int):
    height, width, count, neighbors = 12, 16, 6, 1
    rgb = torch.rand(batch, 3, height, width)
    depth = torch.ones(batch, 1, height, width)
    valid = torch.ones_like(depth, dtype=torch.bool)
    confidence = torch.ones_like(depth)
    intrinsics = _intrinsics(batch)
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    poses = pose[:, None].expand(-1, neighbors, -1, -1).clone()
    poses[:, :, 0, 3] = 0.05
    context = rgb[:, None].clone()

    assert validate_frame_batch({"rgb": rgb, "depth": depth, "intrinsics": intrinsics, "pose_c2w": pose})["rgb"].shape[0] == batch
    assert select_compact(torch.rand(batch, count, 3), torch.ones(batch, count, dtype=torch.bool), capacity=4)["values"].shape[0] == batch
    assert spatial_bin_reduce(torch.rand(batch, count, 3), torch.ones(batch, count, dtype=torch.bool), voxel_size=1.0, origin=torch.zeros(3), capacity=4)["cell_ids"].shape[0] == batch
    assert align_depth(depth, depth * 2, valid)["depth"].shape[0] == batch
    assert fuse_depth([depth, depth], [valid, valid], [confidence, confidence])["depth"].shape[0] == batch
    assert aggregate_depth_cost_volume(torch.rand(batch, 3, height, width), rgb).shape[0] == batch
    assert regularize_dense_depth(depth, valid, confidence, rgb)["depth"].shape[0] == batch
    assert photometric_isp(rgb, exposure_ev=torch.zeros(batch, 1, 1, 1), evidence=False)["rgb_observed"].shape[0] == batch
    assert depth_reproject(depth, intrinsics, pose, intrinsics, pose, depth)["visible"].shape[0] == batch

    selected = dso_select_points(rgb, num_points=count, patch_radius=1)
    assert pose_observable_select(
        rgb, intrinsics, pose, poses[:, 0], num_points=count, min_depth=0.5, max_depth=2.0
    )["pixels_xy"].shape == (batch, count, 2)
    assert spatial_select_points(
        selected["pixels_xy"],
        selected["gradient"],
        selected["valid"],
        num_points=count,
        height=height,
        width=width,
        tiles=(2, 2),
    )["pixels_xy"].shape == (batch, count, 2)
    assert brief_describe(rgb, selected["pixels_xy"], num_bits=8)["descriptor"].shape[:2] == (batch, count)
    direct_kwargs = {
        "rgb_reference": rgb,
        "rgb_neighbors": context,
        "pose_reference": pose,
        "pose_neighbors": poses,
        "intrinsics": intrinsics,
        "pixels_xy": selected["pixels_xy"],
        "num_hypotheses": 2,
        "min_depth": 0.5,
        "max_depth": 2.0,
        "min_confidence": 0.0,
    }
    assert direct_pose_depth_dso(**direct_kwargs)["depth"].shape[0] == batch
    assert direct_pose_depth_tensorized(**direct_kwargs)["depth"].shape[0] == batch
    assert dense_pose_depth(rgb, context, pose, poses, intrinsics, num_hypotheses=2, min_depth=0.5, max_depth=2.0, min_confidence=0.0)["depth"].shape[0] == batch

    temporal = temporal_depth_fuse(depth, [depth], valid, [valid], confidence, [confidence])
    assert temporal["depth"].shape[0] == batch
    projected = temporal_reproject_rgb(rgb, depth, context, intrinsics, pose, intrinsics[:, None], poses)
    assert projected["aligned_rgb"].shape[0] == batch
    compared = temporal_frame_compare(rgb, context)
    assert compared["pairwise_quality"].shape[0] == batch
    assert temporal_best_observation(compared["pairwise_equivalence"], compared["target_quality"], compared["pairwise_quality"])["best_index"].shape == (batch,)
    assert temporal_restore_rgb(rgb, context)["restored_rgb"].shape[0] == batch
    assert temporal_restore_rgbd(rgb, context)["rgb"].shape[0] == batch

    geometry = world_transform(torch.zeros(batch, 3, height, width), pose)
    assert geometry["xyz_world"].shape[0] == batch
    assert rasterize_reprojection(depth, confidence, intrinsics, pose, intrinsics, pose, target_hw=(height, width))["depth"].shape[0] == batch
    pixels_a = selected["pixels_xy"].float()
    pixels_b = pixels_a + torch.tensor([0.2, 0.0])
    assert triangulate_depth(pixels_a, pixels_b, intrinsics, intrinsics, pose, poses[:, 0])["depth"].shape == (batch, count)
    assert select_best_inverse_depth(torch.rand(batch, count, 3), torch.tensor([0.5, 1.0, 2.0]))["inverse_depth"].shape == (batch, count)

    sparse = torch.zeros(batch, 1, height, width, dtype=torch.bool)
    sparse[..., 2, 2] = True
    keypoint = keypoint_evidence(
        selected=sparse,
        valid=sparse,
        confidence=sparse.float(),
        parallax_px=sparse.float(),
        photometric_error=torch.zeros_like(depth),
        gradient=sparse.float(),
    )
    assert keypoint["frame_features"].shape[0] == batch
    assert project_depth_evidence(
        torch.zeros(batch, count, 3),
        torch.ones(batch, count, dtype=torch.bool),
        torch.ones(batch, count, dtype=torch.bool),
        torch.zeros(batch, count, dtype=torch.bool),
        pose,
        intrinsics,
        depth,
        support_threshold=0.1,
        conflict_threshold=0.1,
    )["support"].shape == (batch, count)
