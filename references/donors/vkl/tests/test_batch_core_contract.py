from __future__ import annotations

import pytest
import torch

from vkl.kernels import (
    canonical_rescale,
    depth_convert,
    depth_evidence,
    depth_geometry,
    depth_pyramid,
    fast_select,
    keyframe_filter,
    reproject_rgbd,
    residual_evidence,
    resolution_pyramid,
    rgbd_evidence,
    shi_tomasi_select,
    uniform_select,
)


@pytest.mark.parametrize("batch", [1, 8])
def test_core_image_depth_kernels_preserve_b1_b8(batch: int):
    rgb = torch.rand(batch, 3, 16, 20)
    depth = torch.ones(batch, 1, 16, 20)
    valid = torch.ones_like(depth, dtype=torch.bool)
    intrinsics = torch.tensor([[[10.0, 0, 10.0], [0, 10.0, 8.0], [0, 0, 1]]]).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()

    scaled = canonical_rescale(rgb=rgb, depth=depth, intrinsics=intrinsics, target_pixels=320, min_width=20, min_height=16)
    converted = depth_convert(values=depth, representation="DEPTH", valid=valid)
    evidence = depth_evidence(depth, valid)
    geometry = depth_geometry(depth, intrinsics, pose)
    pyramid = depth_pyramid(depth=depth, valid=valid)
    reprojection = reproject_rgbd(depth, intrinsics, pose, intrinsics, pose, depth)
    rgbd = rgbd_evidence(rgb=rgb, depth=depth)
    residual = residual_evidence(rgb, rgb, observed_depth=depth, predicted_depth=depth, valid=valid)
    selector = [uniform_select(rgb, num_points=8), shi_tomasi_select(rgb, num_points=8), fast_select(rgb, num_points=8)]
    frame = keyframe_filter(frame_score=torch.ones(batch), tile_score=torch.ones(batch, 2, 2))

    assert scaled["rgb"].shape[0] == batch and converted["depth"].shape[0] == batch
    assert evidence["valid_fraction"].shape[0] == batch and geometry["xyz_camera"].shape[0] == batch
    assert all(level["depth"].shape[0] == batch for level in pyramid.values())
    assert reprojection["visible"].shape[0] == batch and rgbd["frame_features"].shape[0] == batch
    assert residual["frame_residual"].shape[0] == batch and resolution_pyramid(rgb=rgb)["half"].shape[0] == batch
    assert all(item["pixels_xy"].shape[:2] == (batch, 8) for item in selector)
    assert frame["admit_mask"].shape == (batch,)
