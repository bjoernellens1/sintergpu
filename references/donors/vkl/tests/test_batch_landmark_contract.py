from __future__ import annotations

import pytest
import torch

from vkl.kernels import (
    binary_local_match,
    brief_describe,
    landmark_multiview_validate,
    photometric_isp,
    project_landmarks,
    sparse_depth_validate,
    sparse_rgbd_observation,
)


@pytest.mark.parametrize("batch", [1, 8])
def test_landmark_and_photometric_primitives_preserve_b1_b8(batch: int):
    height, width, points = 12, 16, 3
    depth_image = torch.ones(batch, 1, height, width)
    pixels = torch.tensor([[[4, 4], [8, 5], [10, 7]]], dtype=torch.int64).expand(batch, -1, -1).clone()
    intrinsics = torch.tensor([[[8.0, 0, 8.0], [0, 8.0, 6.0], [0, 0, 1]]]).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()

    validated = sparse_depth_validate(depth_image, pixels)
    observation = sparse_rgbd_observation(pixels, validated["depth"], intrinsics, pose)
    projected = project_landmarks(
        observation["xyz_world"], intrinsics, pose, image_hw=(height, width), landmark_valid=observation["valid"]
    )
    pairs = torch.tensor([[[-1, 0], [1, 0]], [[0, -1], [0, 1]]], dtype=torch.int64)
    descriptor = brief_describe(
        torch.rand(batch, 3, height, width), pixels, pixels_valid=validated["valid"], pairs=pairs
    )
    match_valid = descriptor["valid"] & projected["valid"]
    matched = binary_local_match(
        descriptor["descriptor"],
        descriptor["descriptor"],
        projected["pixels_xy"],
        pixels,
        radius_px=1.0,
        max_hamming_distance=0,
        landmark_valid=match_valid,
        candidate_valid=descriptor["valid"],
    )
    observed_xy = projected["pixels_xy"][:, None].expand(-1, 2, -1, -1)
    observed_depth = projected["depth"][:, None].expand(-1, 2, -1)
    observed_valid = projected["valid"][:, None].expand(-1, 2, -1)
    multiview = landmark_multiview_validate(
        observation["xyz_world"],
        observed_xy,
        observed_depth,
        intrinsics[:, None].expand(-1, 2, -1, -1),
        pose[:, None].expand(-1, 2, -1, -1),
        landmark_valid=observation["valid"],
        observed_valid=observed_valid,
    )
    photo = photometric_isp(
        torch.rand(batch, 3, height, width),
        exposure_ev=torch.zeros(batch, 1, 1, 1),
        evidence=False,
    )

    assert all(value.shape[0] == batch for value in validated.values() if value.ndim)
    assert observation["xyz_world"].shape == (batch, points, 3)
    assert projected["pixels_xy"].shape == (batch, points, 2)
    assert descriptor["descriptor"].shape == (batch, points, 2)
    assert matched["matched_index"].shape == (batch, points)
    assert matched["valid"].all()
    assert multiview["valid"].shape == (batch, points)
    assert multiview["valid"].all()
    assert photo["rgb_observed"].shape == (batch, 3, height, width)
