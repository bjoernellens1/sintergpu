from __future__ import annotations

import torch

from vkl.kernels import dense_pose_depth


def test_dense_pose_depth_recovers_translated_plane():
    height, width, focal, depth, baseline = 24, 32, 30.0, 3.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    neighbor = torch.roll(image, -disparity, 1)
    neighbor[:, -disparity:] = 0
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[None, None, None].repeat(1, 1, 3, 1, 1), reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=48, min_depth=1.0, max_depth=6.0, min_confidence=0.001,
    )

    recovered = out["depth"][out["valid"]]
    assert recovered.numel() > 200
    assert torch.isclose(recovered.median(), torch.tensor(depth), atol=0.25)


def test_dense_pose_depth_supports_patch_photometric_cost():
    height, width, focal, depth, baseline = 24, 32, 30.0, 3.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    neighbor = torch.roll(image, -disparity, 1)
    neighbor[:, -disparity:] = 0
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[None, None, None].repeat(1, 1, 3, 1, 1), reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=48, min_depth=1.0, max_depth=6.0, patch_radius=1, min_confidence=0.001,
    )

    recovered = out["depth"][out["valid"]]
    assert recovered.numel() > 150
    assert torch.isclose(recovered.median(), torch.tensor(depth), atol=0.25)


def test_dense_pose_depth_supports_edge_aware_cost_regularization():
    height, width, focal, depth, baseline = 24, 32, 30.0, 3.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    neighbor = torch.roll(image, -disparity, 1)
    neighbor[:, -disparity:] = 0
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[None, None, None].repeat(1, 1, 3, 1, 1), reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=48, min_depth=1.0, max_depth=6.0, spatial_cost_radius=1, min_confidence=0.001,
    )

    assert torch.isclose(out["depth"][out["valid"]].median(), torch.tensor(depth), atol=0.25)


def test_dense_pose_depth_supports_cost_volume_aggregation():
    height, width, focal, depth, baseline = 24, 32, 30.0, 3.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    neighbor = torch.roll(image, -disparity, 1)
    neighbor[:, -disparity:] = 0
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[None, None, None].repeat(1, 1, 3, 1, 1), reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=48, min_depth=1.0, max_depth=6.0, min_confidence=0.001, cost_aggregation="sgm",
    )

    assert torch.isclose(out["depth"][out["valid"]].median(), torch.tensor(depth), atol=0.25)


def test_dense_pose_depth_refines_around_per_pixel_inverse_depth_prior():
    height, width, focal, depth, baseline = 24, 40, 40.0, 2.73, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.47) + 0.2 * torch.sin(y * 0.31)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    shift = focal * baseline / depth
    grid = torch.stack(((x.float() + shift) * 2 / (width - 1) - 1, y.float() * 2 / (height - 1) - 1), -1)[None]
    neighbor = torch.nn.functional.grid_sample(reference, grid, align_corners=True)
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[:, None], reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=16, min_depth=1.0, max_depth=6.0, min_confidence=0.001,
        cost_aggregation="sgm",
        inverse_depth_center=torch.full((1, 1, height, width), 1 / 2.6),
        inverse_depth_radius=0.08,
    )

    assert torch.isclose(out["depth"][out["valid"]].median(), torch.tensor(depth), atol=0.05)


def test_dense_pose_depth_keeps_invalid_inverse_depth_prior_invalid():
    height, width, focal, depth, baseline = 24, 40, 40.0, 2.73, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.47) + 0.2 * torch.sin(y * 0.31)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    shift = focal * baseline / depth
    grid = torch.stack(((x.float() + shift) * 2 / (width - 1) - 1, y.float() * 2 / (height - 1) - 1), -1)[None]
    neighbor = torch.nn.functional.grid_sample(reference, grid, align_corners=True)
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline
    prior = torch.full((1, 1, height, width), 1 / depth)
    prior[:, :, 12, 20] = 0

    out = dense_pose_depth(
        reference, neighbor[:, None], reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=16, min_depth=1.0, max_depth=6.0, min_confidence=0.001,
        cost_aggregation="sgm", inverse_depth_center=prior, inverse_depth_radius=0.08,
    )

    assert not out["valid"][0, 0, 12, 20]


def test_dense_pose_depth_rejects_high_patch_error():
    height, width, focal, baseline = 24, 32, 30.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    neighbor = torch.zeros_like(reference)
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].clone()
    neighbor_pose[:, :, 0, 3] = baseline

    out = dense_pose_depth(
        reference, neighbor[:, None], reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=16, min_depth=1.0, max_depth=6.0, min_confidence=0.0,
        max_photometric_error=0.01,
    )

    assert not out["valid"].any()


def test_dense_pose_depth_rejects_disagreeing_views():
    height, width, focal, depth, baseline = 24, 32, 30.0, 3.0, 0.2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (0.5 + 0.2 * torch.sin(x * 0.5) + 0.2 * torch.sin(y * 0.4)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    matching = torch.roll(image, -disparity, 1)
    matching[:, -disparity:] = 0
    disagreeing = torch.zeros_like(image)
    intrinsics = torch.tensor([[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]])
    reference_pose = torch.eye(4)[None]
    neighbor_pose = reference_pose[:, None].repeat(1, 2, 1, 1)
    neighbor_pose[:, :, 0, 3] = baseline
    neighbors = torch.stack((matching, disagreeing), 0)[None, :, None].repeat(1, 1, 3, 1, 1)

    out = dense_pose_depth(
        reference, neighbors, reference_pose, neighbor_pose, intrinsics,
        num_hypotheses=48, min_depth=1.0, max_depth=6.0, min_views=2,
        min_confidence=0.0, max_view_cost_std=0.01,
    )

    assert not out["valid"].any()
