import torch

from vkl.kernels import direct_pose_depth_tensorized, dso_select_points
from vkl.kernels.reference.direct_pose_depth import direct_pose_depth_dso


def test_direct_pose_depth_recovers_translated_plane():
    h, w, focal, depth, baseline = 64, 96, 60.0, 3.0, 0.1
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    image = (0.5 + 0.25 * torch.sin(x * 0.35) + 0.2 * torch.sin(y * 0.27)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth)
    neighbor = torch.roll(image, -disparity, 1)
    neighbor[:, -disparity:] = 0
    neighbors = neighbor[None, None, None].repeat(1, 1, 3, 1, 1)
    intrinsics = torch.tensor([[[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]]], dtype=torch.float32)
    pose = torch.eye(4)[None]
    neighbor_pose = torch.eye(4).reshape(1, 1, 4, 4); neighbor_pose[:, :, 0, 3] = baseline
    out = direct_pose_depth_dso(rgb_reference=reference, rgb_neighbors=neighbors, pose_reference=pose, pose_neighbors=neighbor_pose, intrinsics=intrinsics, num_points=256, num_hypotheses=80, min_depth=1.0, max_depth=8.0, min_confidence=0.001)
    recovered = out["depth"][out["valid"]]
    assert recovered.numel() > 100
    assert torch.isclose(recovered.median(), torch.tensor(depth), atol=0.15)


def test_tensorized_dso_recovers_depth_with_two_neighbors():
    h, w, focal, depth = 48, 72, 60.0, 3.0
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    image = (0.5 + 0.25 * torch.sin(x * 0.35) + 0.2 * torch.sin(y * 0.27)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    baselines = torch.tensor([0.10, 0.15])
    neighbors = []
    for baseline in baselines:
        disparity = round(float(focal * baseline / depth)); neighbor = torch.roll(image, -disparity, 1); neighbor[:, -disparity:] = 0; neighbors.append(neighbor)
    neighbors = torch.stack(neighbors)[None, :, None].repeat(1, 1, 3, 1, 1)
    intrinsics = torch.tensor([[[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]]])
    pose = torch.eye(4)[None]
    neighbor_pose = torch.eye(4).repeat(2, 1, 1)[None]; neighbor_pose[0, :, 0, 3] = baselines
    out = direct_pose_depth_tensorized(rgb_reference=reference, rgb_neighbors=neighbors, pose_reference=pose, pose_neighbors=neighbor_pose, intrinsics=intrinsics, num_points=160, num_hypotheses=64, min_depth=1.0, max_depth=8.0, min_confidence=0.001)
    recovered = out["depth"][out["valid"]]
    assert recovered.numel() > 70
    assert torch.isclose(recovered.median(), torch.tensor(depth), atol=0.2)


def test_dso_select_points_returns_bounded_gradient_evidence():
    rgb = torch.zeros(1, 3, 16, 24); rgb[:, :, :, 12:] = 1
    out = dso_select_points(rgb, num_points=12, patch_radius=1)
    assert out["pixels_xy"].shape == (1, 12, 2)
    assert out["gradient"].shape == (1, 12)
    assert out["valid"].any()
    assert (out["gradient"][~out["valid"]] == 0).all()
    valid_x = out["pixels_xy"][0, out["valid"][0], 0]
    assert (valid_x - 12).abs().median() <= 2


def test_dso_accepts_external_selector_coordinates_without_internal_reselection():
    h, w, focal, depth, baseline = 48, 72, 60.0, 3.0, 0.1
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    image = (0.5 + 0.25 * torch.sin(x * 0.35) + 0.2 * torch.sin(y * 0.27)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth); neighbor = torch.roll(image, -disparity, 1); neighbor[:, -disparity:] = 0
    pixels_xy = dso_select_points(reference, num_points=96, patch_radius=1)["pixels_xy"]
    intrinsics = torch.tensor([[[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]]]); pose = torch.eye(4)[None]
    neighbor_pose = torch.eye(4).reshape(1, 1, 4, 4); neighbor_pose[:, :, 0, 3] = baseline
    out = direct_pose_depth_dso(rgb_reference=reference, rgb_neighbors=neighbor[None, None, None].repeat(1, 1, 3, 1, 1), pose_reference=pose, pose_neighbors=neighbor_pose, intrinsics=intrinsics, pixels_xy=pixels_xy, num_points=1, num_hypotheses=64, min_depth=1.0, max_depth=8.0, min_confidence=0.001)
    expected = torch.zeros_like(out["selected"]); expected[0, 0, pixels_xy[0, :, 1], pixels_xy[0, :, 0]] = True
    assert torch.equal(out["selected"], expected)
    assert torch.isclose(out["depth"][out["valid"]].median(), torch.tensor(depth), atol=0.2)


def test_tensorized_dso_accepts_external_selector_coordinates():
    h, w, focal, depth, baseline = 40, 60, 50.0, 2.5, 0.1
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    image = (0.5 + 0.25 * torch.sin(x * 0.35) + 0.2 * torch.sin(y * 0.27)).clamp(0, 1)
    reference = image[None, None].repeat(1, 3, 1, 1)
    disparity = round(focal * baseline / depth); neighbor = torch.roll(image, -disparity, 1); neighbor[:, -disparity:] = 0
    pixels_xy = dso_select_points(reference, num_points=64, patch_radius=1)["pixels_xy"]
    intrinsics = torch.tensor([[[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]]]); pose = torch.eye(4)[None]
    neighbor_pose = torch.eye(4).reshape(1, 1, 4, 4); neighbor_pose[:, :, 0, 3] = baseline
    out = direct_pose_depth_tensorized(rgb_reference=reference, rgb_neighbors=neighbor[None, None, None].repeat(1, 1, 3, 1, 1), pose_reference=pose, pose_neighbors=neighbor_pose, intrinsics=intrinsics, pixels_xy=pixels_xy, num_points=1, num_hypotheses=48, min_depth=1.0, max_depth=8.0, min_confidence=0.001)
    expected = torch.zeros_like(out["selected"]); expected[0, 0, pixels_xy[0, :, 1], pixels_xy[0, :, 0]] = True
    assert torch.equal(out["selected"], expected)
    assert torch.isclose(out["depth"][out["valid"]].median(), torch.tensor(depth), atol=0.25)
