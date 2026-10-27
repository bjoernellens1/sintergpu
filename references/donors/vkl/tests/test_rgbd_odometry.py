import pytest
import torch

from vkl.bench.rgbd_odometry import (
    estimate_map_rendered_rgbd_odometry,
    estimate_rgbd_odometry,
    run_causal_rgbd_odometry,
    run_prepared_causal_rgbd_odometry,
    run_prepared_map_rendered_rgbd_odometry,
    run_prepared_temporal_cache_rgbd_odometry,
)
from vkl.bench.sparse_landmark_odometry import (
    estimate_sparse_landmark_pose,
    relocalize_sparse_landmarks,
    run_prepared_sparse_landmark_rgbd_odometry,
)
from vkl.data.consistent import wall_floor_boxes
from vkl.kernels import (
    apply_camera_local_tracking_update,
    brief_describe,
    depth_geometry,
    sparse_depth_validate,
    sparse_rgbd_observation,
    uniform_select,
)


def _inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = 16, 20
    y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
    depth = (2.0 + 0.15 * x + 0.08 * y + 0.12 * x.square() - 0.06 * y.square()).reshape(1, 1, height, width).expand(batch, -1, -1, -1).contiguous()
    rgb = (0.5 + 0.2 * x + 0.15 * y + 0.1 * x * y).reshape(1, 1, height, width).expand(batch, 3, -1, -1).contiguous()
    intrinsics = torch.tensor([[[20.0, 0.0, 9.5], [0.0, 20.0, 7.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    reference_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    initial_pose = reference_pose.clone()
    initial_pose[:, 2, 3] = 0.1
    return rgb, depth, intrinsics, reference_pose, initial_pose


def test_prepared_sparse_landmark_odometry_bootstraps_and_tracks_without_pose_input() -> None:
    frame = wall_floor_boxes(height=96, width=128)
    batches = [
        {"rgb": frame["rgb"], "depth": frame["depth"], "intrinsics": frame["intrinsics"], "timestamp": torch.tensor([index / 30.0])}
        for index in range(3)
    ]
    out = run_prepared_sparse_landmark_rgbd_odometry(
        batches, points=64, capacity=128, min_sparse_matches=8, max_hamming_distance=0
    )
    assert out["accepted"].tolist() == [True, True, True]
    assert out["active_count"][0] > 0
    assert (out["matched_count"][1:] >= 8).all()


def test_sparse_landmark_relocalization_fails_closed_for_bad_matches() -> None:
    frame = wall_floor_boxes(height=96, width=128)
    batches = [
        {"rgb": frame["rgb"], "depth": frame["depth"], "intrinsics": frame["intrinsics"], "timestamp": torch.tensor([0.0])},
        {"rgb": 1 - frame["rgb"], "depth": frame["depth"], "intrinsics": frame["intrinsics"], "timestamp": torch.tensor([1 / 30.0])},
    ]

    out = run_prepared_sparse_landmark_rgbd_odometry(
        batches,
        points=64,
        capacity=128,
        min_sparse_matches=8,
        max_hamming_distance=0,
        enable_relocalization=True,
        max_relocalization_rmse_m=0.03,
    )

    assert out["accepted"].tolist() == [True, False]
    assert out["health_reason_bits"].tolist() == [0, 1]


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_landmark_pose_composition_preserves_b1_b8(batch: int) -> None:
    frame = wall_floor_boxes(height=96, width=128)
    rgb = frame["rgb"].expand(batch, -1, -1, -1).contiguous()
    depth = frame["depth"].expand(batch, -1, -1, -1).contiguous()
    intrinsics = frame["intrinsics"].expand(batch, -1, -1).clone()
    selected = uniform_select(rgb, num_points=64)
    described = brief_describe(rgb, selected["pixels_xy"], pixels_valid=selected["valid"])
    measured = sparse_depth_validate(depth, selected["pixels_xy"], pixels_valid=selected["valid"] & described["valid"])
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    observed = sparse_rgbd_observation(selected["pixels_xy"], measured["depth"], intrinsics, pose)
    valid = selected["valid"] & described["valid"] & observed["valid"]
    out = estimate_sparse_landmark_pose(
        rgb=rgb, depth=depth, intrinsics=intrinsics, landmark_xyz_world=observed["xyz_world"],
        landmark_descriptor=described["descriptor"], landmark_valid=valid, initial_pose_c2w=pose,
        points=64, max_hamming_distance=0,
    )
    assert out["pose_c2w"].shape == (batch, 4, 4)
    assert out["matched_count"].shape == (batch,)
    assert out["valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_sparse_landmark_relocalization_recovers_identity_without_pose_seed(batch: int) -> None:
    frame = wall_floor_boxes(height=96, width=128)
    rgb = frame["rgb"].expand(batch, -1, -1, -1).contiguous()
    depth = frame["depth"].expand(batch, -1, -1, -1).contiguous()
    intrinsics = frame["intrinsics"].expand(batch, -1, -1).clone()
    selected = uniform_select(rgb, num_points=64)
    described = brief_describe(rgb, selected["pixels_xy"], pixels_valid=selected["valid"])
    measured = sparse_depth_validate(depth, selected["pixels_xy"], pixels_valid=selected["valid"] & described["valid"])
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    observed = sparse_rgbd_observation(selected["pixels_xy"], measured["depth"], intrinsics, pose)
    valid = selected["valid"] & described["valid"] & observed["valid"]

    out = relocalize_sparse_landmarks(
        rgb=rgb, depth=depth, intrinsics=intrinsics,
        landmark_xyz_world=observed["xyz_world"], landmark_descriptor=described["descriptor"],
        landmark_valid=valid, points=64, max_hamming_distance=0,
    )

    assert out["valid"].all()
    assert (out["matched_count"] >= 6).all()
    assert torch.allclose(out["pose_c2w"], pose, atol=1e-5)


@pytest.mark.parametrize("batch", [1, 8])
def test_rgbd_odometry_estimates_camera_pose_without_groundtruth_input(batch: int):
    rgb, depth, intrinsics, reference_pose, initial_pose = _inputs(batch)

    out = estimate_rgbd_odometry(
        source_rgb=rgb,
        source_depth=depth,
        source_intrinsics=intrinsics,
        reference_rgb=rgb,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        initial_pose_c2w=initial_pose,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=rgb.shape[-2] * rgb.shape[-1],
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert out["valid"].all()
    assert (out["pose_c2w"][:, 2, 3].abs() < initial_pose[:, 2, 3].abs()).all()
    assert out["health_valid"].shape == (batch, 2)


@pytest.mark.parametrize("batch", [1, 8])
def test_rgbd_odometry_rejects_missing_depth_without_updating_pose(batch: int):
    rgb, depth, intrinsics, reference_pose, initial_pose = _inputs(batch)

    out = estimate_rgbd_odometry(
        source_rgb=rgb,
        source_depth=torch.zeros_like(depth),
        source_intrinsics=intrinsics,
        reference_rgb=rgb,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        initial_pose_c2w=initial_pose,
        scales=("/1",),
        iterations_per_level=1,
        target_pixels=rgb.shape[-2] * rgb.shape[-1],
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert not out["valid"].any()
    assert torch.equal(out["pose_c2w"], initial_pose)


def test_causal_odometry_uses_timestamp_cadence_without_groundtruth_poses() -> None:
    rgb, depth, intrinsics, _, _ = _inputs(1)
    frames_rgb = rgb.expand(4, -1, -1, -1).contiguous()
    frames_depth = depth.expand(4, -1, -1, -1).contiguous()
    frames_intrinsics = intrinsics.expand(4, -1, -1).contiguous()

    out = run_causal_rgbd_odometry(
        rgb=frames_rgb,
        depth=frames_depth,
        intrinsics=frames_intrinsics,
        timestamps=torch.tensor(
            [1_341_847_980.00, 1_341_847_980.02, 1_341_847_980.04, 1_341_847_980.08], dtype=torch.float64
        ),
        fps=30.0,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert out["selected_indices"].tolist() == [0, 2, 3]
    assert out["pose_c2w"].shape == (3, 4, 4)
    assert out["accepted"].all()
    assert out["reference_kind"].tolist() == [2, 0, 0]


def test_causal_odometry_keeps_last_accepted_reference_after_rejection() -> None:
    rgb, depth, intrinsics, _, _ = _inputs(1)
    frames_rgb = rgb.expand(3, -1, -1, -1).contiguous()
    frames_depth = depth.expand(3, -1, -1, -1).contiguous()
    frames_depth[1] = 0
    frames_intrinsics = intrinsics.expand(3, -1, -1).contiguous()

    out = run_causal_rgbd_odometry(
        rgb=frames_rgb,
        depth=frames_depth,
        intrinsics=frames_intrinsics,
        timestamps=torch.tensor([0.00, 0.04, 0.08]),
        fps=30.0,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert out["accepted"].tolist() == [True, False, True]


def test_prepared_causal_odometry_consumes_b1_batches_and_emits_only_accepted_frames() -> None:
    rgb, depth, intrinsics, _, _ = _inputs(1)
    invalid_depth = torch.zeros_like(depth)
    accepted_frames: list[tuple[str, torch.Tensor]] = []
    batches = [
        {"rgb": rgb, "depth": depth, "intrinsics": intrinsics, "timestamp": torch.tensor([0.0]), "frame_ids": ["0"]},
        {"rgb": rgb, "depth": invalid_depth, "intrinsics": intrinsics, "timestamp": torch.tensor([0.04]), "frame_ids": ["1"]},
        {"rgb": rgb, "depth": depth, "intrinsics": intrinsics, "timestamp": torch.tensor([0.08]), "frame_ids": ["2"]},
    ]

    out = run_prepared_causal_rgbd_odometry(
        batches,
        fps=30.0,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
        on_accepted=lambda batch, pose: accepted_frames.append((batch["frame_ids"][0], pose.clone())),
    )

    assert out["accepted"].tolist() == [True, False, True]
    assert out["input_indices"].tolist() == [0, 1, 2]
    assert [frame_id for frame_id, _ in accepted_frames] == ["0", "2"]


@pytest.mark.parametrize("batch", [1, 8])
def test_map_rendered_odometry_uses_prior_world_points_as_its_reference(batch: int) -> None:
    rgb, depth, intrinsics, reference_pose, _ = _inputs(batch)
    geometry = depth_geometry(depth, intrinsics, reference_pose)
    map_xyz = geometry["xyz_world"].permute(0, 2, 3, 1).reshape(batch, -1, 3)
    map_rgb = rgb.permute(0, 2, 3, 1).reshape(batch, -1, 3)

    out = estimate_map_rendered_rgbd_odometry(
        source_rgb=rgb,
        source_depth=depth,
        source_intrinsics=intrinsics,
        map_xyz_world=map_xyz,
        map_rgb=map_rgb,
        initial_pose_c2w=reference_pose,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert (out["map_reference_valid"].float().mean(dim=(1, 2, 3)) > 0.9).all()
    assert out["valid"].all()


def test_prepared_map_rendered_odometry_updates_a_voxelized_prior_only_map() -> None:
    rgb, depth, intrinsics, _, _ = _inputs(1)
    batches = [
        {"rgb": rgb, "depth": depth, "intrinsics": intrinsics, "timestamp": torch.tensor([timestamp])}
        for timestamp in (0.0, 0.04, 0.08)
    ]

    out = run_prepared_map_rendered_rgbd_odometry(
        batches,
        fps=30.0,
        voxel_size=0.1,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert out["accepted"].tolist() == [True, True, True]
    assert out["map_xyz_world"].ndim == 3
    assert out["map_xyz_world"].shape[0] == 1
    assert out["map_xyz_world"].shape[1] < depth.numel() * len(batches)


def test_prepared_temporal_cache_odometry_uses_only_accepted_prior_frames() -> None:
    rgb, depth, intrinsics, _, _ = _inputs(1)
    invalid_depth = torch.zeros_like(depth)
    batches = [
        {"rgb": rgb, "depth": candidate_depth, "intrinsics": intrinsics, "timestamp": torch.tensor([timestamp])}
        for timestamp, candidate_depth in ((0.0, depth), (0.04, invalid_depth), (0.08, depth))
    ]

    out = run_prepared_temporal_cache_rgbd_odometry(
        batches,
        fps=30.0,
        cache_size=2,
        scales=("/1",),
        iterations_per_level=2,
        target_pixels=16 * 20,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )

    assert out["accepted"].tolist() == [True, False, True]
    assert out["reference_cache_size"].tolist() == [0, 1, 1]


@pytest.mark.parametrize("batch", [1, 8])
def test_rgbd_odometry_corrects_camera_local_yaw_when_reference_is_rotated(batch: int) -> None:
    reference = wall_floor_boxes(height=32, width=48)
    source = wall_floor_boxes(height=32, width=48, camera_yaw_rad=0.12)
    reference_pose = reference["pose_c2w"].expand(batch, -1, -1).clone()
    true_pose = source["pose_c2w"].expand(batch, -1, -1).clone()
    initial_pose = apply_camera_local_tracking_update(
        true_pose, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.05, 0.0]]).expand(batch, -1)
    )

    out = estimate_rgbd_odometry(
        source_rgb=source["rgb"].expand(batch, -1, -1, -1),
        source_depth=source["depth"].expand(batch, -1, -1, -1),
        source_intrinsics=source["intrinsics"].expand(batch, -1, -1),
        reference_rgb=reference["rgb"].expand(batch, -1, -1, -1),
        reference_depth=reference["depth"].expand(batch, -1, -1, -1),
        reference_intrinsics=reference["intrinsics"].expand(batch, -1, -1),
        reference_pose_c2w=reference_pose,
        initial_pose_c2w=initial_pose,
        scales=("/1",),
        iterations_per_level=4,
        target_pixels=32 * 48,
        min_width=1,
        min_height=1,
        partial_grid=(2, 2),
    )
    initial_error = torch.linalg.vector_norm(initial_pose[:, :3, :3] - true_pose[:, :3, :3], dim=(-2, -1))
    final_error = torch.linalg.vector_norm(out["pose_c2w"][:, :3, :3] - true_pose[:, :3, :3], dim=(-2, -1))
    assert (final_error < initial_error).all()
