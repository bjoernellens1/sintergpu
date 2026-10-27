import inspect

import numpy as np
import pytest

from slang_rgbd_slam.cpu import _exp
from slang_rgbd_slam.cpu_hybrid import (
    CPUFusedTracker,
    CPUHybridSLAM,
    optimize_dense_window,
    optimize_pose_graph,
)
from slang_rgbd_slam.synthetic import render_depth, synthetic_sequence
from slang_rgbd_slam.tracker import Camera


def test_cpu_hybrid_has_no_runtime_or_gpu_allocation():
    import slang_rgbd_slam.cpu_hybrid as module

    source = inspect.getsource(module)
    assert "import slangpy" not in source
    assert "VulkanRuntime" not in source


def test_joint_tracker_identity_and_unknown_model_intensity():
    camera = Camera(64, 48, 60, 60, 31.5, 23.5)
    depth = render_depth(camera, np.eye(4))
    rgb = np.random.default_rng(2).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    tracker = CPUFusedTracker(camera, levels=1, iterations=(4,))
    tracker.set_reference(depth, np.full(depth.shape, -1, np.float32))
    result = tracker.track(depth, rgb)
    assert result.success
    np.testing.assert_allclose(result.pose, np.eye(4), atol=1e-9)


@pytest.mark.parametrize("window", [2, 4])
def test_dense_window_empty_rolls_back_and_coupled_fixture_decreases_cost(window):
    camera = Camera(32, 24, 30, 30, 15.5, 11.5)
    poses = np.repeat(np.eye(4)[None], window, axis=0)
    empty = np.zeros((window, 24, 32), np.float32)
    bad = optimize_dense_window(camera, empty, empty, poses)
    assert not bad["success"]
    np.testing.assert_array_equal(bad["poses"], poses)
    truth = poses.copy()
    for i in range(1, window):
        truth[i] = _exp(i * np.array([0.005, -0.004, 0.002, 0.006, -0.003, 0.008]))
    depths = np.stack([render_depth(camera, p) for p in truth])
    gray = np.zeros_like(depths)
    result = optimize_dense_window(camera, depths, gray, poses)
    assert result["success"]
    assert result["final_cost"] < result["initial_cost"]
    assert result["final_factors"] >= 0.8 * result["initial_factors"]


def test_anchored_pose_graph_corrects_full_rotation_translation():
    truth = np.stack([_exp(np.array([0, 0.05 * i, 0, 0.1 * i, 0, 0.02 * i])) for i in range(4)])
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    measurements = np.stack([np.linalg.inv(truth[a]) @ truth[b] for a, b in edges])
    initial = truth.copy()
    for i in range(1, 4):
        initial[i] = _exp(np.array([0.015 * i, 0, 0.01 * i, 0, 0.03 * i, 0])) @ initial[i]
    result = optimize_pose_graph(initial, edges, measurements)
    assert result["success"]
    np.testing.assert_array_equal(result["poses"][0], initial[0])
    np.testing.assert_allclose(result["poses"], truth, atol=2e-4)


def test_bounded_dense_sequence_runs_without_ground_truth_initialization():
    camera, frames = synthetic_sequence(12, 160, 120)
    slam = CPUHybridSLAM(camera, max_keyframes=3, keyframe_interval=3, ba_every_keyframes=2)
    results = [slam.process(f.depth, f.rgb, timestamp=f.timestamp) for f in frames]
    assert sum(r["success"] for r in results) >= 10
    assert len(slam.active_keyframes) <= 3
    assert slam.ba_events
    assert any(e["success"] for e in slam.ba_events)
    assert all(r["mapping_success"] for r in results)


def test_sparse_verification_recovers_textured_rgbd_pose_without_gt_seed():
    camera = Camera(96, 72, 90, 90, 47.5, 35.5)
    depth = render_depth(camera, np.eye(4))
    rgb = np.random.default_rng(4).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    slam = CPUHybridSLAM(camera, tracker_options=dict(levels=1, iterations=(2,)))
    features = slam.frontend.extract(rgb, depth)
    assert features["count"] >= 20
    slam._capture(depth, rgb, features, 0.0)
    pose = slam._verify_loop(slam.frontend.extract(rgb, depth), 0)
    assert pose is not None
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-8)


def test_verified_loop_graph_corrects_history_and_retains_measurement():
    camera = Camera(64, 48, 60, 60, 31.5, 23.5)
    depth = render_depth(camera, np.eye(4))
    rgb = np.random.default_rng(8).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    slam = CPUHybridSLAM(camera, tracker_options=dict(levels=1, iterations=(1,)))
    features = slam.frontend.extract(rgb, depth)
    slam._capture(depth, rgb, features, 0.0)
    slam.index = 5
    slam.world_from_camera = _exp(np.array([0, 0.02, 0, 0.05, 0, 0.02]))
    slam._capture(depth, rgb, slam.frontend.extract(rgb, depth), 1.0)
    old = slam.world_from_camera.copy()
    slam.trajectory = [(0.0, np.eye(4)), (1.0, old.copy())]
    slam._corrected_history = [(timestamp, pose.copy()) for timestamp, pose in slam.trajectory]
    slam.history_anchors = [(0, 1), (1, 6)]
    assert slam._loop_graph(np.eye(4), 0, 1)
    assert len(slam.retained_loops) == 1
    assert np.linalg.norm(slam.world_from_camera[:3, 3]) < np.linalg.norm(old[:3, 3]) * 0.1
    np.testing.assert_allclose(
        slam.corrected_trajectory()[-1][1], slam.world_from_camera, atol=1e-10
    )
    # Slot reuse must not correct historical frames anchored to an evicted ID.
    preserved = slam.corrected_trajectory()[0][1].copy()
    slam.slots[0]["frame_id"] = 99
    previous = {0: slam.slots[0]["pose"].copy()}
    slam.slots[0]["pose"] = _exp(np.array([0, 0, 0, 0.1, 0, 0]))
    slam._correct_history(previous)
    np.testing.assert_array_equal(slam.corrected_trajectory()[0][1], preserved)


def test_repeated_ba_keeps_sensor_prior_and_reconstructs_full_depth_without_compounding(
    monkeypatch,
):
    import slang_rgbd_slam.cpu_hybrid as module

    camera = Camera(64, 48, 60, 60, 31.5, 23.5)
    depth = render_depth(camera, np.eye(4))
    rgb = np.full((48, 64, 3), 180, np.uint8)
    slam = CPUHybridSLAM(
        camera, tracker_options=dict(levels=1, iterations=(1,)), map_options=dict(max_blocks=256)
    )
    for i in range(2):
        slam.index = i
        slam._capture(depth, rgb, slam.frontend.extract(rgb, depth), i)
    observations = []

    def bounded_result(camera, measured, gray, poses, *, initial_depths):
        observations.append((measured.copy(), initial_depths.copy()))
        rho = np.divide(1, measured * 1.1, out=np.zeros_like(measured), where=measured > 0)
        return dict(
            success=True,
            poses=poses.copy(),
            original_poses=poses.copy(),
            inverse_depth=rho,
            initial_cost=1.0,
            final_cost=0.5,
            initial_factors=100,
            final_factors=100,
        )

    monkeypatch.setattr(module, "optimize_dense_window", bounded_result)
    slam._ba(2.0)
    once = slam.slots[0]["depth"].copy()
    slam._ba(3.0)
    np.testing.assert_array_equal(observations[0][0], observations[1][0])
    np.testing.assert_allclose(observations[1][1], observations[0][0] * 1.1, rtol=1e-6)
    np.testing.assert_array_equal(slam.slots[0]["raw_depth"], depth)
    np.testing.assert_array_equal(slam.slots[0]["measured_low_depth"], depth)
    np.testing.assert_array_equal(slam.slots[0]["depth"], once)
    np.testing.assert_allclose(once, depth * 1.1, rtol=1e-6)


def test_dense_solver_prior_remains_observation_with_distinct_initial_estimate():
    camera = Camera(32, 24, 30, 30, 15.5, 11.5)
    depth = render_depth(camera, np.eye(4))
    measured = np.stack([depth, depth])
    estimate = measured * 1.02
    original = measured.copy()
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    result = optimize_dense_window(
        camera, measured, np.zeros_like(measured), poses, initial_depths=estimate
    )
    assert result["success"]
    assert result["initial_cost"] > 100 * np.sum((1 / estimate - 1 / measured) ** 2) * 0.99
    np.testing.assert_array_equal(measured, original)
    rho = result["inverse_depth"]
    valid = measured > 0
    ratio = 1 / (measured[valid] * rho[valid])
    assert np.min(ratio) >= 0.9 - 1e-6 and np.max(ratio) <= 1.1 + 1e-6
    assert result["final_cost"] <= result["initial_cost"] + 1e-7
