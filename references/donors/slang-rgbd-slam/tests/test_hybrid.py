"""Resident dense/sparse controller integration, lifecycle and GPU trajectory."""

import inspect
import os

import numpy as np
import pytest

from slang_rgbd_slam.hybrid import HybridSLAM


def test_controller_defaults_and_recording_contract():
    signature = inspect.signature(HybridSLAM)
    assert signature.parameters["enable_loops"].default is True
    assert signature.parameters["min_loop_frames"].default == 30
    assert signature.parameters["max_frames"].default == 65536
    for name in ("_record_loop", "_record_after_tracking", "_record_map_rebuild"):
        source = inspect.getsource(getattr(HybridSLAM, name))
        assert "NotImplementedError" not in source
        assert "to_numpy" not in source
        assert "np.linalg" not in source
        assert "submit_command_buffer" not in source
    assert hasattr(HybridSLAM, "corrected_trajectory")


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_gpu_dense_local_ba_and_corrected_history_16_frames():
    from slang_rgbd_slam.synthetic import synthetic_sequence

    camera, frames = synthetic_sequence(16, 160, 120)
    slam = HybridSLAM(
        camera,
        enable_loops=False,
        keyframe_interval=2,
        ba_every_keyframes=2,
        max_keyframes=6,
        max_frames=32,
        map_options={"max_blocks": 2048},
    )
    results, observations = [], {}
    for frame in frames:
        results.append(slam.process(frame.depth, frame.rgb, timestamp=frame.timestamp))
        for slot in slam.active_keyframes:
            saved = slam.keyframe_store.slots[slot]
            key = (slot, slam._slot_ids[slot])
            immutable = tuple(
                saved[name].to_numpy().copy() for name in ("raw_depth", "measured_low_depth")
            )
            if key in observations:
                for original, current in zip(observations[key], immutable):
                    np.testing.assert_array_equal(current, original)
            else:
                observations[key] = immutable
    assert all(r["success"] and r["mapping_success"] for r in results)
    assert len(slam.ba_events) >= 3
    assert any(e["success"] for e in slam.ba_events)
    assert len(slam.active_keyframes) == 6
    history = slam.corrected_trajectory()
    assert len(history) == 16
    assert all(np.isfinite(pose).all() for _, pose in history)
    np.testing.assert_allclose(history[-1][1], results[-1]["world_from_camera"], atol=1e-6)


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_gpu_stationary_textured_loop_and_device_ingestion():
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.synthetic import render_depth
    from slang_rgbd_slam.tracker import Camera

    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    runtime = VulkanRuntime()
    rng = np.random.default_rng(9)
    rgb = np.repeat(rng.integers(0, 256, (120, 160), dtype=np.uint8)[:, :, None], 3, axis=2)
    depth = render_depth(camera, np.eye(4))
    slam = HybridSLAM(
        camera,
        runtime=runtime,
        keyframe_interval=1,
        min_loop_frames=1,
        enable_local_ba=False,
        max_keyframes=4,
        max_features=256,
        max_loop_edges=1,
        max_frames=16,
        map_options={"max_blocks": 2048},
    )
    depth_gpu = runtime.buffer(depth.ravel())
    rgb_gpu = runtime.buffer(rgb.astype(np.float32).ravel())
    results = [slam.process_device(depth_gpu, rgb_gpu, timestamp=i / 30) for i in range(8)]
    assert all(r["success"] and r["mapping_success"] for r in results)
    assert any(e["verified"] and e["accepted"] for e in slam.loop_events)
    assert slam._saved_loops
    assert len(slam.corrected_trajectory()) == 8
    np.testing.assert_allclose(
        slam.corrected_trajectory()[-1][1], results[-1]["world_from_camera"], atol=1e-6
    )

    # FIFO reuse must prune retired generations while a full measurement pool
    # replaces its oldest entry only after successful verification/graph solve.
    assert len(slam.active_keyframes) == 4
    assert len(slam._saved_loops) == 1
    assert all(
        slam._slot_ids[e["candidate_slot"]] == e["candidate_id"]
        and slam._slot_ids[e["query_slot"]] == e["query_id"]
        for e in slam._saved_loops
    )
    retained = slam.loop_measurements[0].to_numpy().copy()

    def request_candidate():
        slam._pending_candidate, slam._pending_query = (
            slam.active_keyframes[0],
            slam.active_keyframes[-1],
        )
        slam._pending_candidate_id = slam._slot_ids[slam._pending_candidate]
        slam._pending_query_id = slam._slot_ids[slam._pending_query]

    # Force dense tracking below its admission threshold while preserving valid
    # RGBD features: sparse verification must recover directly without a PG edge.
    slam.tracker.minimum = 100000
    slam._recovery = True
    request_candidate()
    recovered = slam.process_device(depth_gpu, rgb_gpu, timestamp=8 / 30)
    assert recovered["success"] and recovered["mapping_success"]
    assert recovered["loop"]["recovered"] and not recovered["loop"]["accepted"]
    np.testing.assert_array_equal(slam.loop_measurements[0].to_numpy(), retained)

    # Textureless input cannot verify a proposed loop or overwrite its pool row.
    request_candidate()
    invalid_rgb = runtime.buffer(np.zeros(camera.width * camera.height * 3, np.float32))
    pose_before = slam.world_pose.to_numpy().copy()
    weights_before = slam.map.buffers["weights"].to_numpy().copy()
    rejected = slam.process_device(depth_gpu, invalid_rgb, timestamp=9 / 30)
    assert not rejected["success"]
    np.testing.assert_array_equal(slam.world_pose.to_numpy(), pose_before)
    np.testing.assert_array_equal(slam.map.buffers["weights"].to_numpy(), weights_before)
    assert not rejected["loop"]["verified"] and not rejected["loop"]["accepted"]
    np.testing.assert_array_equal(slam.loop_measurements[0].to_numpy(), retained)


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_gpu_repeated_ba_commit_preserves_sensor_prior_and_exact_accepted_depth(monkeypatch):
    from slang_rgbd_slam.synthetic import synthetic_sequence

    camera, frames = synthetic_sequence(1, 160, 120)
    slam = HybridSLAM(
        camera,
        enable_loops=False,
        enable_local_ba=False,
        max_keyframes=4,
        map_options={"max_blocks": 2048},
        max_frames=8,
    )
    assert slam.process(frames[0].depth, frames[0].rgb)["success"]
    saved = slam.keyframe_store.slots[0]
    raw = saved["raw_depth"].to_numpy().view(np.float32).ravel().copy()
    measured = saved["measured_low_depth"].to_numpy().view(np.float32).ravel().copy()
    low = slam.tracker.levels[-1]["camera"]
    low_indices = (
        (np.arange(camera.height) * low.height // camera.height)[:, None] * low.width
        + (np.arange(camera.width) * low.width // camera.width)[None, :]
    ).ravel()
    for ratio in (1.1, 0.9, 1.05):
        rho = np.divide(1, measured * ratio, out=np.zeros_like(measured), where=measured > 0)
        all_rho = np.zeros(
            slam.ba_problem.inverse_depth.to_numpy().view(np.float32).size, np.float32
        )
        all_rho[: len(rho)] = rho
        slam.ba_problem.inverse_depth.copy_from_numpy(all_rho)
        slam.ba_problem.buffers["status"].copy_from_numpy(np.zeros(1, np.uint32))
        encoder = slam.runtime.device.create_command_encoder()
        slam.keyframe_store.record_ba_commit(
            slam.ba_problem, [0], slam.world_pose, slam.tracker.status, encoder
        )
        slam.runtime.device.submit_command_buffer(encoder.finish())
        committed_low = saved["low_depth"].to_numpy().view(np.float32).ravel()
        committed_full = saved["depth"].to_numpy().view(np.float32).ravel()
        valid = measured > 0
        np.testing.assert_allclose(committed_low[valid], 1 / rho[valid], rtol=1e-6)
        valid_full = valid[low_indices]
        np.testing.assert_allclose(
            committed_full[valid_full],
            raw[valid_full] / (measured[low_indices[valid_full]] * rho[low_indices[valid_full]]),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(saved["raw_depth"].to_numpy().view(np.float32).ravel(), raw)
        np.testing.assert_array_equal(
            saved["measured_low_depth"].to_numpy().view(np.float32).ravel(), measured
        )
    # Observe the next overlapping solve input copies independently of the
    # optimizer: immutable observations and accepted estimates must differ.
    monkeypatch.setattr(slam.dense_ba, "record", lambda *a, **kw: None)
    monkeypatch.setattr(slam.keyframe_store, "record_ba_commit", lambda *a, **kw: None)
    encoder = slam.runtime.device.create_command_encoder()
    slam._record_local_ba([0], encoder)
    slam.runtime.device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(
        slam.ba_problem.buffers["depths"].to_numpy().view(np.float32).ravel()[: len(measured)],
        measured,
    )
    np.testing.assert_allclose(
        slam.ba_problem.buffers["initialDepths"]
        .to_numpy()
        .view(np.float32)
        .ravel()[: len(measured)],
        saved["low_depth"].to_numpy().view(np.float32).ravel(),
        rtol=1e-6,
    )
