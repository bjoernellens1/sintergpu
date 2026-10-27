"""Warmed GPU frame benchmark with resident input and independent pose checks."""

import platform
from importlib.metadata import version
from time import perf_counter

import numpy as np


def run_benchmark(width=160, height=120, iterations=5):
    """Timestamp source maps and all ICP iterations, excluding frame-boundary transfers."""
    for name, value in (("width", width), ("height", height), ("iterations", iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    import slangpy as spy

    from .synthetic import synthetic_sequence
    from .tracker import VulkanTracker

    camera, frames = synthetic_sequence(frames=2, width=width, height=height)
    expected = frames[1].world_from_camera
    reference, current = frames[0].depth, frames[1].depth
    tracker = VulkanTracker(camera)
    tracker.set_reference(reference)
    device = tracker.runtime.device

    # Correctness precedes timing. Failure must never yield a successful timing report.
    warmups = 3
    for _ in range(warmups):
        result = tracker.track(current)
        if not result.success:
            raise RuntimeError("benchmark fixture tracking failed during warmup")
        np.testing.assert_allclose(result.pose, expected, atol=0.004, rtol=0)

    tracker.levels[0]["maps"]["source"]["depth"].copy_from_numpy(tracker._depth(current))
    tracker.initial_pose.copy_from_numpy(np.eye(4, dtype=np.float32).ravel())
    zero_status = tracker.runtime.buffer(np.zeros(2, np.uint32))
    saved_poses = tracker.runtime.buffer(np.zeros(iterations * 16, np.float32))
    saved_status = tracker.runtime.buffer(np.zeros(iterations * 2, np.uint32))
    saved_cost = tracker.runtime.buffer(np.zeros(iterations, np.float32))
    frequency = int(device.info.timestamp_frequency)
    if frequency <= 0:
        raise RuntimeError("device reports no GPU timestamp frequency")
    pool = device.create_query_pool(spy.QueryType.timestamp, 2 * iterations)
    device.wait_for_idle()

    start = perf_counter()
    encoder = device.create_command_encoder()
    for i in range(iterations):
        # Each sample starts from the same pose/status, not the last converged pose.
        # Copies and tiny diagnostic snapshots lie outside the timestamp pair.
        encoder.copy_buffer(tracker.pose, 0, tracker.initial_pose, 0, 64)
        encoder.copy_buffer(tracker.status, 0, zero_status, 0, 8)
        encoder.write_timestamp(pool, 2 * i)
        tracker._record_track(encoder)
        encoder.write_timestamp(pool, 2 * i + 1)
        encoder.copy_buffer(saved_poses, 64 * i, tracker.pose, 0, 64)
        encoder.copy_buffer(saved_status, 8 * i, tracker.status, 0, 8)
        encoder.copy_buffer(saved_cost, 4 * i, tracker.diagnostics, 0, 4)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    wall_ms = (perf_counter() - start) * 1000

    poses = saved_poses.to_numpy().view(np.float32).reshape(iterations, 4, 4)
    status = saved_status.to_numpy().view(np.uint32).reshape(iterations, 2)
    costs = saved_cost.to_numpy().view(np.float32).reshape(iterations)
    if np.any(status[:, 1] != 0) or np.any(status[:, 0] < tracker.minimum):
        raise RuntimeError("benchmark measured tracking failed")
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise RuntimeError("benchmark diagnostic costs are invalid")
    for pose in poses:
        np.testing.assert_allclose(pose, expected, atol=0.004, rtol=0)
    ticks = pool.get_results(0, 2 * iterations)
    differences = np.asarray([ticks[2 * i + 1] - ticks[2 * i] for i in range(iterations)])
    if np.any(differences <= 0):
        raise RuntimeError("GPU timestamps must be strictly increasing")
    samples = differences * (1000.0 / frequency)
    return {
        "width": width,
        "height": height,
        "iterations": iterations,
        "warmups": warmups,
        "pyramid_levels": len(tracker.levels),
        "gn_iterations_per_frame": sum(tracker.iterations),
        "measured_submissions": 1,
        "device": device.info.adapter_name,
        "api": device.info.api_name,
        "versions": {
            name: version(name) for name in ("slangpy", "symforce", "numpy", "caspar-slang")
        },
        "python": platform.python_version(),
        "os": platform.platform(),
        "timestamp_frequency_hz": frequency,
        "gpu_ms": {
            "min": float(samples.min()),
            "median": float(np.median(samples)),
            "mean": float(samples.mean()),
            "p95": float(np.percentile(samples, 95)),
            "samples": samples.tolist(),
        },
        "host_wall_ms_per_frame": wall_ms / iterations,
        "gpu_boundary": "source depth pyramid + vertex/normal maps + association + Caspar residual/Jacobian + hierarchical subgroup normal-equation reductions + 6x6 solve + SE(3) updates; 17 GN iterations",
        "host_boundary": "record + pose/status reset copies + tiny GPU diagnostic snapshots + one submission + wait, amortized per frame",
        "excluded": [
            "fixture rendering",
            "symbolic generation",
            "compilation",
            "buffer allocation",
            "reference maps",
            "input uploads",
            "warmups",
            "query retrieval",
            "diagnostic readback",
        ],
        "validation": {
            "all_samples_passed": True,
            "pose_max_abs_error": float(np.max(np.abs(poses - expected))),
            "correspondences": status[:, 0].tolist(),
            "weighted_rmse": np.sqrt(costs / status[:, 0]).tolist(),
        },
    }
