import sys

import numpy as np
import pytest

from slang_rgbd_slam.cpu import CPUSLAM, CPUTSDF, CPUTracker
from slang_rgbd_slam.synthetic import render_depth
from slang_rgbd_slam.tracker import Camera


def test_cpu_tracker_motion_and_loss():
    camera = Camera(96, 72, 85, 85, 47.5, 35.5)
    tracker = CPUTracker(camera, levels=2, iterations=(5, 4))
    reference = render_depth(camera, np.eye(4))
    tracker.set_reference(reference)
    reference[:] = 0
    pose = np.eye(4)
    angle = 0.009
    pose[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    pose[:3, 3] = [0.012, -0.006, 0.009]
    result = tracker.track(render_depth(camera, pose))
    assert result.success
    np.testing.assert_allclose(result.pose, pose, atol=0.005)
    failed = tracker.track(np.zeros((72, 96), np.float32), initial_pose=pose)
    assert not failed.success
    np.testing.assert_allclose(failed.pose, pose)
    with pytest.raises(ValueError):
        tracker.track(np.zeros((72, 96), np.float64))


def test_tsdf_plane_fusion_raycast_and_weight_cap():
    camera = Camera(32, 24, 30, 30, 15.5, 11.5)
    mapping = CPUTSDF(camera, voxel_size=0.03, max_blocks=128)
    depth = np.ones((24, 32), np.float32)
    rgb = np.full((24, 32, 3), 100, np.uint8)
    mapping.integrate(depth, rgb, np.eye(4))
    model = mapping.raycast(np.eye(4))
    valid = model > 0
    assert valid.sum() > 300
    np.testing.assert_allclose(model[valid], 1, atol=0.008)
    before = mapping.snapshot()
    mapping.integrate(depth, rgb, np.eye(4))
    after = mapping.snapshot()
    np.testing.assert_allclose(before["distances"], after["distances"])
    np.testing.assert_allclose(after["weights"], before["weights"] * 2)
    for _ in range(205):
        mapping.integrate(depth, rgb, np.eye(4))
    assert mapping.snapshot()["weights"].max() == 100
    assert len(mapping) > 100


def test_cpu_slam_fuses_every_successful_frame_and_loss_does_not_mutate():
    camera = Camera(64, 48, 55, 55, 31.5, 23.5)
    slam = CPUSLAM(
        camera, tracker_options={"levels": 2, "iterations": (3, 3)}, map_options={"max_blocks": 512}
    )
    depth = render_depth(camera, np.eye(4))
    assert slam.process(depth, timestamp=0)["success"]
    assert slam.process(depth, timestamp=1)["success"]
    snapshot = slam.map.snapshot()
    assert not slam.process(np.zeros_like(depth), timestamp=2)["success"]
    np.testing.assert_array_equal(snapshot["weights"], slam.map.snapshot()["weights"])
    assert len(slam.trajectory) == 2


def test_tsdf_overflow_and_invalid_pose():
    camera = Camera(32, 24, 30, 30, 15.5, 11.5)
    mapping = CPUTSDF(camera, max_blocks=1)
    depth = np.ones((24, 32), np.float32)
    mapping.integrate(depth, None, np.eye(4))
    assert mapping.stats()["overflow"] > 0
    assert mapping.stats()["blocks"] == 1
    with pytest.raises(ValueError):
        mapping.integrate(depth, None, np.zeros((4, 4)))


def test_sphere_surface_fusion_and_raycast():
    camera = Camera(64, 48, 55, 55, 31.5, 23.5)
    y, x = np.indices((48, 64))
    rays = np.stack(
        ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones_like(x)), axis=-1
    )
    center = np.array([0, 0, 1.4])
    radius = 0.5
    a = np.sum(rays * rays, axis=-1)
    b = -2 * np.sum(rays * center, axis=-1)
    c = center @ center - radius * radius
    discriminant = b * b - 4 * a * c
    depth = np.where(
        discriminant >= 0, (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a), 0
    ).astype(np.float32)
    mapping = CPUTSDF(camera, max_blocks=256)
    mapping.integrate(depth, None, np.eye(4))
    model = mapping.raycast(np.eye(4))
    valid = (model > 0) & (depth > 0)
    assert valid.sum() > 800
    assert np.sqrt(np.mean((model[valid] - depth[valid]) ** 2)) < 0.01
    before = mapping.snapshot()
    with pytest.raises(ValueError):
        mapping.integrate(depth, np.full((48, 64, 3), 256, np.float32), np.eye(4))
    np.testing.assert_array_equal(before["weights"], mapping.snapshot()["weights"])


def test_downsampling_preserves_holes_and_rejects_depth_jump():
    camera = Camera(16, 16, 15, 15, 7.5, 7.5)
    tracker = CPUTracker(camera, levels=2, iterations=(1, 1))
    depth = np.ones((16, 16), np.float32)
    depth[:2, :2] = [[1, 0], [1, 1]]
    depth[:2, 2:4] = [[1, 1], [1, 2]]
    maps = tracker._maps(depth)
    assert maps[1][0][0, 2] == 1
    assert maps[1][0][1, 2] == 0


def test_cpu_module_does_not_import_gpu_stack():
    import os
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from slang_rgbd_slam.cpu import CPUTracker, CPUTSDF; "
            'assert "slangpy" not in sys.modules; '
            'assert "caspar_slang.runtime" not in sys.modules',
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )


def test_color_confidence_does_not_include_uncolored_free_space():
    camera = Camera(32, 24, 40, 40, 15.5, 11.5)
    mapping = CPUTSDF(camera, max_blocks=128)
    mapping.integrate(
        np.full((24, 32), 1.05, np.float32), np.full((24, 32, 3), 200, np.uint8), np.eye(4)
    )
    mapping.integrate(
        np.full((24, 32), 0.93, np.float32), np.full((24, 32, 3), 100, np.uint8), np.eye(4)
    )
    snap = mapping.snapshot()
    valid = (snap["colorWeights"] > 0) & (snap["weights"] > snap["colorWeights"] + 0.1)
    assert np.any(valid)
    np.testing.assert_allclose(snap["colors"][valid], 100, atol=1e-4)


def test_slam_reports_mapping_capacity_failure():
    camera = Camera(32, 24, 30, 30, 15.5, 11.5)
    slam = CPUSLAM(
        camera, tracker_options={"levels": 1, "iterations": (1,)}, map_options={"max_blocks": 1}
    )
    result = slam.process(render_depth(camera, np.eye(4)))
    assert result["success"]
    assert not result["mapping_success"]
    assert result["mapping"]["overflow"] > 0
