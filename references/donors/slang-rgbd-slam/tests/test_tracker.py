import os

import numpy as np
import pytest

from slang_rgbd_slam.tracker import Camera, VulkanTracker


def test_camera_validation():
    with pytest.raises(ValueError):
        Camera(32, 24, 0, 20, 15, 11)


@pytest.mark.skipif(
    os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="opt in to physical GPU tests"
)
def test_curved_surface_identity_and_empty_failure():
    camera = Camera(96, 72, 85, 85, 47.5, 35.5)
    y, x = np.mgrid[:72, :96]
    depth = (1.5 + 0.15 * np.sin(x / 17) + 0.1 * np.cos(y / 13)).astype(np.float32)
    tracker = VulkanTracker(camera, levels=2, iterations=(3, 3))
    tracker.set_reference(depth)
    result = tracker.track(depth)
    assert result.success
    assert result.correspondences > 1000
    np.testing.assert_allclose(result.pose, np.eye(4), atol=1e-4)
    failed = tracker.track(np.zeros_like(depth))
    assert not failed.success
    np.testing.assert_array_equal(failed.pose, np.eye(4, dtype=np.float32))


@pytest.mark.skipif(
    os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="opt in to physical GPU tests"
)
def test_recovers_projective_six_dof_motion():
    from slang_rgbd_slam.synthetic import render_depth

    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    pose = np.eye(4)
    angle = 0.012
    pose[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    pose[:3, 3] = [0.015, -0.009, 0.012]
    tracker = VulkanTracker(camera)
    tracker.set_reference(render_depth(camera, np.eye(4)))
    result = tracker.track(render_depth(camera, pose))
    assert result.success
    assert result.correspondences > 1000
    np.testing.assert_allclose(result.pose[:3, 3], pose[:3, 3], atol=0.004)
    np.testing.assert_allclose(result.pose[:3, :3], pose[:3, :3], atol=0.004)
