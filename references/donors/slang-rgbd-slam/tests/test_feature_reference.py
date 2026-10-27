import numpy as np
import pytest

from slang_rgbd_slam.feature_reference import optimize_reprojection
from slang_rgbd_slam.tracker import Camera


def test_reprojection_pose_recovers_motion_without_ground_truth_initialization():
    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    rng = np.random.default_rng(42)
    points = rng.uniform([-1, -0.7, 2], [1, 0.7, 4], (300, 3))
    angle = 0.025
    truth = np.eye(4)
    truth[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    truth[:3, 3] = [0.04, -0.02, 0.03]
    transformed = points @ truth[:3, :3].T + truth[:3, 3]
    pixels = transformed[:, :2] / transformed[:, 2, None] * [camera.fx, camera.fy] + [
        camera.cx,
        camera.cy,
    ]
    result = optimize_reprojection(points, pixels, camera)
    assert result.success
    assert result.correspondences == len(points)
    assert result.rmse < 1e-4
    np.testing.assert_allclose(result.pose, truth, atol=1e-5)


def test_reprojection_empty_and_validation():
    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    result = optimize_reprojection(np.empty((0, 3)), np.empty((0, 2)), camera)
    assert not result.success
    np.testing.assert_array_equal(result.pose, np.eye(4))
    with pytest.raises(ValueError):
        optimize_reprojection(np.zeros((10, 2)), np.zeros((10, 2)), camera)


def test_textured_translation_feature_matching_and_backprojection():
    from slang_rgbd_slam.feature_reference import CPUFeatures

    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    rng = np.random.default_rng(9)
    gray = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    shifted = np.zeros_like(rgb)
    shifted[2:, 3:] = rgb[:-2, :-3]
    frontend = CPUFeatures(camera, levels=2, max_features=256)
    depth = np.full((120, 160), 2, np.float32)
    a, b = frontend.extract(rgb, depth), frontend.extract(shifted, depth)
    matches = frontend.match(a, b)
    valid = matches != 0xFFFFFFFF
    assert valid.sum() > 30
    movement = b["positions"][matches[valid], :2] - a["positions"][valid, :2]
    np.testing.assert_allclose(movement, np.broadcast_to([3, 2], movement.shape), atol=1)
    points = a["landmarks"][:, :3]
    projected = points[:, :2] / points[:, 2, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    np.testing.assert_allclose(projected, a["positions"][:, :2], atol=1e-5)
    empty = frontend.extract(rgb, np.zeros_like(depth))
    assert empty["count"] == 0


def test_descriptor_ratio_rejects_duplicate_and_single_candidates():
    from slang_rgbd_slam.feature_reference import match_features

    source = {"descriptors": np.zeros((1, 8), np.uint32)}
    assert match_features(source, {"descriptors": np.zeros((2, 8), np.uint32)})[0] == 0xFFFFFFFF
    assert match_features(source, source)[0] == 0xFFFFFFFF
