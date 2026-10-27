import json
from types import SimpleNamespace

import numpy as np
import pytest

from slang_rgbd_slam.comparison import compare_backends
from slang_rgbd_slam.dataset import Frame
from slang_rgbd_slam.tracker import Camera


class FakeSLAM:
    constructions = []

    def __init__(self, camera, *, tracker_options=None, map_options=None):
        self.constructions.append((camera, tracker_options, map_options))
        self.trajectory = []
        self.map = SimpleNamespace(
            raycast=lambda pose: np.ones((camera.height, camera.width), np.float32),
            stats=lambda: {"blocks": 1, "overflow": 0},
        )

    def process(self, depth, rgb, *, timestamp):
        pose = np.eye(4)
        pose[0, 3] = timestamp * 0.01
        self.trajectory.append((timestamp, pose))
        return {"success": True, "world_from_camera": pose, "rmse": 0.001}


def test_comparison_runs_same_config_and_serializes(tmp_path, monkeypatch):
    import slang_rgbd_slam.comparison as module

    monkeypatch.setattr(module, "_gpu_class", lambda: FakeSLAM)
    monkeypatch.setattr(module, "CPUSLAM", FakeSLAM)
    camera = Camera(16, 16, 15, 15, 7.5, 7.5)
    frames = []
    for timestamp in range(3):
        pose = np.eye(4)
        pose[0, 3] = timestamp * 0.01
        frames.append(
            Frame(float(timestamp), np.ones((16, 16), np.float32), world_from_camera=pose)
        )
    report = compare_backends(
        frames,
        camera,
        tmp_path,
        tracker_options={"levels": 1, "iterations": (2,)},
        map_options={"max_blocks": 128},
    )
    assert report["cpu"]["tracked_frames"] == report["gpu"]["tracked_frames"] == 3
    assert report["cpu"]["ate"]["rmse_m"] < 1e-10
    assert report["map_comparison"]["depth_rmse_m"] == 0
    assert report["map_comparison"]["paired_pixels"] == 3 * 16 * 16
    assert report["cpu"]["wall_ms"]["median"] >= 0
    assert FakeSLAM.constructions[-2:] == [FakeSLAM.constructions[-1]] * 2
    assert json.loads((tmp_path / "comparison.json").read_text()) == report


def test_comparison_missing_gpu_raises_without_cpu_fallback(tmp_path, monkeypatch):
    import slang_rgbd_slam.comparison as module

    def fail():
        raise ImportError("GPU unavailable")

    monkeypatch.setattr(module, "_gpu_class", fail)
    with pytest.raises(ImportError, match="GPU unavailable"):
        compare_backends(
            [Frame(0.0, np.ones((16, 16), np.float32))], Camera(16, 16, 15, 15, 7.5, 7.5), tmp_path
        )


def test_empty_comparison_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        compare_backends([], Camera(16, 16, 15, 15, 7.5, 7.5), tmp_path)


def test_ground_truth_map_evaluation_normalizes_nonidentity_origin(tmp_path, monkeypatch):
    import slang_rgbd_slam.comparison as module

    class CheckedSLAM(FakeSLAM):
        def __init__(self, camera, **options):
            super().__init__(camera, **options)
            self.map.raycast = self.raycast

        def raycast(self, pose):
            # Reconstruction is in the first camera's frame, not the dataset's
            # arbitrary translated/rotated global ground-truth coordinate frame.
            assert abs(pose[0, 3]) < 0.03
            np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)
            return np.ones((16, 16), np.float32)

    class Sequence(list):
        ground_truth = []

    monkeypatch.setattr(module, "_gpu_class", lambda: CheckedSLAM)
    monkeypatch.setattr(module, "CPUSLAM", CheckedSLAM)
    camera = Camera(16, 16, 15, 15, 7.5, 7.5)
    origin = np.eye(4)
    angle = 0.7
    origin[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    origin[:3, 3] = [100, -30, 4]
    frames = Sequence()
    for timestamp in range(3):
        local = np.eye(4)
        local[0, 3] = timestamp * 0.01
        # Separate TUM-style GT timestamps, with no pose attached to Frame.
        frames.append(Frame(float(timestamp), np.ones((16, 16), np.float32)))
        frames.ground_truth.append((timestamp + 0.005, origin @ local))
    report = compare_backends(frames, camera, tmp_path)
    assert report["map_comparison"]["evaluation_pose_source"] == "ground_truth_in_initial_camera"
    assert report["map_comparison"]["evaluation_poses"] == 3
    assert report["map_comparison"]["cpu_measured_depth_rmse_m"] == 0


def test_map_quality_sampling_does_not_change_timed_frame_count(tmp_path, monkeypatch):
    import slang_rgbd_slam.comparison as module

    monkeypatch.setattr(module, "_gpu_class", lambda: FakeSLAM)
    monkeypatch.setattr(module, "CPUSLAM", FakeSLAM)
    frames = [
        Frame(float(i), np.ones((16, 16), np.float32), world_from_camera=np.eye(4))
        for i in range(5)
    ]
    report = compare_backends(
        frames, Camera(16, 16, 15, 15, 7.5, 7.5), tmp_path, max_evaluation_poses=2
    )
    assert report["cpu"]["wall_ms"]["count"] == report["gpu"]["wall_ms"]["count"] == 5
    assert report["map_comparison"]["evaluation_timestamps"] == [0.0, 4.0]
    assert report["map_comparison"]["paired_pixels"] == 2 * 16 * 16
    with pytest.raises(ValueError, match="max_evaluation_poses"):
        compare_backends(frames, Camera(16, 16, 15, 15, 7.5, 7.5), tmp_path, max_evaluation_poses=0)
