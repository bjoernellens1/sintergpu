from types import SimpleNamespace

import numpy as np
import pytest

from slang_rgbd_slam.slam import LegacyRGBDSLAM as RGBDSLAM
from slang_rgbd_slam.slam import run_sequence
from slang_rgbd_slam.synthetic import synthetic_sequence


class FakeTracker:
    def __init__(self):
        self.references = 0
        self.calls = 0

    def set_reference(self, depth):
        self.references += 1

    def track(self, depth):
        self.calls += 1
        p = np.eye(4)
        p[0, 3] = 0.01
        return SimpleNamespace(
            pose=p, success=self.calls != 2, correspondences=100, rmse=0.001, elapsed_ms=1
        )


def test_keyframe_pose_composition_and_loss():
    c, frames = synthetic_sequence(5, width=32, height=24)
    tracker = FakeTracker()
    s = RGBDSLAM(c, tracker=tracker, keyframe_interval=1)
    r = [s.process(f.depth, f.rgb, timestamp=f.timestamp) for f in frames]
    assert r[2]["status"] == "lost"
    assert r[2]["world_from_camera"] is None
    assert s.keyframes == 4 and tracker.references == 4
    assert len(s.trajectory) == 4
    assert abs(s.trajectory[-1][1][0, 3] - 0.03) < 1e-8


def test_sequence_artifacts(tmp_path):
    c, frames = synthetic_sequence(2, width=32, height=24)
    report = run_sequence(frames, c, tmp_path, backend="legacy", tracker=FakeTracker())
    assert report["frames"] == 2 and report["ate"]["count"] == 2
    assert "CPU" in report["mapping_backend"]
    assert (tmp_path / "map.ply").exists() and (tmp_path / "trajectory.txt").exists()


def test_initial_empty_depth_is_lost():
    c, _ = synthetic_sequence(1, width=32, height=24)
    tracker = FakeTracker()
    s = RGBDSLAM(c, tracker=tracker)
    assert not s.process(np.zeros((24, 32)), timestamp=0)["success"]
    assert tracker.references == 0 and s.keyframes == 0


def test_bad_input_and_unconstrained_initial():
    import pytest

    c, _ = synthetic_sequence(1, width=32, height=24)
    s = RGBDSLAM(c, tracker=FakeTracker())
    with pytest.raises(ValueError):
        s.process(np.full((24, 32), np.nan), timestamp=0)
    with pytest.raises(ValueError):
        s.process(np.full((24, 32), -1), timestamp=0)
    assert not s.process(np.ones((24, 32)), timestamp=0)["success"]
    with pytest.raises(ValueError):
        s.process(np.ones((24, 32)), timestamp=0)


@pytest.mark.skipif(
    __import__("os").environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in"
)
def test_real_vulkan_sequence_tracks_keyframes_and_exports(tmp_path):
    from slang_rgbd_slam.slam import run_sequence
    from slang_rgbd_slam.synthetic import synthetic_sequence

    camera, frames = synthetic_sequence(frames=12)
    report = run_sequence(frames, camera, tmp_path, engine="baseline")
    assert report["tracked_frames"] == 12
    assert report["lost_frames"] == 0
    assert report["keyframes"] == 12
    assert report["voxels"] > 1000
    assert report["ate"]["rmse_m"] < 0.003
    assert len((tmp_path / "trajectory.txt").read_text().splitlines()) == 12
    assert (tmp_path / "map.ply").read_text().startswith("ply\n")
