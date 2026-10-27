import os

import numpy as np
import pytest

from slang_rgbd_slam.synthetic import synthetic_sequence

pytestmark = pytest.mark.skipif(
    os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in"
)


def test_frame_to_model_pipeline_and_failed_frame():
    from slang_rgbd_slam.gpu_slam import GPUSLAM

    camera, frames = synthetic_sequence(4, 160, 120)
    slam = GPUSLAM(camera, map_options={"max_blocks": 2048})
    results = [slam.process(f.depth, f.rgb, timestamp=f.timestamp) for f in frames]
    assert all(r["success"] for r in results)
    before = slam.map.snapshot()
    pose = slam.world_pose.to_numpy().copy()
    failed = slam.process(np.zeros_like(frames[0].depth), timestamp=frames[-1].timestamp + 1)
    assert not failed["success"]
    np.testing.assert_array_equal(slam.world_pose.to_numpy(), pose)
    after = slam.map.snapshot()
    for name in ("keys", "distances", "weights", "colors"):
        np.testing.assert_array_equal(before[name], after[name])
    assert len(slam.trajectory) == 4
    assert slam.map.stats()["blocks"] > 0
