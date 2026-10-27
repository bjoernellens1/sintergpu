import os

import numpy as np
import pytest

from slang_rgbd_slam.synthetic import synthetic_sequence


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
@pytest.mark.parametrize(
    "group_size,reduction",
    [
        (32, "subgroup"),
        (64, "workgroup"),
        (128, "subgroup"),
        (256, "workgroup"),
        (256, "subgroup"),
    ],
)
def test_fused_icp_matches_existing_math_and_retains_no_residual_tensors(group_size, reduction):
    from slang_rgbd_slam.dense_tracker import FusedVulkanTracker
    from slang_rgbd_slam.tracker import VulkanTracker

    camera, frames = synthetic_sequence(2, 160, 120)
    baseline, fused = (
        VulkanTracker(camera),
        FusedVulkanTracker(camera, photo_weight=0, group_size=group_size, reduction=reduction),
    )
    baseline.set_reference(frames[0].depth)
    fused.set_reference(frames[0].depth)
    a, b = baseline.track(frames[1].depth), fused.track(frames[1].depth)
    assert a.success and b.success
    assert a.correspondences == b.correspondences
    np.testing.assert_allclose(a.pose, b.pose, atol=2e-5, rtol=0)
    assert all(not level["problem"].buffers for level in fused.levels)


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_fused_icp_photo_model_tracking_and_raycast_intensity():
    from slang_rgbd_slam.dense_tracker import FusedVulkanTracker
    from slang_rgbd_slam.gpu_slam import GPUSLAM

    camera, frames = synthetic_sequence(4, 160, 120)
    slam = GPUSLAM(camera, tracker_factory=FusedVulkanTracker, map_options={"max_blocks": 2048})
    results = [slam.process(f.depth, f.rgb, timestamp=f.timestamp) for f in frames]
    assert all(r["success"] and r["mapping_success"] for r in results)
    intensity = (
        slam.tracker.levels[0]["maps"]["reference"]["intensity"].to_numpy().view(np.float32).ravel()
    )
    assert np.isfinite(intensity).all() and np.max(intensity) <= 1
    assert np.count_nonzero(intensity > 0) > intensity.size * 0.8
    assert (
        np.linalg.norm(
            results[-1]["world_from_camera"][:3, 3] - frames[-1].world_from_camera[:3, 3]
        )
        < 0.004
    )
    old = slam.world_pose.to_numpy().copy()
    failed = slam.process(np.zeros_like(frames[0].depth), timestamp=1)
    assert not failed["success"]
    np.testing.assert_array_equal(slam.world_pose.to_numpy(), old)
