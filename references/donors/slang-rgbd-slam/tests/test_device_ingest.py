"""External device image ingestion and guarded map-resource contracts."""

import inspect
import os

import numpy as np
import pytest

from slang_rgbd_slam.gpu_slam import GPUSLAM
from slang_rgbd_slam.synthetic import synthetic_sequence


def test_device_ingestion_has_no_image_upload_or_readback():
    source = inspect.getsource(GPUSLAM.process_device)
    assert "copy_from_numpy" not in source
    assert "to_numpy" not in source
    assert "_dispatch_frame(depth_buffer, rgb_buffer" in source


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_external_device_images_match_host_and_preserve_owned_upload_buffers():
    camera, frames = synthetic_sequence(3, 160, 120)
    host = GPUSLAM(camera, map_options={"max_blocks": 1024})
    device = GPUSLAM(camera, map_options={"max_blocks": 1024})
    external_depth = device.runtime.buffer(frames[0].depth.ravel())
    external_rgb = device.runtime.buffer(frames[0].rgb.astype(np.float32).ravel())
    owned_depth, owned_rgb = device._owned_depth, device._owned_rgb
    for index, frame in enumerate(frames):
        expected = host.process(frame.depth, frame.rgb, timestamp=frame.timestamp)
        if index == 1:
            actual = device.process(frame.depth, frame.rgb, timestamp=frame.timestamp)
            assert device.tracker.levels[0]["maps"]["source"]["depth"] is owned_depth
            assert device.rgb is owned_rgb
        else:
            external_depth.copy_from_numpy(frame.depth.ravel())
            external_rgb.copy_from_numpy(frame.rgb.astype(np.float32).ravel())
            depth_before, rgb_before = external_depth.to_numpy(), external_rgb.to_numpy()
            actual = device.process_device(external_depth, external_rgb, timestamp=frame.timestamp)
            np.testing.assert_array_equal(external_depth.to_numpy(), depth_before)
            np.testing.assert_array_equal(external_rgb.to_numpy(), rgb_before)
        assert expected["success"] and actual["success"]
        assert expected["correspondences"] == actual["correspondences"]
        np.testing.assert_allclose(actual["world_from_camera"], expected["world_from_camera"], atol=1e-6)
    assert device._owned_depth is owned_depth and device._owned_rgb is owned_rgb
    actual_map, expected_map = device.map.snapshot(), host.map.snapshot()
    # Atomic block allocation order is independent of map coordinates.
    actual_order = np.lexsort(actual_map["keys"].T)
    expected_order = np.lexsort(expected_map["keys"].T)
    np.testing.assert_array_equal(
        actual_map["keys"][actual_order], expected_map["keys"][expected_order]
    )
    for name in ("distances", "weights", "colors"):
        np.testing.assert_allclose(
            actual_map[name][actual_order], expected_map[name][expected_order], atol=1e-6, rtol=1e-6
        )
    with pytest.raises(ValueError, match="sizes"):
        device.process_device(external_rgb, external_depth, timestamp=1)
    with pytest.raises(ValueError, match="Timestamps"):
        device.process_device(external_depth, external_rgb, timestamp=frames[-1].timestamp)


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_tsdf_reset_guard_and_missing_color_intensity():
    from slang_rgbd_slam.gpu_mapping import GPUTSDF

    camera, frames = synthetic_sequence(1, 80, 60)
    mapping = GPUTSDF(camera, max_blocks=1024)
    mapping.integrate(frames[0].depth, frames[0].rgb, np.eye(4))
    before = mapping.snapshot()
    failed_status = mapping.runtime.buffer(np.array([0, 1], np.uint32))
    encoder = mapping.runtime.device.create_command_encoder()
    mapping.record_reset(encoder, tracking_status=failed_status)
    mapping.runtime.device.submit_command_buffer(encoder.finish())
    after = mapping.snapshot()
    for name in ("keys", "distances", "weights", "colors"):
        np.testing.assert_array_equal(before[name], after[name])
    mapping.buffers["colorWeights"].copy_from_numpy(np.zeros(mapping.max_blocks * 512, np.float32))
    depth = mapping.raycast(np.eye(4))
    assert np.count_nonzero(depth) > depth.size // 2
    intensity = mapping.intensity_output.to_numpy().view(np.float32)
    assert np.all(intensity == -1)
    encoder = mapping.runtime.device.create_command_encoder()
    mapping.record_reset(encoder)
    mapping.runtime.device.submit_command_buffer(encoder.finish())
    assert mapping.stats()["blocks"] == 0
