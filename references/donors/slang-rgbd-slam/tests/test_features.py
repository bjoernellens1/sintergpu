"""Clean oriented-BRIEF feature contracts; not exact OpenCV ORB compatibility."""

import os

import numpy as np
import pytest

from slang_rgbd_slam.tracker import Camera


def test_pattern_and_spatial_budget():
    from slang_rgbd_slam.features import feature_brief_pattern, feature_grids

    pattern = feature_brief_pattern()
    assert pattern.shape == (256, 4)
    assert np.max(np.linalg.norm(pattern.reshape(-1, 2), axis=1)) <= 15
    np.testing.assert_array_equal(pattern, feature_brief_pattern())
    c = Camera(160, 120, 120.0, 120.0, 79.5, 59.5)
    grids = feature_grids(c, 1024, 4, 4)
    assert sum(x * y * q for x, y, q in grids) <= 1024
    assert all(x > 0 and y > 0 and q > 0 for x, y, q in grids)


@pytest.mark.skipif(
    os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="explicit GPU tests opt-in"
)
def test_gpu_textured_identity_and_translation():
    pytest.importorskip("slangpy")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.features import GPUFeatures

    runtime = VulkanRuntime()
    camera = Camera(160, 120, 120.0, 120.0, 79.5, 59.5)
    frontend = GPUFeatures(camera, runtime, max_features=256, levels=2, fast_threshold=12)
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8).astype(np.float32)
    depth = np.ones((120, 160), np.float32)
    a, b = frontend.create_set(), frontend.create_set()
    rgb_buffer, depth_buffer = runtime.buffer(rgb.ravel()), runtime.buffer(depth.ravel())
    encoder = runtime.device.create_command_encoder()
    frontend.record_extract(rgb_buffer, depth_buffer, a, encoder)
    frontend.record_extract(rgb_buffer, depth_buffer, b, encoder)
    matches = runtime.buffer(np.zeros(256, np.uint32))
    frontend.record_match(a, b, matches, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    first, second = frontend.snapshot(a), frontend.snapshot(b)
    assert first["count"] > 30
    assert first["count"] <= 256
    result = matches.to_numpy().view(np.uint32).ravel()
    valid = result[: first["count"]] != 0xFFFFFFFF
    assert valid.sum() >= first["count"] * 0.95
    np.testing.assert_allclose(
        first["positions"][valid, :3],
        second["positions"][result[: first["count"]][valid], :3],
        atol=1e-5,
    )
    np.testing.assert_allclose(first["landmarks"][:, 2], 1, atol=1e-6)
    assert np.all(result[first["count"] :] == 0xFFFFFFFF)

    from slang_rgbd_slam.feature_reference import CPUFeatures

    cpu = CPUFeatures(camera, max_features=256, levels=2, fast_threshold=12).extract(rgb, depth)
    gpu_order = np.lexsort(first["positions"][:, [1, 0, 3]].T)
    cpu_order = np.lexsort(cpu["positions"][:, [1, 0, 3]].T)
    assert first["count"] == cpu["count"]
    np.testing.assert_array_equal(first["positions"][gpu_order], cpu["positions"][cpu_order])
    np.testing.assert_allclose(
        first["landmarks"][gpu_order], cpu["landmarks"][cpu_order], atol=1e-6
    )
    # FMA grayscale rounding can flip a comparison of almost equal samples.
    difference = first["descriptors"][gpu_order] ^ cpu["descriptors"][cpu_order]
    hamming = np.unpackbits(difference.view(np.uint8), axis=1).sum(axis=1)
    assert hamming.max(initial=0) <= 1

    shifted = np.zeros_like(rgb)
    shifted[:, 3:] = rgb[:, :-3]
    rgb_buffer.copy_from_numpy(shifted.ravel())
    encoder = runtime.device.create_command_encoder()
    frontend.record_extract(rgb_buffer, depth_buffer, b, encoder)
    frontend.record_match(a, b, matches, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    shifted_features = frontend.snapshot(b)
    result = matches.to_numpy().view(np.uint32).ravel()[: first["count"]]
    valid = (result != 0xFFFFFFFF) & (first["positions"][:, 3] == 0)
    valid_indices = np.flatnonzero(valid)
    valid_indices = valid_indices[shifted_features["positions"][result[valid_indices], 3] == 0]
    assert len(valid_indices) > 10
    delta = (
        shifted_features["positions"][result[valid_indices], :2]
        - first["positions"][valid_indices, :2]
    )
    np.testing.assert_array_equal(delta, np.tile([3.0, 0.0], (len(delta), 1)))


@pytest.mark.skipif(
    os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="explicit GPU tests opt-in"
)
def test_hamming_ratio_and_mutual_check():
    pytest.importorskip("slangpy")
    from slang_rgbd_slam.features import GPUFeatures

    c = Camera(64, 64, 60.0, 60.0, 31.5, 31.5)
    f = GPUFeatures(c, max_features=16, levels=1)
    a, b = f.create_set(), f.create_set()
    rng = np.random.default_rng(7)
    words = rng.integers(0, 2**32, (4, 8), dtype=np.uint32)
    permutation = np.array([2, 0, 3, 1])
    for target, data in ((a, words), (b, words[permutation])):
        padded = np.zeros((16, 8), np.uint32)
        padded[:4] = data
        target["descriptors"].copy_from_numpy(padded.ravel())
        target["count"].copy_from_numpy(np.array([4], np.uint32))
    result = f._buffer(16)
    e = f.runtime.device.create_command_encoder()
    f.record_match(a, b, result, e)
    f.runtime.device.submit_command_buffer(e.finish())
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32)[:4], np.argsort(permutation))
    assert int(a["match_count"].to_numpy().view(np.uint32)[0]) == 4
    # Duplicate best descriptors make the ratio test reject the ambiguous match.
    padded = np.zeros((16, 8), np.uint32)
    padded[:4] = words[permutation]
    padded[1] = padded[0]
    b["descriptors"].copy_from_numpy(padded.ravel())
    e = f.runtime.device.create_command_encoder()
    f.record_match(a, b, result, e)
    f.runtime.device.submit_command_buffer(e.finish())
    assert result.to_numpy().view(np.uint32)[2] == 0xFFFFFFFF
