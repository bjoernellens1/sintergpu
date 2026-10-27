"""Numerical and capacity contracts for sparse GPU TSDF mapping."""

import numpy as np
import pytest

from slang_rgbd_slam.tracker import Camera


@pytest.fixture
def mapper():
    pytest.importorskip("slangpy")
    from slang_rgbd_slam.gpu_mapping import GPUTSDF

    return GPUTSDF(Camera(32, 24, 40.0, 40.0, 15.5, 11.5), max_blocks=128)


def test_plane_fusion_and_raycast(mapper):
    depth = np.ones((24, 32), np.float32)
    pose = np.eye(4, dtype=np.float32)
    mapper.integrate(depth, None, pose)
    first = mapper.snapshot()
    mapper.integrate(depth, None, pose)
    second = mapper.snapshot()
    assert mapper.stats()["overflow"] == 0
    np.testing.assert_allclose(first["distances"], second["distances"], atol=1e-6)
    assert np.max(second["weights"]) > np.max(first["weights"])
    hit = mapper.raycast(pose)
    assert np.count_nonzero(hit) > hit.size // 2
    np.testing.assert_allclose(hit[hit > 0], 1.0, atol=0.015)


def test_failed_frame_and_capacity(mapper):
    runtime = mapper.runtime
    depth = runtime.buffer(np.ones(32 * 24, np.float32))
    rgb = runtime.buffer(np.ones(32 * 24 * 3, np.float32))
    pose = runtime.buffer(np.eye(4, dtype=np.float32).ravel())
    failed = runtime.buffer(np.array([0, 1], np.uint32))
    encoder = runtime.device.create_command_encoder()
    mapper.record_integrate(depth, rgb, pose, encoder, tracking_status=failed, frame_index=1)
    runtime.device.submit_command_buffer(encoder.finish())
    assert mapper.stats()["blocks"] == 0
    mapper.integrate(np.ones((24, 32), np.float32), None, np.eye(4, dtype=np.float32))
    assert mapper.stats()["blocks"] > 0


def test_negative_blocks_and_export(mapper, tmp_path):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = -0.5
    mapper.integrate(np.ones((24, 32), np.float32), None, pose)
    assert np.any(mapper.snapshot()["keys"][:, 0] < 0)
    path = tmp_path / "surface.ply"
    mapper.export_ply(path)
    text = path.read_text()
    assert "element vertex 0\n" not in text


def test_capacity_and_domain_errors():
    pytest.importorskip("slangpy")
    from slang_rgbd_slam.gpu_mapping import GPUTSDF

    c = Camera(32, 24, 40.0, 40.0, 15.5, 11.5)
    m = GPUTSDF(c, max_blocks=1)
    d = np.ones((24, 32), np.float32)
    m.integrate(d, None, np.eye(4, dtype=np.float32))
    assert m.stats()["blocks"] == 1
    assert m.stats()["overflow"] > 0
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 1000.0
    m.integrate(d, None, pose)
    assert m.stats()["domain_errors"] > 0


def test_sphere_depth_zero_crossing(mapper):
    c = mapper.camera
    y, x = np.indices((c.height, c.width), dtype=np.float32)
    dx, dy = (x - c.cx) / c.fx, (y - c.cy) / c.fy
    a = dx * dx + dy * dy + 1
    discriminant = 1.5**2 - a * (1.5**2 - 0.6**2)
    depth = np.where(discriminant >= 0, (1.5 - np.sqrt(np.maximum(discriminant, 0))) / a, 0).astype(
        np.float32
    )
    mapper.integrate(depth, None, np.eye(4, dtype=np.float32))
    hit = mapper.raycast(np.eye(4, dtype=np.float32))
    valid = (hit > 0) & (depth > 0)
    assert np.count_nonzero(valid) > hit.size // 3
    np.testing.assert_allclose(hit[valid], depth[valid], atol=0.03)


def test_color_confidence_is_independent_of_free_space_weight(mapper):
    pose = np.eye(4, dtype=np.float32)
    # First observation gives the z=.915 voxel free-space weight but no color.
    mapper.integrate(np.full((24, 32), 1.05, np.float32), np.full((24, 32, 3), 200, np.uint8), pose)
    mapper.integrate(np.full((24, 32), 0.93, np.float32), np.full((24, 32, 3), 100, np.uint8), pose)
    snap = mapper.snapshot()
    valid = (snap["colorWeights"] > 0) & (snap["weights"] > snap["colorWeights"] + 0.1)
    assert np.any(valid)
    np.testing.assert_allclose(snap["colors"][valid], 100, atol=1e-4)


def test_colliding_hash_keys_both_allocate(mapper):
    mask = mapper.hash_capacity - 1

    def hashed(x):
        key = (x + 512) | (512 << 10) | ((4 + 512) << 20)
        key ^= key >> 16
        key = (key * 0x7FEB352D) & 0xFFFFFFFF
        key ^= key >> 15
        key = (key * 0x846CA68B) & 0xFFFFFFFF
        return (key ^ (key >> 16)) & mask

    buckets = {}
    pair = None
    for x in range(-100, 101):
        h = hashed(x)
        if h in buckets and abs(x - buckets[h]) > 5:
            pair = (buckets[h], x)
            break
        buckets[h] = x
    assert pair is not None
    for x in pair:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = x * mapper.voxel_size * 8 + 0.12
        pose[1, 3] = 0.12
        mapper.integrate(np.ones((24, 32), np.float32), None, pose)
    keys = set(map(tuple, mapper.snapshot()["keys"]))
    assert all((x, 0, 4) in keys for x in pair)
    assert mapper.stats()["probe_exhaustion"] == 0
