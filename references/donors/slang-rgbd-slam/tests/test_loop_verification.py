import numpy as np
import pytest

from slang_rgbd_slam.loop_verification import ransac_reference, rigid_from_three


def test_three_point_rigid_alignment_and_degeneracy():
    source = np.array([[0.0, 0, 2], [0.4, 0, 2], [0, 0.3, 2]])
    target = source + [0.04, -0.03, 0.02]
    pose = rigid_from_three(source, target)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], [0.04, -0.03, 0.02], atol=1e-12)
    with pytest.raises(ValueError):
        rigid_from_three(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]]), target)


def test_textured_rgbd_views_ransac_robust_to_descriptor_outliers():
    from slang_rgbd_slam.feature_reference import CPUFeatures
    from slang_rgbd_slam.tracker import Camera

    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    rng = np.random.default_rng(9)
    rgb = np.repeat(rng.integers(0, 256, (120, 160), dtype=np.uint8)[:, :, None], 3, axis=2)
    shifted = np.zeros_like(rgb)
    shifted[2:, 3:] = rgb[:-2, :-3]
    features = CPUFeatures(camera, levels=2, max_features=256)
    depth = np.full((120, 160), 2, np.float32)
    source, target = features.extract(shifted, depth), features.extract(rgb, depth)
    matches = features.match(source, target)
    valid = np.flatnonzero(matches != 0xFFFFFFFF)
    matches[valid[::4]] = rng.integers(0, target["count"], len(valid[::4]), dtype=np.uint32)
    result = ransac_reference(source, target, matches)
    assert result["success"]
    assert result["inliers"] > 40
    np.testing.assert_allclose(result["target_from_source"][:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(
        result["target_from_source"][:3, 3], [-6 / 130, -4 / 130, 0], atol=1e-5
    )


def test_verifier_record_never_reads_features_or_solves_on_cpu():
    import inspect
    from pathlib import Path

    from slang_rgbd_slam.loop_verification import RGBDLoopVerifier

    record = inspect.getsource(RGBDLoopVerifier.record)
    assert "to_numpy" not in record
    assert "np.linalg" not in record
    assert "self.refiner.record" in record
    shader = (
        Path(__file__)
        .parents[1]
        .joinpath("src/slang_rgbd_slam/shaders/loop_verification.slang")
        .read_text()
    )
    assert "index=lane;index<n;index+=64" in shader
    assert "GroupMemoryBarrierWithGroupSync" in shader
    assert "caspar_transform(" in shader
    assert "verificationStats[1]!=0||refinementStats[1]!=0" in shader


def test_gpu_textured_rgbd_candidate_with_outliers_and_failed_pose_immutability():
    import os

    if os.environ.get("SLANG_RGBD_GPU_TESTS") != "1":
        pytest.skip("opt in after independent GPU review")
    pytest.importorskip("slangpy")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.features import GPUFeatures
    from slang_rgbd_slam.loop_verification import RGBDLoopVerifier
    from slang_rgbd_slam.tracker import Camera

    camera = Camera(160, 120, 130, 130, 79.5, 59.5)
    runtime = VulkanRuntime()
    frontend = GPUFeatures(camera, runtime, max_features=256, levels=1)
    verifier = RGBDLoopVerifier(camera, runtime, max_features=256)
    problem = verifier.prepare()
    source, target = frontend.create_set(), frontend.create_set()
    rng = np.random.default_rng(9)
    rgb = np.repeat(rng.integers(0, 256, (120, 160), dtype=np.uint8)[:, :, None], 3, axis=2)
    shifted = np.zeros_like(rgb)
    shifted[2:, 3:] = rgb[:-2, :-3]
    depth = runtime.buffer(np.full(160 * 120, 2, np.float32))
    original = runtime.buffer(rgb.astype(np.float32).ravel())
    current = runtime.buffer(shifted.astype(np.float32).ravel())
    matches = frontend._buffer(256)
    original_matches = frontend._buffer(256)
    target_world_pose = runtime.buffer(np.eye(4, dtype=np.float32).ravel())
    inject = runtime.compile(
        """
    RWStructuredBuffer<uint> matches;StructuredBuffer<uint> count;
        StructuredBuffer<float4> targetPoints;
    [shader("compute")][numthreads(64,1,1)]
    void inject(uint3 tid:SV_DispatchThreadID) {
      uint i=tid.x,n=min(count[0],256u);
      if(i>=256 || i%4!=0 || matches[i]>=n)return;
      uint correct=matches[i];float3 original=targetPoints[correct].xyz;
      // Atomic compaction changes ordering: an index offset can select an
      // actual geometric inlier. Require a genuinely distant replacement.
      for(uint offset=1;offset<=n;offset++) {
        uint candidate=(correct+offset)%n;
        float3 displacement=targetPoints[candidate].xyz-original;
        if(targetPoints[candidate].w>0 && dot(displacement,displacement)>.25*.25) {
          matches[i]=candidate;return;
        }
      }
    }""",
        entry="inject",
    )
    encoder = runtime.device.create_command_encoder()
    frontend.record_extract(current, depth, source, encoder)
    frontend.record_extract(original, depth, target, encoder)
    frontend.record_match(source, target, matches, encoder)
    encoder.global_barrier()
    encoder.copy_buffer(original_matches, 0, matches, 0, 256 * 4)
    encoder.global_barrier()
    inject.dispatch(
        thread_count=[256, 1, 1],
        vars=dict(
            matches=matches,
            count=target["count"],
            targetPoints=target["landmarks"],
        ),
        command_encoder=encoder,
    )
    encoder.global_barrier()
    verifier.record(problem, source, target, matches, target_world_pose, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    before = original_matches.to_numpy().view(np.uint32).ravel()
    after = matches.to_numpy().view(np.uint32).ravel()
    points = target["landmarks"].to_numpy().view(np.float32).reshape(-1, 4)
    changed = np.flatnonzero((before != after) & (before != 0xFFFFFFFF))
    assert len(changed) >= 20  # Retain a substantial genuine descriptor-outlier case.
    assert np.all(changed % 4 == 0)
    displacement = np.linalg.norm(points[before[changed], :3] - points[after[changed], :3], axis=1)
    assert np.all(displacement > 0.25)
    stats = problem.stats.to_numpy().view(np.uint32).ravel()
    assert stats[1] == 0
    assert stats[0] > 40
    pose = problem.pose.to_numpy().view(np.float32).reshape(4, 4).copy()
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-4)
    np.testing.assert_allclose(pose[:3, 3], [6 / 130, 4 / 130, 0], atol=1e-4)
    # A subsequent empty match set never commits a pose, even if internal
    # refinement/status buffers held values from the preceding valid candidate.
    matches.copy_from_numpy(np.full(256, 0xFFFFFFFF, np.uint32))
    encoder = runtime.device.create_command_encoder()
    verifier.record(problem, source, target, matches, target_world_pose, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    assert problem.stats.to_numpy().view(np.uint32).ravel()[1] != 0
    np.testing.assert_array_equal(problem.pose.to_numpy().view(np.float32).reshape(4, 4), pose)
    # The archive's global coordinate frame may be rotated and translated.
    from slang_rgbd_slam.cpu import _exp

    world_from_target = _exp(np.array([0.01, 0.03, -0.02, 0.2, -0.1, 0.1]))
    target_world_pose.copy_from_numpy(world_from_target.astype(np.float32).ravel())
    encoder = runtime.device.create_command_encoder()
    frontend.record_match(source, target, matches, encoder)
    verifier.record(problem, source, target, matches, target_world_pose, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    assert problem.stats.to_numpy().view(np.uint32).ravel()[1] == 0
    expected = pose @ np.linalg.inv(world_from_target)
    np.testing.assert_allclose(
        problem.pose.to_numpy().view(np.float32).reshape(4, 4), expected, atol=1e-4
    )
