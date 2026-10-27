import numpy as np
import pytest

from slang_rgbd_slam.dense_ba import schur_step


def test_schur_elimination_matches_full_coupled_pose_depth_solve():
    rng = np.random.default_rng(17)
    jpose = rng.normal(size=(11, 4, 12))
    jdepth = rng.normal(size=(11, 4))
    residual = rng.normal(size=(11, 4))
    prior = rng.normal(size=11)
    weight, damping = 3.0, 1e-3
    dp, dd = schur_step(jpose, jdepth, residual, prior, depth_weight=weight, damping=damping)
    design = np.zeros((11 * 5, 12 + 11))
    values = np.zeros(11 * 5)
    for pixel in range(11):
        design[pixel * 5 : pixel * 5 + 4, :12] = jpose[pixel]
        design[pixel * 5 : pixel * 5 + 4, 12 + pixel] = jdepth[pixel]
        design[pixel * 5 + 4, 12 + pixel] = weight
        values[pixel * 5 : pixel * 5 + 4] = residual[pixel]
        values[pixel * 5 + 4] = weight * prior[pixel]
    full = np.linalg.solve(design.T @ design + damping * np.eye(23), -design.T @ values)
    np.testing.assert_allclose(dp, full[:12], atol=1e-10)
    np.testing.assert_allclose(dd, full[12:], atol=1e-10)


def test_schur_empty_or_invalid_parameters():
    with pytest.raises(ValueError):
        schur_step(np.zeros((2, 3, 6)), np.zeros((2, 4)), np.zeros((2, 3)), np.zeros(2))


def test_coupled_world_pose_and_inverse_depth_factor_central_differences():
    from slang_rgbd_slam.cpu import _exp
    from slang_rgbd_slam.dense_ba import dense_factor_reference
    from slang_rgbd_slam.tracker import Camera

    source = _exp(np.array([0.02, -0.03, 0.01, 0.1, -0.05, 0.03]))
    target = _exp(np.array([-0.01, 0.02, 0.025, -0.06, 0.03, -0.02]))
    ray = np.array([0.2, -0.1, 1.0])
    rho = 0.6
    camera = Camera(80, 60, 70, 70, 39.5, 29.5)
    world = source[:3, :3] @ (ray / rho) + source[:3, 3]
    point = target[:3, :3].T @ (world - target[:3, 3])
    pixel = point[:2] / point[2] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    q, n = point + [0.01, -0.02, 0.03], np.array([0.2, -0.3, 0.93])
    gradient = np.array([1.5, -2.3])

    def factor(s, t, depth):
        return dense_factor_reference(
            s, t, ray, depth, q, n, 100, 103, gradient, camera, linearization_pixel=pixel
        )

    residual, jpose, jdepth = factor(source, target, rho)
    numerical = np.empty((2, 12))
    epsilon = 1e-6
    for coordinate in range(12):
        delta = np.zeros(6)
        delta[coordinate % 6] = epsilon
        if coordinate < 6:
            plus, minus = (
                factor(_exp(delta) @ source, target, rho),
                factor(_exp(-delta) @ source, target, rho),
            )
        else:
            plus, minus = (
                factor(source, _exp(delta) @ target, rho),
                factor(source, _exp(-delta) @ target, rho),
            )
        numerical[:, coordinate] = (plus[0] - minus[0]) / (2 * epsilon)
    depth_numerical = (
        factor(source, target, rho + epsilon)[0] - factor(source, target, rho - epsilon)[0]
    ) / (2 * epsilon)
    np.testing.assert_allclose(jpose, numerical, atol=1e-6)
    np.testing.assert_allclose(jdepth, depth_numerical, atol=1e-6)
    assert residual.shape == (2,)


def test_dense_ba_record_has_no_cpu_solver_or_readback():
    import inspect
    from pathlib import Path

    from slang_rgbd_slam.dense_ba import DenseBA

    record = inspect.getsource(DenseBA.record)
    assert "to_numpy" not in record
    assert "np.linalg" not in record
    shader = (
        Path(__file__).parents[1].joinpath("src/slang_rgbd_slam/shaders/dense_ba.slang").read_text()
    )
    assert "float J[36]" in shader and "crossTerms[18]" in shader
    assert "caspar_dense_factor(" in shader
    assert "caspar_se3_coefficients(" in shader
    assert "caspar_twist_point(" in shader
    assert "value-=crossTerms[row]*crossTerms[column]/depthH" in shader
    assert "frame==0" in shader
    assert "self.record_evaluate" in record
    assert "rollbackDense" in record
    assert "final<=initial+abs(initial)*1e-4+1e-7" in shader


def test_gpu_dense_ba_empty_window_rejects_and_keeps_poses():
    import os

    if os.environ.get("SLANG_RGBD_GPU_TESTS") != "1":
        pytest.skip("opt in to physical GPU tests after independent review")
    pytest.importorskip("slangpy")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.dense_ba import DenseBA
    from slang_rgbd_slam.tracker import Camera

    ba = DenseBA(VulkanRuntime(), Camera(64, 48, 55, 55, 31.5, 23.5), window=2)
    problem = ba.prepare()
    ba.solve(problem, iterations=2)
    assert int(problem.buffers["status"].to_numpy().view(np.uint32).ravel()[0]) != 0
    np.testing.assert_array_equal(
        problem.poses.to_numpy().view(np.float32).reshape(2, 4, 4),
        np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
    )


@pytest.mark.parametrize("window", [2, 4])
def test_gpu_dense_ba_joint_pose_and_depth_reduces_residual(window):
    import os

    if os.environ.get("SLANG_RGBD_GPU_TESTS") != "1":
        pytest.skip("opt in to physical GPU tests after independent review")
    pytest.importorskip("slangpy")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.dense_ba import DenseBA
    from slang_rgbd_slam.synthetic import render_depth
    from slang_rgbd_slam.tracker import Camera

    camera = Camera(80, 60, 70, 70, 39.5, 29.5)
    truth = np.eye(4)
    truth[:3, 3] = [0.012, -0.006, 0.01]
    depths, colors = [], []
    yy, xx = np.indices((60, 80), dtype=np.float32)
    poses_truth = []
    for frame in range(window):
        pose = np.eye(4)
        pose[:3, 3] = truth[:3, 3] * frame
        poses_truth.append(pose)
        depth = render_depth(camera, pose)
        points = np.stack(
            ((xx - camera.cx) * depth / camera.fx, (yy - camera.cy) * depth / camera.fy, depth),
            axis=-1,
        )
        world = points @ pose[:3, :3].T + pose[:3, 3]
        value = 120 + 45 * np.sin(world[:, :, 0] * 9) + 40 * np.cos(world[:, :, 1] * 7)
        depths.append(depth)
        colors.append(np.repeat(value[:, :, None], 3, axis=2).astype(np.float32))
    ba = DenseBA(VulkanRuntime(), camera, window=window)
    # Both camera estimates start at identity. Ground truth only renders/evaluates
    # the test sequence and is never supplied as an optimizer initialization.
    problem = ba.prepare(np.stack(depths), np.stack(colors))
    ba.solve(problem, iterations=1)
    initial_cost = float(problem.buffers["costs"].to_numpy().view(np.float32).ravel()[0])
    ba.solve(problem, iterations=3)
    final_cost = float(problem.buffers["costs"].to_numpy().view(np.float32).ravel()[1])
    assert int(problem.buffers["status"].to_numpy().view(np.uint32).ravel()[0]) == 0
    assert final_cost < initial_cost * 0.5
    poses = problem.poses.to_numpy().view(np.float32).reshape(window, 4, 4)
    np.testing.assert_array_equal(poses[0], np.eye(4, dtype=np.float32))
    assert (
        max(
            np.linalg.norm(estimate[:3, 3] - target[:3, 3])
            for estimate, target in zip(poses, poses_truth)
        )
        < 0.005
    )
    inverse_depth = problem.inverse_depth.to_numpy().view(np.float32).reshape(window, 60, 80)
    observed = np.stack(depths) > 0
    assert np.isfinite(inverse_depth[observed]).all()
    assert np.max(abs(inverse_depth[observed] - 1 / np.stack(depths)[observed])) > 1e-7
    ratio = 1 / (np.stack(depths)[observed] * inverse_depth[observed])
    assert ratio.min() >= 0.9 - 1e-6 and ratio.max() <= 1.1 + 1e-6
    # A fresh evaluation of exactly the accepted pose/rho must reproduce final cost.
    encoder = ba.runtime.device.create_command_encoder()
    ba.record_evaluate(problem, encoder)
    ba.runtime.device.submit_command_buffer(encoder.finish())
    evaluated = problem.partials[-1][1].to_numpy().view(np.float32).ravel()[-2]
    np.testing.assert_allclose(evaluated, final_cost, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("final_cost, final_factors", [(2, 100), (0.5, 79)])
def test_gpu_worsened_cost_or_disappearing_factors_rolls_back_whole_window(
    final_cost, final_factors
):
    import os

    if os.environ.get("SLANG_RGBD_GPU_TESTS") != "1":
        pytest.skip("opt in after independent GPU review")
    pytest.importorskip("slangpy")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.dense_ba import DenseBA
    from slang_rgbd_slam.tracker import Camera

    runtime = VulkanRuntime()
    ba = DenseBA(runtime, Camera(16, 16, 15, 15, 7.5, 7.5), window=2)
    problem = ba.prepare()
    b = problem.buffers
    original = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    proposed = original.copy()
    proposed[:, 0, 3] = 0.5
    b["backupPoses"].copy_from_numpy(original.ravel())
    b["poses"].copy_from_numpy(proposed.ravel())
    b["backupDepth"].copy_from_numpy(np.ones(problem.count, np.float32))
    b["inverseDepth"].copy_from_numpy(np.full(problem.count, 2, np.float32))
    b["costs"].copy_from_numpy(np.array([1, 0, 100, 0], np.float32))
    b["transactionStatus"].copy_from_numpy(np.zeros(1, np.uint32))
    terms = np.zeros(ba.terms, np.float32)
    terms[-2:] = [final_cost, final_factors]
    final = runtime.buffer(terms)
    encoder = runtime.device.create_command_encoder()
    ba._dispatch(
        "finalizeDense",
        1,
        encoder,
        ba._params(problem),
        dict(
            inputTerms=final,
            costs=b["costs"],
            transactionStatus=b["transactionStatus"],
            status=b["status"],
        ),
    )
    encoder.global_barrier()
    ba._dispatch(
        "rollbackDense",
        problem.count,
        encoder,
        ba._params(problem),
        {k: b[k] for k in ("backupPoses", "backupDepth", "poses", "inverseDepth", "status")},
    )
    runtime.device.submit_command_buffer(encoder.finish())
    assert b["status"].to_numpy().view(np.uint32).ravel()[0] != 0
    np.testing.assert_array_equal(b["poses"].to_numpy().view(np.float32).reshape(2, 4, 4), original)
    np.testing.assert_array_equal(
        b["inverseDepth"].to_numpy().view(np.float32).ravel(), np.ones(problem.count, np.float32)
    )


def test_gpu_depth_update_clamps_before_final_cost_and_keeps_measured_input():
    import os

    if os.environ.get("SLANG_RGBD_GPU_TESTS") != "1":
        pytest.skip("opt in after independent GPU review")
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.dense_ba import DenseBA
    from slang_rgbd_slam.tracker import Camera

    runtime = VulkanRuntime()
    ba = DenseBA(runtime, Camera(16, 16, 15, 15, 7.5, 7.5), window=2)
    measured = np.full((2, 16, 16), 2, np.float32)
    initial = measured * 1.05
    problem = ba.prepare(measured, initial_depths=initial)
    b = problem.buffers
    b["inverseDepth"].copy_from_numpy(np.full(problem.count, 0.5, np.float32))
    b["hpd"].copy_from_numpy(np.zeros(problem.count * ba.pose_dofs, np.float32))
    b["hdd"].copy_from_numpy(np.ones(problem.count, np.float32))
    # Alternating unconstrained candidates -1.0 and 2.0 inverse metres.
    bd = np.where(np.arange(problem.count) % 2 == 0, 1.5, -1.5).astype(np.float32)
    b["bd"].copy_from_numpy(bd)
    b["delta"].copy_from_numpy(np.zeros(ba.pose_dofs, np.float32))
    b["status"].copy_from_numpy(np.zeros(1, np.uint32))
    encoder = runtime.device.create_command_encoder()
    ba._dispatch(
        "updateDenseDepth",
        problem.count,
        encoder,
        ba._params(problem),
        {k: b[k] for k in ("depths", "inverseDepth", "hpd", "hdd", "bd", "delta", "status")},
    )
    runtime.device.submit_command_buffer(encoder.finish())
    actual = b["inverseDepth"].to_numpy().view(np.float32).ravel()
    expected = np.where(np.arange(problem.count) % 2 == 0, 0.5 / 1.1, 0.5 / 0.9)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(
        b["depths"].to_numpy().view(np.float32).reshape(measured.shape), measured
    )
    np.testing.assert_array_equal(
        b["initialDepths"].to_numpy().view(np.float32).reshape(initial.shape), initial
    )
