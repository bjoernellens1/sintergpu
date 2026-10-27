"""Projection derivatives, generated source and persistent dispatch contracts."""

import importlib.util
import shutil

import numpy as np
import pytest


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_reprojection_has_real_caspar_derivatives_and_scalar_interface():
    from caspar_slang.reprojection import reproject_kernel, reproject_source

    kernel = reproject_kernel()
    assert kernel.metadata["source"] == "symforce.caspar"
    assert kernel.metadata["schedule"] == "DABSEG/get_lines"
    assert len([b for b in kernel.buffers if b.access.value == "write"]) == 20
    source = reproject_source()
    assert "void caspar_reproject(" in source
    assert source.count("float caspar_copysign(") == 1
    assert "StructuredBuffer" not in source
    assert "params." not in source


def test_reference_jacobians_agree_with_central_difference():
    from caspar_slang.reprojection import reference

    angle = 0.3
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [0.1, -0.2, 0.3]
    points = np.array([[0.2, 0.3, 2], [-0.4, 0.2, 3], [0.4, -0.1, 1.5]])
    measured = np.array([[100, 120], [150, 90], [140, 150]])
    calibration = (320, 300, 160, 120)
    residual, jpose, jpoint = reference(points, measured, pose, calibration)
    eps = 1e-6
    q = points @ R.T + pose[:3, 3]

    def project(p):
        return (
            np.column_stack(
                (
                    calibration[0] * p[:, 0] / p[:, 2] + calibration[2],
                    calibration[1] * p[:, 1] / p[:, 2] + calibration[3],
                )
            )
            - measured
        )

    for i in range(6):
        if i < 3:
            axis = np.eye(3)[i]
            plus = q + eps * np.cross(axis, q)
            minus = q - eps * np.cross(axis, q)
        else:
            plus = q + eps * np.eye(3)[i - 3]
            minus = q - eps * np.eye(3)[i - 3]
        np.testing.assert_allclose(
            jpose[:, :, i], (project(plus) - project(minus)) / (2 * eps), rtol=1e-7, atol=1e-7
        )
    for i in range(3):
        plus = (points + eps * np.eye(3)[i]) @ R.T + pose[:3, 3]
        minus = (points - eps * np.eye(3)[i]) @ R.T + pose[:3, 3]
        np.testing.assert_allclose(
            jpoint[:, :, i], (project(plus) - project(minus)) / (2 * eps), rtol=1e-7, atol=1e-7
        )
    np.testing.assert_allclose(residual, project(q), atol=1e-10)


@pytest.mark.skipif(
    shutil.which("slangc") is None or importlib.util.find_spec("symforce") is None,
    reason="requires slangc/SymForce",
)
@pytest.mark.parametrize(
    "entry",
    ["resetReprojection", "buildReprojection", "updateReprojection", "evaluateReprojection"],
)
def test_reprojection_compiles_to_spirv(tmp_path, entry):
    from caspar_slang.compiler import compile_slang
    from caspar_slang.reprojection import optimizer_source

    path = tmp_path / "reprojection.slang"
    path.write_text(optimizer_source())
    output = tmp_path / (entry + ".spv")
    compile_slang(path, output, entry=entry)
    assert output.read_bytes()[:4] == b"\x03\x02\x23\x07"


def _evaluate_schedule(ir, values):
    """Execute canonical scalar scheduled IR on CPU, independent of Slang/GPU."""
    import re

    from caspar_slang.intrinsics import lower_scalar_line

    env = dict(
        fma=lambda a, b, c: a * b + c,
        sqrt=np.sqrt,
        rsqrt=lambda x: 1 / np.sqrt(x),
        min=min,
        max=max,
    )
    outputs = {}
    for raw in ir.code_lines:
        load = re.fullmatch(r"(r\d+) = (\w+)\[.*\];", raw)
        store = re.fullmatch(r"(out_\d+)\[.*\] = (r\d+);", raw)
        if load:
            env[load[1]] = values[load[2]]
        elif store:
            outputs[store[1]] = env[store[2]]
        else:
            lhs, rhs = lower_scalar_line(raw).rstrip(";").split("=", 1)
            env[lhs.strip()] = eval(rhs, {"__builtins__": {}}, env)
    return np.array([outputs[f"out_{i}"] for i in range(len(outputs))])


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_actual_caspar_reprojection_schedule_matches_numpy():
    from caspar_slang.reprojection import reference, reproject_kernel

    p = np.array([[0.3, -0.2, 2.0]])
    uv = np.array([[120.0, 140.0]])
    angle = 0.2
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    pose[:3, 3] = [0.1, 0.2, 0.3]
    q = p[0] @ pose[:3, :3].T + pose[:3, 3]
    args = dict(
        zip(("x", "y", "z", "fx", "fy", "cx", "cy", "u", "v"), [*q, 310, 300, 160, 120, *uv[0]])
    )
    args.update({f"r{a}{b}": pose[a, b] for a in range(3) for b in range(3)})
    actual = _evaluate_schedule(reproject_kernel(), args)
    residual, jpose, jpoint = reference(p, uv, pose, (310, 300, 160, 120))
    np.testing.assert_allclose(
        actual, np.r_[residual.ravel(), jpose.ravel(), jpoint.ravel()], atol=1e-10
    )


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_dense_caspar_factors_match_world_pose_and_inverse_depth_finite_difference():
    from caspar_slang.reprojection import factor_kernels

    sa = 0.3
    ta = 0.2
    source = np.array([[np.cos(sa), 0, np.sin(sa)], [0, 1, 0], [-np.sin(sa), 0, np.cos(sa)]])
    target = np.array([[np.cos(ta), -np.sin(ta), 0], [np.sin(ta), np.cos(ta), 0], [0, 0, 1]])
    ray = np.array([0.1, -0.2, 1.0])
    rho = 0.6
    source_t = np.array([0.3, 0.1, 0.2])
    target_t = np.array([-0.2, 0.1, 0.3])
    world = source @ ray / rho + source_t
    camera = target @ world + target_t
    measured = camera + np.array([0.03, -0.01, 0.02])
    normal = np.array([0.2, 0.3, 0.9])
    normal /= np.linalg.norm(normal)
    fx, fy, gx, gy = 310.0, 300.0, 0.2, -0.15
    args = dict(zip(("wx", "wy", "wz", "x", "y", "z"), [*world, *camera]))
    args.update({f"tr{a}{b}": target[a, b] for a in range(3) for b in range(3)})
    args.update({f"sr{a}{b}": source[a, b] for a in range(3) for b in range(3)})
    args.update(
        dict(
            zip(
                ("rx", "ry", "rz", "rho", "qx", "qy", "qz", "nx", "ny", "nz"),
                [*ray, rho, *measured, *normal],
            )
        )
    )
    args.update(targetIntensity=0.6, sourceIntensity=0.4, gx=gx, gy=gy, fx=fx, fy=fy)
    actual = _evaluate_schedule(factor_kernels()["dense_factor"], args)

    def residual(w):
        c = target @ w + target_t
        photo = (
            0.2
            + gx * fx * (c[0] / c[2] - camera[0] / camera[2])
            + gy * fy * (c[1] / c[2] - camera[1] / camera[2])
        )
        return np.array([normal @ (c - measured), photo])

    eps = 1e-6
    jac = np.zeros((2, 6))
    for k in range(6):
        direction = np.cross(np.eye(3)[k], world) if k < 3 else np.eye(3)[k - 3]
        jac[:, k] = (residual(world + eps * direction) - residual(world - eps * direction)) / (
            2 * eps
        )
    rho_jac = (
        residual(source @ ray / (rho + eps) + source_t)
        - residual(source @ ray / (rho - eps) + source_t)
    ) / (2 * eps)
    np.testing.assert_allclose(
        actual, np.r_[residual(world), jac[0], jac[1], rho_jac], rtol=2e-7, atol=2e-8
    )
    assert factor_kernels()["dense_factor"].metadata["depth_basis"] == "source inverse z depth"


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_single_frame_geometric_and_photo_schedules():
    from caspar_slang.reprojection import factor_kernels

    p = np.array([0.3, -0.2, 1.8])
    q = np.array([0.31, -0.22, 1.9])
    n = np.array([0.1, 0.2, 0.97])
    args = dict(zip(("px", "py", "pz", "qx", "qy", "qz", "nx", "ny", "nz"), [*p, *q, *n]))
    actual = _evaluate_schedule(factor_kernels()["point_plane"], args)
    np.testing.assert_allclose(actual, np.r_[n @ (p - q), np.cross(p, n), n], atol=1e-10)
    x, y, z = p
    fx, fy, gx, gy = 250.0, 270.0, -0.08, 0.2
    args = dict(
        zip(
            ("x", "y", "z", "fx", "fy", "gx", "gy", "targetI", "sourceI"),
            [*p, fx, fy, gx, gy, 0.7, 0.4],
        )
    )
    grad = np.array([gx * fx / z, gy * fy / z, -(gx * fx * x + gy * fy * y) / z**2])
    actual = _evaluate_schedule(factor_kernels()["photo_factor"], args)
    np.testing.assert_allclose(actual, np.r_[0.3, np.cross(p, grad), grad], atol=1e-10)


def test_recording_uses_preallocated_buffers_and_preserves_failure():
    from types import SimpleNamespace

    from caspar_slang.reprojection import ReprojectionGN, ReprojectionProblem

    recorded = []

    class Kernel:
        def dispatch(self, **kw):
            assert kw["command_encoder"] is encoder
            recorded.append(kw)

    class Encoder:
        def global_barrier(self):
            pass

    class Runtime:
        def buffer(self, *args):
            raise AssertionError("allocation in recording")

    encoder = Encoder()
    solver = object.__new__(ReprojectionGN)
    solver.runtime = Runtime()
    solver.intrinsics = (310.0, 300.0, 160.0, 120.0)
    solver.kernels = {
        name: Kernel()
        for name in (
            "resetReprojection",
            "resetReprojectionCount",
            "buildReprojection",
            "updateReprojection",
            "evaluateReprojection",
        )
    }
    solver.reduce_partials = Kernel()
    solver.solve = Kernel()
    problem = ReprojectionProblem(
        65, [(2, object()), (1, object())], object(), object(), object(), object(), solver
    )
    problem.reset(encoder)
    for _ in range(2):
        solver.record(problem, object(), object(), object(), object(), encoder)
    assert len(recorded) == 11
    # Failure-clearing entry runs only for the explicit frame reset.
    assert "stats" in recorded[0]["vars"] and "cost" in recorded[0]["vars"]
    assert [r["thread_count"][0] for r in recorded] == [1, 1, 65, 2, 1, 1, 1, 65, 2, 1, 1]
    assert not hasattr(problem, "residual") and not hasattr(problem, "jacobian")
    recorded.clear()
    solver.record_evaluate(problem, object(), object(), object(), object(), encoder)
    assert len(recorded) == 4
    assert "pose" not in recorded[-1]["vars"]
    assert "delta" not in recorded[-1]["vars"]

    other = SimpleNamespace(owner=object())
    with pytest.raises(ValueError, match="another solver"):
        solver.record(other, None, None, None, None, encoder)


@pytest.mark.skipif(
    __import__("os").environ.get("CASPAR_SLANG_GPU_TESTS") != "1",
    reason="set CASPAR_SLANG_GPU_TESTS=1",
)
def test_gpu_reprojection_normal_equations_and_pose_convergence():
    from caspar_slang.reprojection import ReprojectionGN, reference
    from caspar_slang.runtime import VulkanRuntime

    runtime = VulkanRuntime()
    calibration = (310.0, 300.0, 160.0, 120.0)
    solver = ReprojectionGN(runtime, calibration)
    rng = np.random.default_rng(26)
    p = rng.uniform([-1, -0.7, 1.5], [1, 0.7, 4], size=(83, 3)).astype(np.float32)
    theta = 0.03
    true = np.eye(4)
    true[:3, :3] = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
    true[:3, 3] = [0.025, -0.015, 0.012]
    q = p @ true[:3, :3].T + true[:3, 3]
    uv = np.column_stack((310 * q[:, 0] / q[:, 2] + 160, 300 * q[:, 1] / q[:, 2] + 120)).astype(
        np.float32
    )
    mask = np.ones(len(p), np.uint32)
    mask[-17:] = 0
    pose = runtime.buffer(np.eye(4, dtype=np.float32).ravel())
    points = runtime.buffer(p)
    pixels = runtime.buffer(uv)
    valid = runtime.buffer(mask)
    problem = solver.prepare(len(p))
    encoder = runtime.device.create_command_encoder()
    problem.reset(encoder)
    solver.record(problem, points, pixels, valid, pose, encoder, huber=3.0)
    runtime.device.submit_command_buffer(encoder.finish())
    terms = problem.partials[-1][1].to_numpy().view(np.float32).ravel()
    r, j, _ = reference(p[mask > 0], uv[mask > 0], np.eye(4), calibration)
    w = np.sqrt(np.minimum(1.0, 3.0 / np.maximum(np.linalg.norm(r, axis=1), 1e-12)))
    weighted_r = (r * w[:, None]).ravel()
    weighted_j = (j * w[:, None, None]).reshape(-1, 6)
    np.testing.assert_allclose(
        terms[:36].reshape(6, 6), weighted_j.T @ weighted_j, rtol=1e-4, atol=0.3
    )
    np.testing.assert_allclose(terms[36:42], weighted_j.T @ weighted_r, rtol=1e-4, atol=0.03)
    np.testing.assert_allclose(terms[42], weighted_r @ weighted_r, rtol=1e-4, atol=0.002)
    np.testing.assert_array_equal(problem.stats.to_numpy().view(np.uint32).ravel(), [66, 0])
    encoder = runtime.device.create_command_encoder()
    for _ in range(7):
        solver.record(problem, points, pixels, valid, pose, encoder, huber=3.0)
    runtime.device.submit_command_buffer(encoder.finish())
    np.testing.assert_allclose(pose.to_numpy().view(np.float32).reshape(4, 4), true, atol=2e-5)
    # Final-pose evaluation updates count/cost, with no further pose mutation.
    final_pose = pose.to_numpy().view(np.float32).copy()
    encoder = runtime.device.create_command_encoder()
    solver.record_evaluate(problem, points, pixels, valid, pose, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(pose.to_numpy().view(np.float32), final_pose)
    final_r, _, _ = reference(p[mask > 0], uv[mask > 0], final_pose.reshape(4, 4), calibration)
    np.testing.assert_allclose(
        problem.cost.to_numpy().view(np.float32).ravel()[0], np.sum(final_r * final_r), atol=1e-7
    )
    # A failed iteration stays failed and cannot mutate pose during this frame.
    before = pose.to_numpy().view(np.float32).copy()
    empty = runtime.buffer(np.zeros(len(p), np.uint32))
    encoder = runtime.device.create_command_encoder()
    solver.record(problem, points, pixels, empty, pose, encoder)
    solver.record(problem, points, pixels, valid, pose, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    assert problem.stats.to_numpy().view(np.uint32).ravel()[1] == 1
    np.testing.assert_array_equal(pose.to_numpy().view(np.float32), before)
