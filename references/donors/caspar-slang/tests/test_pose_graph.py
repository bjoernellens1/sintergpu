"""SE(3) pose graph gauge, canonical derivatives and bounded GPU contracts."""

import importlib.util
import shutil

import numpy as np
import pytest


def _poses():
    poses = np.tile(np.eye(4), (4, 1, 1))
    for i in range(4):
        angle = i * 0.12
        poses[i, :3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        poses[i, :3, 3] = [i * 0.3, 0.1 * np.sin(i), 0.03 * i]
    return poses


def test_pose_graph_reference_has_six_dof_and_fixed_anchor():
    from caspar_slang.pose_graph import reference

    poses = _poses()
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.uint32)
    measures = np.array([np.linalg.inv(poses[i]) @ poses[j] for i, j in edges])
    perturbed = poses.copy()
    perturbed[1:, :3, 3] += [[0.02, 0.01, -0.01], [0.04, -0.02, 0.02], [0.08, 0.03, -0.04]]
    system = reference(perturbed, measures, edges)
    assert system["jacobian"].shape == (24, 18)
    assert system["hessian"].shape == (18, 18)
    assert np.linalg.matrix_rank(system["hessian"]) == 18
    delta = np.linalg.solve(system["hessian"] + 1e-4 * np.eye(18), -system["gradient"])
    assert np.linalg.norm(delta) > 0
    # Global frame change leaves every relative edge measurement unchanged.
    world = np.eye(4)
    a = 0.4
    world[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    world[:3, 3] = [0.3, -0.2, 0.4]
    transformed = np.array([world @ p for p in perturbed])
    np.testing.assert_allclose(
        reference(transformed, measures, edges)["residual"], system["residual"], atol=1e-9
    )


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_pose_factor_canonical_source():
    from caspar_slang.pose_graph import pose_graph_kernel, pose_graph_source

    kernel = pose_graph_kernel()
    assert kernel.metadata["source"] == "symforce.caspar"
    assert kernel.metadata["pose_basis"] == "left world SE3 rotation then translation"
    assert sum(b.access.value == "write" for b in kernel.buffers) == 78
    source = pose_graph_source()
    assert "void caspar_pose_graph_factor(" in source
    assert source.count("float caspar_copysign(") == 1
    assert "StructuredBuffer" not in source


@pytest.mark.skipif(
    shutil.which("slangc") is None or importlib.util.find_spec("symforce") is None,
    reason="requires slangc/SymForce",
)
@pytest.mark.parametrize(
    "entry",
    [
        "initializeMeasurements",
        "resetGraph",
        "linearizeEdges",
        "initializeCG",
        "operatorEdges",
        "operatorNodes",
        "pcgStep",
        "updateNodes",
        "evaluateEdges",
        "checkCost",
        "rollbackNodes",
    ],
)
def test_pose_graph_spirv(tmp_path, entry):
    from caspar_slang.compiler import compile_slang
    from caspar_slang.pose_graph import optimizer_source

    path = tmp_path / "pose_graph.slang"
    path.write_text(optimizer_source())
    output = tmp_path / (entry + ".spv")
    compile_slang(path, output, entry=entry)
    assert output.read_bytes()[:4] == b"\x03\x02\x23\x07"


def _apply_world_step(pose, delta):
    w = np.asarray(delta[:3])
    v = np.asarray(delta[3:])
    theta = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if theta < 1e-5:
        a = 1 - theta**2 / 6
        b = 0.5 - theta**2 / 24
        c = 1 / 6 - theta**2 / 120
    else:
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta**2
        c = (theta - np.sin(theta)) / theta**3
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + a * K + b * K @ K
    T[:3, 3] = (np.eye(3) + b * K + c * K @ K) @ v
    return T @ pose


def test_cpu_closed_loop_refines_rotations_and_translations():
    from caspar_slang.pose_graph import reference

    truth = _poses()
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.uint32)
    measurements = np.array([np.linalg.inv(truth[i]) @ truth[j] for i, j in edges])
    poses = truth.copy()
    for i in range(1, 4):
        poses[i] = _apply_world_step(
            poses[i], np.array([0.015 * i, -0.01 * i, 0.02 * i, 0.03 * i, -0.01 * i, 0.02 * i])
        )
    before = reference(poses, measurements, edges)["cost"]
    anchor = poses[0].copy()
    for _ in range(4):
        system = reference(poses, measurements, edges)
        delta = np.linalg.solve(system["hessian"] + 1e-4 * np.eye(18), -system["gradient"]).reshape(
            -1, 6
        )
        for i in range(1, 4):
            poses[i] = _apply_world_step(poses[i], delta[i - 1])
    assert reference(poses, measurements, edges)["cost"] < before * 1e-7
    np.testing.assert_array_equal(poses[0], anchor)
    np.testing.assert_allclose(poses, truth, atol=2e-6)


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_canonical_pose_factor_schedule_matches_all_72_numeric_derivatives():
    import re

    from caspar_slang.intrinsics import lower_scalar_line
    from caspar_slang.pose_graph import _matrix_quaternion, pose_graph_kernel, reference

    truth = _poses()
    poses = truth.copy()
    poses[1] = _apply_world_step(poses[1], np.array([0.02, -0.01, 0.025, 0.015, 0.02, -0.01]))
    poses[2] = _apply_world_step(poses[2], np.array([-0.01, 0.02, -0.015, 0.02, -0.01, 0.015]))
    measurement = np.linalg.inv(truth[1]) @ truth[2]
    qi = _matrix_quaternion(poses[1, :3, :3])
    qj = _matrix_quaternion(poses[2, :3, :3])
    qm = _matrix_quaternion(measurement[:3, :3])
    values = dict(
        zip(
            (
                "ix",
                "iy",
                "iz",
                "iw",
                "itx",
                "ity",
                "itz",
                "jx",
                "jy",
                "jz",
                "jw",
                "jtx",
                "jty",
                "jtz",
                "mx",
                "my",
                "mz",
                "mw",
                "mtx",
                "mty",
                "mtz",
            ),
            [*qi, *poses[1, :3, 3], *qj, *poses[2, :3, 3], *qm, *measurement[:3, 3]],
        )
    )

    def execute(inputs):
        env = dict(
            fma=lambda a, b, c: a * b + c,
            sqrt=np.sqrt,
            rsqrt=lambda a: 1 / np.sqrt(a),
            abs=abs,
            min=min,
            max=max,
        )
        outputs = {}
        for raw in pose_graph_kernel().code_lines:
            load = re.fullmatch(r"(r\d+) = (\w+)\[.*\];", raw)
            store = re.fullmatch(r"(out_\d+)\[.*\] = (r\d+);", raw)
            if load:
                env[load[1]] = inputs[load[2]]
            elif store:
                outputs[store[1]] = env[store[2]]
            else:
                lhs, rhs = lower_scalar_line(raw).rstrip(";").split("=", 1)
                if "?" in rhs:
                    cond, choices = rhs.split("?", 1)
                    yes, no = choices.split(":", 1)
                    rhs = yes if eval(cond, {"__builtins__": {}}, env) else no
                env[lhs.strip()] = eval(rhs, {"__builtins__": {}}, env)
        return np.array([outputs[f"out_{i}"] for i in range(78)])

    actual = execute(values)
    system = reference(poses, measurement[None], np.array([[1, 2]], np.uint32))
    expected = np.r_[
        system["residual"].ravel(),
        system["jacobian"][:, :6].ravel(),
        system["jacobian"][:, 6:12].ravel(),
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
    # Quaternion double cover must not change either residuals or derivatives.
    values.update({name: -values[name] for name in ("jx", "jy", "jz", "jw")})
    np.testing.assert_allclose(execute(values), actual, atol=1e-12)


def test_graph_recording_no_allocations_transfers_or_submissions():
    from caspar_slang.pose_graph import PoseGraph

    events = []

    class Buffer:
        def copy_from_numpy(self, value):
            assert value.dtype == np.uint32

    class Kernel:
        def __init__(self, name):
            self.name = name

        def dispatch(self, **kw):
            events.append((self.name, kw))
            assert kw["command_encoder"] is encoder

    class Runtime:
        def buffer(self, value):
            return Buffer()

        def compile(self, source, entry):
            return Kernel(entry)

    class Encoder:
        def global_barrier(self):
            pass

        def copy_buffer(self, *args):
            assert args[-1] <= 64 * 64

    runtime = Runtime()
    encoder = Encoder()
    graph = PoseGraph(runtime, max_nodes=4, max_edges=6)
    problem = graph.prepare(np.array([[0, 1], [1, 2], [2, 3]], np.uint32))
    runtime.buffer = lambda *args: pytest.fail("allocation while recording")
    graph.record_initialize_measurements(problem, encoder, first_edge=2, edge_count=1)
    assert events[0][1]["vars"]["cfg"]["firstEdge"] == 2
    assert events[0][1]["vars"]["cfg"]["initCount"] == 1
    graph.record(problem, encoder, iterations=1, cg_iterations=2)
    assert sum(name == "operatorEdges" for name, _ in events) == 2
    assert sum(name == "pcgStep" for name, _ in events) == 2
    assert events[-1][0] == "rollbackNodes"
    assert "hessian" not in problem.buffers
    # Updating graph topology writes integer controls only; measured edges stay GPU-owned.
    problem.update_topology(np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.uint32))
    assert problem.edge_count == 4


@pytest.mark.skipif(
    __import__("os").environ.get("CASPAR_SLANG_GPU_TESTS") != "1",
    reason="set CASPAR_SLANG_GPU_TESTS=1",
)
@pytest.mark.parametrize("large_rotations", [False, True])
def test_gpu_se3_closed_loop_gauge_and_transaction(large_rotations):
    from caspar_slang.pose_graph import PoseGraph, reference
    from caspar_slang.runtime import VulkanRuntime

    runtime = VulkanRuntime()
    truth = _poses()
    if large_rotations:
        for i, angle in enumerate([0, 1.1, 2.1, 2.9]):
            truth[i, :3, :3] = [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.uint32)
    graph = PoseGraph(runtime, max_nodes=4, max_edges=4)
    problem = graph.prepare(edges, truth)
    encoder = runtime.device.create_command_encoder()
    graph.record_initialize_measurements(problem, encoder)
    runtime.device.submit_command_buffer(encoder.finish())
    measurements = problem.measurements.to_numpy().view(np.float32).reshape(4, 4, 4).copy()
    expected = np.array([np.linalg.inv(truth[i]) @ truth[j] for i, j in edges])
    np.testing.assert_allclose(measurements, expected, atol=2e-7)
    poses = truth.copy()
    for i in range(1, 4):
        poses[i] = _apply_world_step(
            poses[i], np.array([0.015 * i, -0.01 * i, 0.02 * i, 0.03 * i, -0.01 * i, 0.02 * i])
        )
    initial_cost = reference(poses, measurements, edges)["cost"]
    problem.poses.copy_from_numpy(poses.astype(np.float32).ravel())
    # Initializing one new temporal edge must preserve all other measurements.
    encoder = runtime.device.create_command_encoder()
    graph.record_initialize_measurements(problem, encoder, first_edge=1, edge_count=1)
    runtime.device.submit_command_buffer(encoder.finish())
    updated_measures = problem.measurements.to_numpy().view(np.float32).reshape(4, 4, 4)
    np.testing.assert_array_equal(updated_measures[[0, 2, 3]], measurements[[0, 2, 3]])
    assert np.linalg.norm(updated_measures[1] - measurements[1]) > 1e-4
    problem.measurements.copy_from_numpy(measurements.ravel())
    encoder = runtime.device.create_command_encoder()
    graph.record(problem, encoder, iterations=4, cg_iterations=20)
    runtime.device.submit_command_buffer(encoder.finish())
    stats = problem.stats.to_numpy().view(np.uint32).ravel()
    np.testing.assert_array_equal(stats, [4, 0])
    corrected = problem.poses.to_numpy().view(np.float32).reshape(4, 4, 4).copy()
    np.testing.assert_array_equal(corrected[0], poses.astype(np.float32)[0])
    np.testing.assert_allclose(corrected, truth, atol=3e-4)
    assert reference(corrected, measurements, edges)["cost"] < initial_cost * 1e-5
    # Failed verification may disable a loop with zero information, even if its
    # numerical measurement is unavailable. Connected temporal edges still solve.
    bad = measurements.copy()
    bad[3] = np.nan
    information = np.ones((4, 2), np.float32)
    information[3] = 0
    problem.measurements.copy_from_numpy(bad.ravel())
    problem.edge_information.copy_from_numpy(information.ravel())
    encoder = runtime.device.create_command_encoder()
    graph.record(problem, encoder, iterations=1, cg_iterations=20)
    runtime.device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(problem.stats.to_numpy().view(np.uint32).ravel(), [3, 0])
    # Nonzero information on an invalid edge is a visible failure and restores
    # all poses transactionally; an anchor alone is not a substitute for this.
    information[3] = 1
    problem.edge_information.copy_from_numpy(information.ravel())
    before = problem.poses.to_numpy().view(np.float32).copy()
    encoder = runtime.device.create_command_encoder()
    graph.record(problem, encoder, iterations=2, cg_iterations=5)
    runtime.device.submit_command_buffer(encoder.finish())
    assert problem.stats.to_numpy().view(np.uint32).ravel()[1] == 1
    np.testing.assert_array_equal(problem.poses.to_numpy().view(np.float32), before)
