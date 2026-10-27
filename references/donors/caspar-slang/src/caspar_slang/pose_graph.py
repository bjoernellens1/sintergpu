"""Bounded GPU SE(3) pose graph with Caspar factors and matrix-free PCG.

The measurements are camera_i_from_camera_j = inverse(world_from_i) @
world_from_j. Rotation error is shortest-sign quaternion chordal error;
translation error is in camera i. Pose increments use world-frame left SE(3).
Only topology indices and configuration are host control data while recording.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .caspar_adapter import from_caspar_kernel
from .primitives import emit_scalar_function, primitives_source


@lru_cache(maxsize=1)
def pose_graph_kernel():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    R = T.Annotated[sf.Scalar, mem.ReadSequential]
    W = T.Annotated[sf.Scalar, mem.WriteSequential]
    Outputs = T.Tuple[(W,) * 78]
    lib = CasparLibrary(name="pose_graph")

    @lib.add_kernel
    def pose_graph_factor(
        ix: R,
        iy: R,
        iz: R,
        iw: R,
        itx: R,
        ity: R,
        itz: R,
        jx: R,
        jy: R,
        jz: R,
        jw: R,
        jtx: R,
        jty: R,
        jtz: R,
        mx: R,
        my: R,
        mz: R,
        mw: R,
        mtx: R,
        mty: R,
        mtz: R,
    ) -> Outputs:
        xi = sf.V6.symbolic("xi")
        xj = sf.V6.symbolic("xj")
        qi = sf.Quaternion(sf.V3(ix, iy, iz), iw)
        qj = sf.Quaternion(sf.V3(jx, jy, jz), jw)
        qm = sf.Quaternion(sf.V3(mx, my, mz), mw)
        qi_changed = sf.Quaternion(sf.V3(xi[:3]) / 2, 1) * qi
        qj_changed = sf.Quaternion(sf.V3(xj[:3]) / 2, 1) * qj
        ti = sf.V3(itx, ity, itz)
        tj = sf.V3(jtx, jty, jtz)
        ti_changed = ti + sf.V3(xi[:3]).cross(ti) + sf.V3(xi[3:])
        tj_changed = tj + sf.V3(xj[:3]).cross(tj) + sf.V3(xj[3:])
        error_quat = qm.conj() * (qi_changed.conj() * qj_changed)
        sign = sf.Piecewise((-1, error_quat.w < 0), (1, True))
        translation = sf.Rot3(qi_changed).to_rotation_matrix().T * (
            tj_changed - ti_changed
        ) - sf.V3(mtx, mty, mtz)
        error = sf.V6([*(2 * sign * error_quat.xyz), *translation])
        zero = {value: 0 for value in [*xi, *xj]}
        residual = [v.subs(zero) for v in error]
        jac_i = [e.diff(v).subs(zero) for e in error for v in xi]
        jac_j = [e.diff(v).subs(zero) for e in error for v in xj]
        return tuple(residual + jac_i + jac_j)

    ir = from_caspar_kernel(lib.kernels[0])
    ir.metadata.update(
        schedule="DABSEG/get_lines",
        register_count=str(len(ir.registers)),
        pose_basis="left world SE3 rotation then translation",
        residual="shortest-sign quaternion chordal plus relative translation",
    )
    return ir


@lru_cache(maxsize=1)
def pose_graph_source():
    return (
        primitives_source()
        + "\n"
        + emit_scalar_function(pose_graph_kernel(), "caspar_pose_graph_factor")
    )


@lru_cache(maxsize=1)
def optimizer_source():
    return (
        pose_graph_source()
        + "\n"
        + Path(__file__).with_name("shaders").joinpath("pose_graph.slang").read_text()
    )


def _matrix_quaternion(rotation):
    """Independent NumPy matrix-to-quaternion conversion, xyzw order."""
    m = np.asarray(rotation, dtype=np.float64)
    # Symmetric eigenproblem avoids sharing the shader's trace/diagonal branches.
    K = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], m[1, 0] + m[0, 1], m[2, 0] + m[0, 2], m[2, 1] - m[1, 2]],
            [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], m[2, 1] + m[1, 2], m[0, 2] - m[2, 0]],
            [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m.trace()],
        ]
    )
    _, vectors = np.linalg.eigh(K)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def _quaternion_product(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def _residual(poses, measurements, edges, information):
    result = []
    for e, (i, j) in enumerate(edges):
        relative = np.linalg.inv(poses[i]) @ poses[j]
        pred = _matrix_quaternion(relative[:3, :3])
        measured = _matrix_quaternion(measurements[e, :3, :3])
        qerror = _quaternion_product(np.r_[-measured[:3], measured[3]], pred)
        sign = 1 if qerror[3] >= 0 else -1
        result.append(
            np.r_[
                2 * sign * qerror[:3] * information[e, 0],
                (relative[:3, 3] - measurements[e, :3, 3]) * information[e, 1],
            ]
        )
    return np.asarray(result, dtype=np.float64).reshape(-1, 6)


def reference(poses, measurements, edges, edge_information=None):
    """Independent float64 full anchored system with central SE(3) differences.

    This oracle is deliberately separate from the generated derivative code;
    it is used for validation, never by the GPU solver.
    """
    p = np.asarray(poses, dtype=np.float64)
    m = np.asarray(measurements, dtype=np.float64)
    e = np.asarray(edges)
    if (
        p.ndim != 3
        or p.shape[1:] != (4, 4)
        or m.shape != (len(e), 4, 4)
        or e.ndim != 2
        or e.shape[1] != 2
        or e.dtype.kind not in "iu"
        or np.any(e < 0)
        or np.any(e >= len(p))
        or not np.isfinite(p).all()
        or not np.isfinite(m).all()
    ):
        raise ValueError("poses Nx4x4, measurements Ex4x4 and valid integer Ex2 edges required")
    info = (
        np.ones((len(e), 2))
        if edge_information is None
        else np.asarray(edge_information, dtype=np.float64)
    )
    if info.shape != (len(e), 2) or not np.isfinite(info).all() or np.any(info < 0):
        raise ValueError("edge information must be nonnegative finite Ex2 square-root weights")
    residual = _residual(p, m, e, info)
    jacobian = np.zeros((len(e) * 6, (len(p) - 1) * 6))
    epsilon = 1e-6
    for node in range(1, len(p)):
        for axis in range(6):
            plus = p.copy()
            minus = p.copy()
            if axis < 3:
                v = np.eye(3)[axis]
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                Rp = np.eye(3) + np.sin(epsilon) * K + (1 - np.cos(epsilon)) * (K @ K)
                Rm = Rp.T
                plus[node, :3, :] = Rp @ p[node, :3, :]
                minus[node, :3, :] = Rm @ p[node, :3, :]
            else:
                plus[node, axis - 3, 3] += epsilon
                minus[node, axis - 3, 3] -= epsilon
            jacobian[:, (node - 1) * 6 + axis] = (
                _residual(plus, m, e, info) - _residual(minus, m, e, info)
            ).ravel() / (2 * epsilon)
    vector = residual.ravel()
    return dict(
        residual=residual,
        jacobian=jacobian,
        hessian=jacobian.T @ jacobian,
        gradient=jacobian.T @ vector,
        cost=float(vector @ vector),
    )


@dataclass
class PoseGraphProblem:
    buffers: dict
    owner: object
    node_count: int
    edge_count: int

    @property
    def poses(self):
        return self.buffers["poses"]

    @property
    def measurements(self):
        return self.buffers["measurements"]

    @property
    def edge_information(self):
        return self.buffers["edgeInformation"]

    @property
    def stats(self):
        return self.buffers["stats"]

    @property
    def cost(self):
        return self.buffers["cost"]

    def update_topology(self, edges, node_count=None):
        self.owner.update_topology(self, edges, node_count)


class PoseGraph:
    """Anchored six-DoF GPU graph. Bounds: 64 nodes, 256 edges.

    All numerical values and PCG work vectors stay resident. Edge Jacobians are
    six-by-twelve; the normal matrix is never assembled. CPU topology updates
    upload only integer adjacency/prefix data. ``edge_information`` contains
    square-root weights for rotation and translation; both zero disable an edge.
    ``stats`` is uint[2] (valid edges, failure), ``cost`` float[2] (before/after
    the last nonlinear iteration). Failure restores the entire call's input
    poses. Node zero is fixed and is never assigned an optimization variable.
    """

    def __init__(self, runtime=None, max_nodes=64, max_edges=256):
        from .runtime import VulkanRuntime

        if (
            not isinstance(max_nodes, int)
            or isinstance(max_nodes, bool)
            or not 1 <= max_nodes <= 64
            or not isinstance(max_edges, int)
            or isinstance(max_edges, bool)
            or not 1 <= max_edges <= 256
        ):
            raise ValueError("pose graph supports 1..64 nodes and 1..256 edges")
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        source = optimizer_source()
        names = (
            "initializeMeasurements",
            "resetGraph",
            "resetEdgeCount",
            "linearizeEdges",
            "initializeCG",
            "operatorEdges",
            "operatorNodes",
            "pcgStep",
            "updateNodes",
            "evaluateEdges",
            "checkCost",
            "rollbackNodes",
        )
        self.kernels = {name: self.runtime.compile(source, entry=name) for name in names}

    def _topology(self, edges, node_count=None):
        e = np.asarray(edges)
        if (
            e.ndim != 2
            or e.shape[1] != 2
            or e.dtype.kind not in "iu"
            or len(e) > self.max_edges
            or np.any(e < 0)
            or np.any(e >= self.max_nodes)
            or np.any(e[:, 0] == e[:, 1])
        ):
            raise ValueError("topology requires non-self integer Ex2 edges within graph capacity")
        n = (int(e.max()) + 1 if e.size else 1) if node_count is None else node_count
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or not 1 <= n <= self.max_nodes
            or (e.size and int(e.max()) >= n)
        ):
            raise ValueError("node_count must include every topology endpoint")
        topology = np.zeros(self.max_edges * 2, np.uint32)
        topology[: e.size] = e.astype(np.uint32).ravel()
        incident = [[] for _ in range(self.max_nodes)]
        for index, (i, j) in enumerate(e):
            incident[i].append(index * 2)
            incident[j].append(index * 2 + 1)
        offsets = np.zeros(self.max_nodes + 1, np.uint32)
        offsets[1:] = np.cumsum([len(a) for a in incident], dtype=np.uint32)
        adjacency = np.zeros(self.max_edges * 2, np.uint32)
        values = [edge for row in incident for edge in row]
        adjacency[: len(values)] = values
        return n, len(e), topology, offsets, adjacency

    def prepare(self, topologyedges, poses=None):
        initial = np.tile(np.eye(4, dtype=np.float32), (self.max_nodes, 1, 1))
        n_init = None
        if poses is not None:
            p = np.asarray(poses, dtype=np.float32)
            if (
                p.ndim != 3
                or p.shape[1:] != (4, 4)
                or not 1 <= len(p) <= self.max_nodes
                or not np.isfinite(p).all()
                or not np.allclose(p[:, 3, :], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(
                    p[:, :3, :3].transpose(0, 2, 1) @ p[:, :3, :3], np.eye(3), atol=2e-3
                )
                or not np.allclose(np.linalg.det(p[:, :3, :3]), 1, atol=2e-3)
            ):
                raise ValueError("initial poses must be finite proper rigid transforms")
            initial[: len(p)] = p
            n_init = len(p)
        n, m, topology, offsets, adjacency = self._topology(topologyedges, n_init)
        arrays = dict(
            poses=initial.ravel(),
            backup=initial.ravel(),
            measurements=np.tile(np.eye(4, dtype=np.float32), (self.max_edges, 1, 1)).ravel(),
            edgeInformation=np.ones(self.max_edges * 2, np.float32),
            topology=topology,
            offsets=offsets,
            adjacency=adjacency,
            initMask=np.ones(self.max_edges, np.uint32),
            edgeJacobian=np.zeros(self.max_edges * 72, np.float32),
            edgeResidual=np.zeros(self.max_edges * 6, np.float32),
            edgeAction=np.zeros(self.max_edges * 6, np.float32),
            edgeCosts=np.zeros(self.max_edges, np.float32),
            edgeValid=np.zeros(self.max_edges, np.uint32),
            solution=np.zeros(self.max_nodes * 6, np.float32),
            residualCG=np.zeros(self.max_nodes * 6, np.float32),
            directionCG=np.zeros(self.max_nodes * 6, np.float32),
            preconditioner=np.zeros(self.max_nodes * 6, np.float32),
            appliedCG=np.zeros(self.max_nodes * 6, np.float32),
            cgScalars=np.zeros(4, np.float32),
            cgState=np.zeros(1, np.uint32),
            stats=np.zeros(2, np.uint32),
            cost=np.zeros(2, np.float32),
        )
        return PoseGraphProblem(
            {name: self.runtime.buffer(value) for name, value in arrays.items()}, self, n, m
        )

    def update_topology(self, problem, edges, node_count=None):
        self._owned(problem)
        n, m, topology, offsets, adjacency = self._topology(edges, node_count)
        for name, value in (("topology", topology), ("offsets", offsets), ("adjacency", adjacency)):
            problem.buffers[name].copy_from_numpy(value)
        problem.node_count, problem.edge_count = n, m

    def _owned(self, problem):
        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another graph")

    def _dispatch(self, entry, count, problem, encoder, cfg, names, extra=None):
        self.kernels[entry].dispatch(
            thread_count=[count, 1, 1],
            vars=dict(cfg=cfg, **{name: problem.buffers[name] for name in names}, **(extra or {})),
            command_encoder=encoder,
        )
        encoder.global_barrier()

    def record_initialize_measurements(
        self, problem, encoder, first_edge=0, edge_count=None, edge_mask=None
    ):
        self._owned(problem)
        count = problem.edge_count - first_edge if edge_count is None else edge_count
        if (
            not isinstance(first_edge, int)
            or not isinstance(count, int)
            or first_edge < 0
            or count < 0
            or first_edge + count > problem.edge_count
        ):
            raise ValueError("measurement initialization range exceeds topology")
        if count:
            cfg = dict(
                nodeCount=problem.node_count,
                edgeCount=problem.edge_count,
                damping=0.0,
                firstEdge=first_edge,
                initCount=count,
            )
            self._dispatch(
                "initializeMeasurements",
                count,
                problem,
                encoder,
                cfg,
                ("topology", "poses", "measurements"),
                {"initMask": problem.buffers["initMask"] if edge_mask is None else edge_mask},
            )

    def record(
        self,
        problem,
        encoder,
        iterations=3,
        cg_iterations=20,
        active_nodes=None,
        active_edges=None,
        damping=1e-4,
    ):
        self._owned(problem)
        n = problem.node_count if active_nodes is None else active_nodes
        m = problem.edge_count if active_edges is None else active_edges
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or not 1 <= n <= self.max_nodes
            or not isinstance(m, int)
            or isinstance(m, bool)
            or not 0 <= m <= problem.edge_count
            or not isinstance(iterations, int)
            or not 1 <= iterations <= 20
            or not isinstance(cg_iterations, int)
            or not 1 <= cg_iterations <= 100
            or not np.isfinite(damping)
            or not 0 <= damping <= np.finfo(np.float32).max
        ):
            raise ValueError("invalid bounded graph optimization configuration")
        cfg = dict(nodeCount=n, edgeCount=m, damping=float(damping), firstEdge=0, initCount=0)
        encoder.copy_buffer(problem.buffers["backup"], 0, problem.poses, 0, n * 64)
        encoder.global_barrier()
        self._dispatch("resetGraph", 1, problem, encoder, cfg, ("stats", "cost"))
        for _ in range(iterations):
            self._dispatch("resetEdgeCount", 1, problem, encoder, cfg, ("stats",))
            if m:
                self._dispatch(
                    "linearizeEdges",
                    m,
                    problem,
                    encoder,
                    cfg,
                    (
                        "topology",
                        "poses",
                        "measurements",
                        "edgeInformation",
                        "edgeJacobian",
                        "edgeResidual",
                        "edgeCosts",
                        "edgeValid",
                        "stats",
                    ),
                )
            self._dispatch(
                "initializeCG",
                64,
                problem,
                encoder,
                cfg,
                (
                    "topology",
                    "offsets",
                    "adjacency",
                    "edgeJacobian",
                    "edgeResidual",
                    "edgeValid",
                    "edgeCosts",
                    "solution",
                    "residualCG",
                    "directionCG",
                    "preconditioner",
                    "cgScalars",
                    "cgState",
                    "stats",
                    "cost",
                ),
            )
            for _ in range(cg_iterations):
                if m:
                    self._dispatch(
                        "operatorEdges",
                        m * 6,
                        problem,
                        encoder,
                        cfg,
                        (
                            "topology",
                            "edgeJacobian",
                            "edgeValid",
                            "directionCG",
                            "edgeAction",
                            "cgState",
                        ),
                    )
                self._dispatch(
                    "operatorNodes",
                    n * 6,
                    problem,
                    encoder,
                    cfg,
                    (
                        "offsets",
                        "adjacency",
                        "edgeJacobian",
                        "edgeValid",
                        "edgeAction",
                        "directionCG",
                        "appliedCG",
                        "cgState",
                    ),
                )
                self._dispatch(
                    "pcgStep",
                    64,
                    problem,
                    encoder,
                    cfg,
                    (
                        "solution",
                        "residualCG",
                        "directionCG",
                        "preconditioner",
                        "appliedCG",
                        "cgScalars",
                        "cgState",
                        "stats",
                    ),
                )
            self._dispatch("updateNodes", n, problem, encoder, cfg, ("poses", "solution", "stats"))
            if m:
                self._dispatch(
                    "evaluateEdges",
                    m,
                    problem,
                    encoder,
                    cfg,
                    (
                        "topology",
                        "poses",
                        "measurements",
                        "edgeInformation",
                        "edgeValid",
                        "edgeCosts",
                        "stats",
                    ),
                )
            self._dispatch(
                "checkCost", 64, problem, encoder, cfg, ("edgeCosts", "cgScalars", "stats", "cost")
            )
        self._dispatch("rollbackNodes", n * 16, problem, encoder, cfg, ("poses", "backup", "stats"))
