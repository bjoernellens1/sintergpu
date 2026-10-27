"""GPU resident point-to-plane Gauss--Newton with Caspar symbolic derivatives."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class StepResult:
    residual: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    gradient: np.ndarray
    delta: np.ndarray
    status: int
    cost: float


@dataclass
class PreparedProblem:
    count: int
    point_count: int
    buffers: dict
    partials: list
    delta: object
    status: object
    owner: object


@lru_cache(maxsize=1)
def residual_kernel():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    from .caspar_adapter import from_caspar_kernel

    lib = CasparLibrary()

    @lib.add_kernel
    def point_plane(
        points: T.Annotated[sf.V3, mem.ReadIndexed],
        targets: T.Annotated[sf.V3, mem.ReadSequential],
        normals: T.Annotated[sf.V3, mem.ReadSequential],
    ) -> T.Tuple[
        T.Annotated[sf.Scalar, mem.WriteSequential], T.Annotated[sf.V6, mem.WriteSequential]
    ]:
        # Differentiate the infinitesimal rigid transformation before evaluating at zero.
        # Caspar schedules both the residual and these symbolic derivatives through DABSEG.
        xi = sf.V6.symbolic("xi")
        omega = sf.V3(xi[:3])
        translation = sf.V3(xi[3:])
        transformed = points + omega.cross(points) + translation
        expression = normals.dot(transformed - targets)
        at_zero = {v: 0 for v in xi}
        return expression.subs(at_zero), sf.V6([expression.diff(v).subs(at_zero) for v in xi])

    return from_caspar_kernel(lib.kernels[0])


def _validate(points, targets, normals, indices, damping):
    arrays = [np.asarray(a) for a in (points, targets, normals)]
    p, q, normal = arrays
    for name, a in zip(("points", "targets", "normals"), arrays):
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a finite Nx3 array")
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or idx.size == 0:
        raise ValueError("indices must be a nonempty integer vector")
    if q.shape != normal.shape or len(q) != len(idx):
        raise ValueError("targets and normals must match correspondence count")
    if np.any(idx < 0) or np.any(idx >= len(p)):
        raise ValueError("point indices out of bounds")
    _validate_damping(damping)
    converted = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
    if not all(np.isfinite(a).all() for a in converted):
        raise ValueError("input values must be representable as float32")
    return (*converted, np.ascontiguousarray(idx, dtype=np.uint32))


def _validate_damping(damping):
    if (
        not np.isscalar(damping)
        or not np.isfinite(damping)
        or damping < 0
        or damping > np.finfo(np.float32).max
    ):
        raise ValueError("damping must be finite, nonnegative float32")


class GaussNewton:
    """All preparation allocations are retained across repeated dispatches.

    ``prepare`` uploads once; ``dispatch`` records residuals, hierarchical sums and
    Cholesky in one command buffer and returns GPU buffers. Only ``readback`` copies
    results to the host. A prepared problem belongs to its creating solver.
    """

    def __init__(self, runtime=None, reduction="workgroup"):
        from .emitter import emit_slang
        from .runtime import VulkanRuntime

        if reduction not in ("workgroup", "subgroup"):
            raise ValueError("reduction must be workgroup or subgroup")
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.ir = residual_kernel()
        self.residual_source = emit_slang(self.ir)
        self.residual = self.runtime.compile(self.residual_source)
        source = Path(__file__).with_name("shaders").joinpath("gn.slang").read_text()
        self.reduce = self.runtime.compile(
            source,
            entry=("reduceResidualSubgroup" if reduction == "subgroup" else "reduceResidual"),
        )
        self.reduce_partials = self.runtime.compile(source, entry="reducePartials")
        self.solve = self.runtime.compile(source, entry="solveSystem")

    def prepare(self, points, targets, normals, indices):
        from .layout import pack

        p, q, normal, idx = _validate(points, targets, normals, indices, 0)
        n = len(idx)
        arrays = dict(
            points=pack(p),
            targets=pack(q),
            normals=pack(normal),
            points_indices=idx,
            out_0=np.zeros(n, np.float32),
            out_1=pack(np.zeros((n, 6), np.float32)),
        )
        buffers = {name: self.runtime.buffer(a) for name, a in arrays.items()}
        partials = []
        count = n
        while True:
            count = (count + 63) // 64
            partials.append((count, self.runtime.buffer(np.zeros(count * 43, np.float32))))
            if count == 1:
                break
        return PreparedProblem(
            n,
            len(p),
            buffers,
            partials,
            self.runtime.buffer(np.zeros(6, np.float32)),
            self.runtime.buffer(np.zeros(1, np.uint32)),
            self,
        )

    def dispatch(self, problem, damping=1e-3, *, command_encoder=None):
        _validate_damping(damping)
        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another solver")
        encoder = (
            command_encoder
            if command_encoder is not None
            else self.runtime.device.create_command_encoder()
        )
        allocations = {
            name: problem.point_count if name == "points" else problem.count
            for name in problem.buffers
        }
        params = {"problem_size": problem.count}
        params.update(
            {
                f"{b.name}_num_alloc": allocations[b.name]
                for b in self.ir.buffers
                if b.num_alloc_param
            }
        )
        self.residual.dispatch(
            thread_count=[problem.count, 1, 1],
            vars={**problem.buffers, "params": params},
            command_encoder=encoder,
        )
        self.reduce.dispatch(
            thread_count=[problem.count, 1, 1],
            vars={
                "residual": problem.buffers["out_0"],
                "jacobian": problem.buffers["out_1"],
                "outputTerms": problem.partials[0][1],
                "params": {"count": problem.count},
            },
            command_encoder=encoder,
        )
        for (count, previous), (_, output) in zip(problem.partials, problem.partials[1:]):
            self.reduce_partials.dispatch(
                thread_count=[count, 1, 1],
                vars={"inputTerms": previous, "outputTerms": output, "params": {"count": count}},
                command_encoder=encoder,
            )
        self.solve.dispatch(
            thread_count=[1, 1, 1],
            vars={
                "inputTerms": problem.partials[-1][1],
                "delta": problem.delta,
                "status": problem.status,
                "solveParams": {"damping": float(damping)},
            },
            command_encoder=encoder,
        )
        if command_encoder is None:
            self.runtime.device.submit_command_buffer(encoder.finish())
        return problem

    def readback(self, problem):
        from .layout import unpack

        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another solver")

        def raw(buffer):
            return buffer.to_numpy().view(np.float32).reshape(-1)

        terms = raw(problem.partials[-1][1])
        return StepResult(
            raw(problem.buffers["out_0"]).copy(),
            unpack(raw(problem.buffers["out_1"]), problem.count, 6),
            terms[:36].reshape(6, 6).copy(),
            terms[36:42].copy(),
            raw(problem.delta).copy(),
            int(problem.status.to_numpy().view(np.uint32).reshape(-1)[0]),
            float(terms[42]),
        )

    def step(self, points, targets, normals, indices, damping=1e-3):
        _validate_damping(damping)
        problem = self.prepare(points, targets, normals, indices)
        self.dispatch(problem, damping)
        return self.readback(problem)


def reference(points, targets, normals, indices, damping=1e-3):
    """Independent float64 NumPy point-to-plane normal equations."""
    _validate(points, targets, normals, indices, damping)
    p = np.asarray(points, dtype=np.float64)[np.asarray(indices)]
    q = np.asarray(targets, dtype=np.float64)
    normal = np.asarray(normals, dtype=np.float64)
    residual = np.einsum("ij,ij->i", normal, p - q)
    jacobian = np.column_stack((np.cross(p, normal), normal))
    hessian = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    try:
        L = np.linalg.cholesky(hessian + damping * np.eye(6))
        delta = np.linalg.solve(L.T, np.linalg.solve(L, -gradient))
        status = 0
    except np.linalg.LinAlgError:
        delta = np.zeros(6)
        status = 1
    return dict(
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        gradient=gradient,
        delta=delta,
        status=status,
        cost=float(residual @ residual),
    )


def synthetic_problem(n, seed=41):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n + 7, 3)).astype(np.float32)
    indices = rng.integers(0, len(points), size=n, dtype=np.uint32)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals.astype(np.float32)
    targets = (points[indices] + rng.normal(scale=0.02, size=(n, 3))).astype(np.float32)
    return points, targets, normals, indices
