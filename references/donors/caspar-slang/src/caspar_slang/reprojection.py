"""Caspar reprojection derivatives and persistent GPU pose-only optimization.

Pose perturbations are left-multiplied camera-frame SE(3), ordered rotation
then translation. Landmark derivatives use the world coordinate basis.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .caspar_adapter import from_caspar_kernel
from .primitives import emit_scalar_function, primitives_source


@lru_cache(maxsize=1)
def reproject_kernel():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    R = T.Annotated[sf.Scalar, mem.ReadSequential]
    W = T.Annotated[sf.Scalar, mem.WriteSequential]
    lib = CasparLibrary(name="reprojection")

    @lib.add_kernel
    def reproject(
        x: R,
        y: R,
        z: R,
        fx: R,
        fy: R,
        cx: R,
        cy: R,
        u: R,
        v: R,
        r00: R,
        r01: R,
        r02: R,
        r10: R,
        r11: R,
        r12: R,
        r20: R,
        r21: R,
        r22: R,
    ) -> T.Tuple[W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W]:
        xi = sf.V6.symbolic("xi")
        landmark = sf.V3.symbolic("landmark_delta")
        rotation = sf.M33([r00, r01, r02, r10, r11, r12, r20, r21, r22])
        q = sf.V3(x, y, z)
        perturbed = q + sf.V3(xi[:3]).cross(q) + sf.V3(xi[3:]) + rotation * landmark
        error = sf.V2(
            fx * perturbed[0] / perturbed[2] + cx - u, fy * perturbed[1] / perturbed[2] + cy - v
        )
        zero = {value: 0 for value in [*xi, *landmark]}
        residual = [entry.subs(zero) for entry in error]
        pose_jac = [entry.diff(value).subs(zero) for entry in error for value in xi]
        landmark_jac = [entry.diff(value).subs(zero) for entry in error for value in landmark]
        return tuple(residual + pose_jac + landmark_jac)

    ir = from_caspar_kernel(lib.kernels[0])
    ir.metadata.update(
        schedule="DABSEG/get_lines",
        register_count=str(len(ir.registers)),
        pose_basis="left camera SE3 omega_xyz translation_xyz",
        landmark_basis="world xyz",
    )
    return ir


@lru_cache(maxsize=1)
def factor_kernels():
    """Canonical geometric/photometric factors for fused dense GPU solvers."""
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    R = T.Annotated[sf.Scalar, mem.ReadSequential]
    W = T.Annotated[sf.Scalar, mem.WriteSequential]
    lib = CasparLibrary(name="dense_factors")

    @lib.add_kernel
    def point_plane(
        px: R, py: R, pz: R, qx: R, qy: R, qz: R, nx: R, ny: R, nz: R
    ) -> T.Tuple[W, W, W, W, W, W, W]:
        point = sf.V3(px, py, pz)
        xi = sf.V6.symbolic("xi")
        moved = point + sf.V3(xi[:3]).cross(point) + sf.V3(xi[3:])
        error = sf.V3(nx, ny, nz).dot(moved - sf.V3(qx, qy, qz))
        zero = {v: 0 for v in xi}
        return tuple([error.subs(zero)] + [error.diff(v).subs(zero) for v in xi])

    @lib.add_kernel
    def photo_factor(
        x: R, y: R, z: R, fx: R, fy: R, gx: R, gy: R, targetI: R, sourceI: R
    ) -> T.Tuple[W, W, W, W, W, W, W]:
        point = sf.V3(x, y, z)
        xi = sf.V6.symbolic("xi")
        moved = point + sf.V3(xi[:3]).cross(point) + sf.V3(xi[3:])
        error = (
            targetI
            - sourceI
            + gx * fx * (moved[0] / moved[2] - x / z)
            + gy * fy * (moved[1] / moved[2] - y / z)
        )
        zero = {v: 0 for v in xi}
        return tuple([error.subs(zero)] + [error.diff(v).subs(zero) for v in xi])

    @lib.add_kernel
    def dense_factor(
        wx: R,
        wy: R,
        wz: R,
        x: R,
        y: R,
        z: R,
        tr00: R,
        tr01: R,
        tr02: R,
        tr10: R,
        tr11: R,
        tr12: R,
        tr20: R,
        tr21: R,
        tr22: R,
        sr00: R,
        sr01: R,
        sr02: R,
        sr10: R,
        sr11: R,
        sr12: R,
        sr20: R,
        sr21: R,
        sr22: R,
        rx: R,
        ry: R,
        rz: R,
        rho: R,
        qx: R,
        qy: R,
        qz: R,
        nx: R,
        ny: R,
        nz: R,
        targetIntensity: R,
        sourceIntensity: R,
        gx: R,
        gy: R,
        fx: R,
        fy: R,
    ) -> T.Tuple[W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W]:
        target_rotation = sf.M33([tr00, tr01, tr02, tr10, tr11, tr12, tr20, tr21, tr22])
        source_rotation = sf.M33([sr00, sr01, sr02, sr10, sr11, sr12, sr20, sr21, sr22])
        xi = sf.V6.symbolic("xi")
        drho = sf.Symbol("delta_rho")
        world_point = sf.V3(wx, wy, wz)
        change = sf.V3(xi[:3]).cross(world_point) + sf.V3(xi[3:])
        change += source_rotation * sf.V3(rx, ry, rz) * (1 / (rho + drho) - 1 / rho)
        moved = sf.V3(x, y, z) + target_rotation * change
        geometric = sf.V3(nx, ny, nz).dot(moved - sf.V3(qx, qy, qz))
        photometric = (
            targetIntensity
            - sourceIntensity
            + gx * fx * (moved[0] / moved[2] - x / z)
            + gy * fy * (moved[1] / moved[2] - y / z)
        )
        zero = {v: 0 for v in [*xi, drho]}
        values = [geometric.subs(zero), photometric.subs(zero)]
        values += [geometric.diff(v).subs(zero) for v in xi]
        values += [photometric.diff(v).subs(zero) for v in xi]
        values += [geometric.diff(drho).subs(zero), photometric.diff(drho).subs(zero)]
        return tuple(values)

    kernels = {kernel.name: from_caspar_kernel(kernel) for kernel in lib.kernels}
    for ir in kernels.values():
        ir.metadata.update(
            schedule="DABSEG/get_lines",
            register_count=str(len(ir.registers)),
            source_pose_sign="positive",
            target_pose_sign="negative",
        )
    kernels["dense_factor"].metadata.update(
        pose_basis="left world SE3 omega_xyz translation_xyz", depth_basis="source inverse z depth"
    )
    return kernels


@lru_cache(maxsize=1)
def dense_factor_source():
    """Self-contained dense factors; runtime supplies sampled fixed observations."""
    return (
        primitives_source()
        + "\n"
        + "\n".join(
            emit_scalar_function(ir, "caspar_" + name) for name, ir in factor_kernels().items()
        )
    )


@lru_cache(maxsize=1)
def reproject_source():
    """Self-contained geometry helpers and 20-output calibrated residual function.

    ``caspar_reproject(x,y,z,fx,fy,cx,cy,u,v,r00..r22,out out_0..out_19)``:
    residual[2], row-major pose Jacobian[2,6], landmark Jacobian[2,3].
    x/y/z are already transformed camera-frame coordinates; R is camera_from_world.
    """
    return (
        dense_factor_source() + "\n" + emit_scalar_function(reproject_kernel(), "caspar_reproject")
    )


@lru_cache(maxsize=1)
def optimizer_source():
    return (
        reproject_source()
        + "\n"
        + Path(__file__).with_name("shaders").joinpath("reprojection.slang").read_text()
    )


def reference(points, observations, pose, intrinsics):
    """Independent float64 calibrated projection and analytical derivatives."""
    p = np.asarray(points, dtype=np.float64)
    measured = np.asarray(observations, dtype=np.float64)
    transform = np.asarray(pose, dtype=np.float64)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or measured.shape != (len(p), 2)
        or transform.shape != (4, 4)
        or calibration.shape != (4,)
        or not all(np.isfinite(a).all() for a in (p, measured, transform, calibration))
        or np.any(calibration[:2] <= 0)
    ):
        raise ValueError(
            "finite Nx3 points, Nx2 observations, 4x4 pose and calibrated intrinsics required"
        )
    q = p @ transform[:3, :3].T + transform[:3, 3]
    if np.any(q[:, 2] <= 0):
        raise ValueError("reference points must be in front of the camera")
    x, y, z = q.T
    fx, fy, cx, cy = calibration
    residual = np.column_stack((fx * x / z + cx, fy * y / z + cy)) - measured
    projection = np.zeros((len(p), 2, 3))
    projection[:, 0, 0] = fx / z
    projection[:, 0, 2] = -fx * x / z**2
    projection[:, 1, 1] = fy / z
    projection[:, 1, 2] = -fy * y / z**2
    skew = np.zeros((len(p), 3, 3))
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    pose_jac = np.concatenate((projection @ (-skew), projection), axis=2)
    landmark_jac = projection @ transform[:3, :3]
    return residual, pose_jac, landmark_jac


@dataclass
class ReprojectionProblem:
    max_count: int
    partials: list
    delta: object
    solve_status: object
    stats: object
    cost: object
    owner: object

    def reset(self, encoder):
        """Clear persistent failure once at frame start; pose restoration is caller-owned."""
        self.owner.reset(self, encoder)


class ReprojectionGN:
    """Record GPU reprojection, normal equations, solve and SE(3) update.

    Inputs are fixed-capacity AoS point Nx3 / pixel Nx2 float32 arrays and uint32
    masks. Zero excludes a point. Call ``problem.reset(encoder)`` once per frame,
    then ``record`` repeatedly. Failure is sticky until reset; no host work,
    allocations, transfers or submissions occur while recording. The pose is a
    row-major 16-float camera_from_world rigid transform. stats is uint[2]
    (valid matches, failure); cost is float[1] weighted squared pixel error.
    """

    def __init__(self, runtime=None, intrinsics=(1.0, 1.0, 0.0, 0.0)):
        from .runtime import VulkanRuntime

        c = np.asarray(intrinsics, dtype=np.float64)
        if (
            c.shape != (4,)
            or not np.isfinite(c).all()
            or np.any(c[:2] <= 0)
            or np.max(np.abs(c)) > np.finfo(np.float32).max
        ):
            raise ValueError("intrinsics must be finite fx/fy/cx/cy with positive focal lengths")
        self.intrinsics = tuple(float(v) for v in c)
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        source = optimizer_source()
        self.kernels = {
            name: self.runtime.compile(source, entry=name)
            for name in (
                "resetReprojection",
                "resetReprojectionCount",
                "buildReprojection",
                "updateReprojection",
                "evaluateReprojection",
            )
        }
        reduction = Path(__file__).with_name("shaders").joinpath("gn.slang").read_text()
        self.reduce_partials = self.runtime.compile(reduction, entry="reducePartials")
        self.solve = self.runtime.compile(reduction, entry="solveSystem")

    def prepare(self, max_count):
        if (
            not isinstance(max_count, int)
            or isinstance(max_count, bool)
            or not 1 <= max_count <= 2**20
        ):
            raise ValueError("max_count must be a positive integer no larger than 2**20")
        # One fused contribution group per 64 matches: residual/Jacobian
        # vectors never leave the producing lane's registers.
        partials = []
        count = max_count
        while True:
            count = (count + 63) // 64
            partials.append((count, self.runtime.buffer(np.zeros(count * 43, np.float32))))
            if count == 1:
                break
        return ReprojectionProblem(
            max_count,
            partials,
            self.runtime.buffer(np.zeros(6, np.float32)),
            self.runtime.buffer(np.zeros(1, np.uint32)),
            self.runtime.buffer(np.zeros(2, np.uint32)),
            self.runtime.buffer(np.zeros(1, np.float32)),
            self,
        )

    def reset(self, problem, encoder):
        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another solver")
        self.kernels["resetReprojection"].dispatch(
            thread_count=[1, 1, 1],
            vars={"stats": problem.stats, "cost": problem.cost},
            command_encoder=encoder,
        )
        encoder.global_barrier()

    def record(
        self,
        problem,
        points_buffer,
        uv_buffer,
        mask_buffer,
        pose_buffer,
        encoder,
        damping=1e-4,
        huber=3.0,
        min_matches=6,
    ):
        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another solver")
        if (
            not np.isfinite([damping, huber]).all()
            or damping < 0
            or huber <= 0
            or max(damping, huber) > np.finfo(np.float32).max
        ):
            raise ValueError(
                "damping must be nonnegative and Huber threshold positive finite float32"
            )
        if (
            not isinstance(min_matches, int)
            or isinstance(min_matches, bool)
            or not 4 <= min_matches <= 0xFFFFFFFF
        ):
            raise ValueError("min_matches must be an integer of at least four")
        params = self._record_factors(
            problem, points_buffer, uv_buffer, mask_buffer, pose_buffer, encoder, huber, min_matches
        )
        self.solve.dispatch(
            thread_count=[1, 1, 1],
            vars={
                "inputTerms": problem.partials[-1][1],
                "delta": problem.delta,
                "status": problem.solve_status,
                "solveParams": {"damping": damping},
            },
            command_encoder=encoder,
        )
        encoder.global_barrier()
        self.kernels["updateReprojection"].dispatch(
            thread_count=[1, 1, 1],
            vars={
                "pose": pose_buffer,
                "delta": problem.delta,
                "solveStatus": problem.solve_status,
                "stats": problem.stats,
                "cost": problem.cost,
                "normalTerms": problem.partials[-1][1],
                "reprojectionParams": params,
            },
            command_encoder=encoder,
        )
        encoder.global_barrier()

    def _record_factors(
        self,
        problem,
        points_buffer,
        uv_buffer,
        mask_buffer,
        pose_buffer,
        encoder,
        huber,
        min_matches,
    ):
        if problem.owner is not self:
            raise ValueError("prepared problem belongs to another solver")
        if not np.isfinite(huber) or not 0 < huber <= np.finfo(np.float32).max:
            raise ValueError("Huber threshold must be positive finite float32")
        if (
            not isinstance(min_matches, int)
            or isinstance(min_matches, bool)
            or not 4 <= min_matches <= 0xFFFFFFFF
        ):
            raise ValueError("min_matches must be an integer of at least four")
        fx, fy, cx, cy = self.intrinsics
        params = dict(
            count=problem.max_count, fx=fx, fy=fy, cx=cx, cy=cy, huber=huber, minMatches=min_matches
        )
        self.kernels["resetReprojectionCount"].dispatch(
            thread_count=[1, 1, 1], vars={"stats": problem.stats}, command_encoder=encoder
        )
        encoder.global_barrier()
        self.kernels["buildReprojection"].dispatch(
            thread_count=[problem.max_count, 1, 1],
            vars=dict(
                worldPoints=points_buffer,
                observations=uv_buffer,
                validMask=mask_buffer,
                pose=pose_buffer,
                outputTerms=problem.partials[0][1],
                stats=problem.stats,
                reprojectionParams=params,
            ),
            command_encoder=encoder,
        )
        encoder.global_barrier()
        for (count, previous), (_, output) in zip(problem.partials, problem.partials[1:]):
            self.reduce_partials.dispatch(
                thread_count=[count, 1, 1],
                vars={"inputTerms": previous, "outputTerms": output, "params": {"count": count}},
                command_encoder=encoder,
            )
            encoder.global_barrier()
        return params

    def record_evaluate(
        self,
        problem,
        points_buffer,
        uv_buffer,
        mask_buffer,
        pose_buffer,
        encoder,
        huber=3.0,
        min_matches=6,
    ):
        """Evaluate final-pose weighted pixel cost/count without solve or mutation.

        This retains sticky failure semantics, and is required when a caller's
        acceptance test must judge the pose after the last optimizer update.
        """
        params = self._record_factors(
            problem, points_buffer, uv_buffer, mask_buffer, pose_buffer, encoder, huber, min_matches
        )
        self.kernels["evaluateReprojection"].dispatch(
            thread_count=[1, 1, 1],
            vars={
                "stats": problem.stats,
                "cost": problem.cost,
                "normalTerms": problem.partials[-1][1],
                "reprojectionParams": params,
            },
            command_encoder=encoder,
        )
        encoder.global_barrier()
