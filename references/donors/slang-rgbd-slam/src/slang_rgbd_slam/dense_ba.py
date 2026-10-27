"""GPU coupled dense pose/inverse-depth bundle adjustment for RGB-D keyframes.

The first world-from-camera pose anchors the window. Each source pixel owns one
inverse-depth variable. Its observations against all other frames share that
variable and couple both camera poses before Schur elimination. The CPU helper
below is an independent linear-algebra oracle, never used by GPU processing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import Camera


def schur_step(jpose, jdepth, residual, depth_prior, *, depth_weight=10.0, damping=1e-4):
    jpose, jdepth, residual, depth_prior = (
        np.asarray(a, np.float64) for a in (jpose, jdepth, residual, depth_prior)
    )
    if (
        jpose.ndim != 3
        or jdepth.shape != jpose.shape[:2]
        or residual.shape != jdepth.shape
        or depth_prior.shape != (len(jpose),)
        or not 1 <= jpose.shape[2] <= 18
        or any(not np.isfinite(a).all() for a in (jpose, jdepth, residual, depth_prior))
        or not np.isfinite([depth_weight, damping]).all()
        or depth_weight <= 0
        or damping < 0
    ):
        raise ValueError("invalid dense BA linearization")
    hpp = np.einsum("nki,nkj->ij", jpose, jpose)
    bp = np.einsum("nki,nk->i", jpose, residual)
    hpd = np.einsum("nki,nk->ni", jpose, jdepth)
    hdd = np.sum(jdepth * jdepth, axis=1) + depth_weight**2 + damping
    bd = np.sum(jdepth * residual, axis=1) + depth_weight**2 * depth_prior
    schur = hpp - np.einsum("ni,nj,n->ij", hpd, hpd, 1 / hdd) + damping * np.eye(jpose.shape[2])
    rhs = bp - np.einsum("ni,n->i", hpd, bd / hdd)
    delta_pose = np.linalg.solve(schur, -rhs)
    delta_depth = -(bd + hpd @ delta_pose) / hdd
    return delta_pose, delta_depth


def dense_factor_reference(
    source_pose,
    target_pose,
    ray,
    rho,
    target_point,
    target_normal,
    source_intensity,
    target_intensity,
    gradient,
    camera,
    *,
    linearization_pixel=None,
):
    """Independent raw geometric/photometric factor and source/target/depth Jacobians.

    ``target_intensity`` is measured at ``linearization_pixel``. Keeping that
    pixel fixed permits central differences of the local bilinear image model.
    Returned pose Jacobians use world left increments, source first, target next.
    """
    from .cpu import _pose

    source_pose, target_pose = _pose(source_pose), _pose(target_pose)
    ray, target_point, target_normal, gradient = (
        np.asarray(value, np.float64) for value in (ray, target_point, target_normal, gradient)
    )
    if (
        ray.shape != (3,)
        or target_point.shape != (3,)
        or target_normal.shape != (3,)
        or gradient.shape != (2,)
        or not np.isfinite(rho)
        or rho <= 0
    ):
        raise ValueError("invalid dense factor geometry")
    world = source_pose[:3, :3] @ (ray / rho) + source_pose[:3, 3]
    point = target_pose[:3, :3].T @ (world - target_pose[:3, 3])
    if point[2] <= 0:
        raise ValueError("factor must lie in front of target camera")
    pixel = point[:2] / point[2] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    base = pixel if linearization_pixel is None else np.asarray(linearization_pixel, np.float64)
    residual = np.array(
        [
            target_normal @ (point - target_point),
            target_intensity - source_intensity + gradient @ (pixel - base),
        ]
    )
    photo_normal = np.array(
        [
            gradient[0] * camera.fx / point[2],
            gradient[1] * camera.fy / point[2],
            -(gradient[0] * camera.fx * point[0] + gradient[1] * camera.fy * point[1])
            / point[2] ** 2,
        ]
    )
    normals = np.stack((target_normal, photo_normal))
    world_normals = normals @ target_pose[:3, :3].T
    positive = np.concatenate((np.cross(world, world_normals), world_normals), axis=1)
    jpose = np.concatenate((positive, -positive), axis=1)
    inverse_depth_direction = target_pose[:3, :3].T @ source_pose[:3, :3] @ (-ray / rho**2)
    return residual, jpose, normals @ inverse_depth_direction


@dataclass
class DenseBAProblem:
    buffers: dict
    partials: list
    count: int

    @property
    def poses(self):
        return self.buffers["poses"]

    @property
    def inverse_depth(self):
        return self.buffers["inverseDepth"]


class DenseBA:
    """Bounded W<=4 dense keyframe optimizer; recording performs no readback."""

    def __init__(
        self,
        runtime,
        camera,
        window=4,
        *,
        depth_weight=10.0,
        geo_weight=10.0,
        photo_weight=1 / 255,
        damping=1e-4,
        distance_threshold=0.15,
        depth_jump=0.1,
        min_factor_ratio=0.8,
    ):
        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if not isinstance(window, int) or not 2 <= window <= 4:
            raise ValueError("dense BA window must contain two to four frames")
        if (
            not np.isfinite(
                [
                    depth_weight,
                    geo_weight,
                    photo_weight,
                    damping,
                    distance_threshold,
                    depth_jump,
                    min_factor_ratio,
                ]
            ).all()
            or min(depth_weight, geo_weight, photo_weight, distance_threshold, depth_jump) <= 0
            or damping < 0
            or not 0 <= min_factor_ratio <= 1
        ):
            raise ValueError("invalid dense BA configuration")
        from caspar_slang.reprojection import dense_factor_source

        self.runtime, self.camera, self.window = runtime, camera, window
        self.pose_dofs = 6 * (window - 1)
        self.terms = self.pose_dofs**2 + self.pose_dofs + 2
        self.options = dict(
            depthWeight=depth_weight,
            geoWeight=geo_weight,
            photoWeight=photo_weight,
            damping=damping,
            distance=distance_threshold,
            depthJump=depth_jump,
            minFactorRatio=min_factor_ratio,
        )
        source = (
            dense_factor_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("dense_ba.slang").read_text()
        )
        self.kernels = {
            name: runtime.compile(source, entry=name)
            for name in (
                "initializeDense",
                "linearizeSchur",
                "reduceSchur",
                "solveSchur",
                "updateDensePoses",
                "commitDensePoses",
                "updateDenseDepth",
                "saveInitialCost",
                "latchDenseFailure",
                "finalizeDense",
                "rollbackDense",
            )
        }

    def _buffer(self, count, data=None):
        import slangpy as spy

        buffer = self.runtime.device.create_buffer(
            element_count=count,
            struct_size=4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )
        if data is not None:
            buffer.copy_from_numpy(np.ascontiguousarray(data).ravel())
        return buffer

    def prepare(self, depths=None, rgb=None, poses=None, *, initial_depths=None):
        """Allocate one persistent window; optional uploads are outside GPU iteration.

        Immutable observed frame-major depth metres, RGB 0..255, and row-major world-from-camera
        poses. initial_depths supplies the current mutable estimate for rho initialization
        and defaults to the observations. Callers can instead copy GPU keyframe buffers into these
        named buffers without downloading image or pose arrays.
        """
        height, width = self.camera.height, self.camera.width
        count = self.window * height * width
        depths = (
            np.zeros((self.window, height, width), np.float32)
            if depths is None
            else np.asarray(depths, np.float32)
        )
        initial_depths = (
            depths.copy() if initial_depths is None else np.asarray(initial_depths, np.float32)
        )
        rgb = (
            np.zeros((self.window, height, width, 3), np.float32)
            if rgb is None
            else np.asarray(rgb, np.float32)
        )
        poses = (
            np.repeat(np.eye(4, dtype=np.float32)[None], self.window, axis=0)
            if poses is None
            else np.asarray(poses, np.float32)
        )
        if (
            depths.shape != (self.window, height, width)
            or initial_depths.shape != depths.shape
            or not np.isfinite(initial_depths).all()
            or np.any(initial_depths < 0)
            or rgb.shape != (*depths.shape, 3)
            or poses.shape != (self.window, 4, 4)
            or not np.isfinite(depths).all()
            or np.any(depths < 0)
            or not np.isfinite(rgb).all()
            or np.any(rgb < 0)
            or np.any(rgb > 255)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("invalid dense BA window arrays")
        from .cpu import _pose

        for pose in poses:
            _pose(pose)
        buffers = dict(
            depths=self._buffer(count, depths),
            initialDepths=self._buffer(count, initial_depths),
            rgb=self._buffer(count * 3, rgb),
            poses=self._buffer(self.window * 16, poses),
            candidatePoses=self._buffer(self.window * 16),
            backupPoses=self._buffer(self.window * 16),
            backupDepth=self._buffer(count),
            costs=self._buffer(4),
            transactionStatus=self._buffer(1),
            gray=self._buffer(count),
            normals=self._buffer(count * 4),
            inverseDepth=self._buffer(count),
            hpd=self._buffer(count * self.pose_dofs),
            hdd=self._buffer(count),
            bd=self._buffer(count),
            delta=self._buffer(self.pose_dofs),
            status=self._buffer(1, np.zeros(1, np.uint32)),
        )
        partials = []
        remaining = count
        while True:
            groups = (remaining + 63) // 64
            partials.append((groups, self._buffer(groups * self.terms)))
            if groups == 1:
                break
            remaining = groups
        return DenseBAProblem(buffers, partials, count)

    def _params(self, problem, count=None, active_frames=None):
        camera = self.camera
        return dict(
            width=camera.width,
            height=camera.height,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            frames=self.window,
            activeFrames=self.window if active_frames is None else active_frames,
            dofs=self.pose_dofs,
            terms=self.terms,
            count=problem.count if count is None else count,
            **self.options,
        )

    def _dispatch(self, entry, count, encoder, params, buffers):
        self.kernels[entry].dispatch(
            thread_count=[count, 1, 1], vars=dict(cfg=params, **buffers), command_encoder=encoder
        )

    def record_evaluate(self, problem, encoder, *, active_frames=None):
        """Recompute current pose/depth cost and Schur terms without any update."""
        params = self._params(problem, active_frames=active_frames)
        buffers = problem.buffers
        names = ("depths", "gray", "normals", "poses", "inverseDepth", "hpd", "hdd", "bd")
        self._dispatch(
            "linearizeSchur",
            problem.count,
            encoder,
            params,
            {**{k: buffers[k] for k in names}, "outputTerms": problem.partials[0][1]},
        )
        encoder.global_barrier()
        for (input_count, input_buffer), (output_count, output_buffer) in zip(
            problem.partials, problem.partials[1:]
        ):
            self._dispatch(
                "reduceSchur",
                output_count * 64,
                encoder,
                self._params(problem, input_count, active_frames),
                dict(inputTerms=input_buffer, outputTerms=output_buffer),
            )
            encoder.global_barrier()

    def record(self, problem, encoder, iterations=3, *, active_frames=None):
        """GPU transaction: accept only finite nonincreasing final cost, else rollback.

        `costs[0:2]` records actual before/after residual costs and `costs[2:4]`
        the corresponding scalar factor counts. Acceptance retains the configured
        minimum fraction of initial factors. A status of zero
        commits all poses/inverse depths; nonzero restores the complete window.
        All decisions, backups, equations, solves and restoration remain on GPU.
        """
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be positive")
        if active_frames is not None and (
            not isinstance(active_frames, int) or not 2 <= active_frames <= self.window
        ):
            raise ValueError("active_frames must be between two and window size")
        params = self._params(problem, active_frames=active_frames)
        b = problem.buffers
        self._dispatch(
            "initializeDense",
            problem.count,
            encoder,
            params,
            {
                k: b[k]
                for k in ("depths", "initialDepths", "rgb", "gray", "normals", "inverseDepth")
            },
        )
        encoder.global_barrier()
        encoder.copy_buffer(b["backupPoses"], 0, b["poses"], 0, self.window * 16 * 4)
        encoder.copy_buffer(b["backupDepth"], 0, b["inverseDepth"], 0, problem.count * 4)
        encoder.global_barrier()
        for iteration in range(iterations):
            self.record_evaluate(problem, encoder, active_frames=active_frames)
            if iteration == 0:
                self._dispatch(
                    "saveInitialCost",
                    1,
                    encoder,
                    params,
                    dict(
                        inputTerms=problem.partials[-1][1],
                        costs=b["costs"],
                        transactionStatus=b["transactionStatus"],
                    ),
                )
                encoder.global_barrier()
            self._dispatch(
                "solveSchur",
                1,
                encoder,
                params,
                dict(inputTerms=problem.partials[-1][1], delta=b["delta"], status=b["status"]),
            )
            encoder.global_barrier()
            self._dispatch(
                "updateDensePoses",
                self.window,
                encoder,
                params,
                dict(
                    poses=b["poses"],
                    candidatePoses=b["candidatePoses"],
                    delta=b["delta"],
                    status=b["status"],
                ),
            )
            encoder.global_barrier()
            self._dispatch(
                "commitDensePoses",
                self.window * 16,
                encoder,
                params,
                dict(poses=b["poses"], candidatePoses=b["candidatePoses"], status=b["status"]),
            )
            self._dispatch(
                "updateDenseDepth",
                problem.count,
                encoder,
                params,
                {
                    k: b[k]
                    for k in ("depths", "inverseDepth", "hpd", "hdd", "bd", "delta", "status")
                },
            )
            self._dispatch(
                "latchDenseFailure",
                1,
                encoder,
                params,
                dict(status=b["status"], transactionStatus=b["transactionStatus"]),
            )
            encoder.global_barrier()
        self.record_evaluate(problem, encoder, active_frames=active_frames)
        self._dispatch(
            "finalizeDense",
            1,
            encoder,
            params,
            dict(
                inputTerms=problem.partials[-1][1],
                costs=b["costs"],
                transactionStatus=b["transactionStatus"],
                status=b["status"],
            ),
        )
        encoder.global_barrier()
        self._dispatch(
            "rollbackDense",
            max(problem.count, self.window * 16),
            encoder,
            params,
            {k: b[k] for k in ("backupPoses", "backupDepth", "poses", "inverseDepth", "status")},
        )
        encoder.global_barrier()

    def solve(self, problem, iterations=3, *, active_frames=None):
        """Convenience submission; explicit caller readback remains optional."""
        encoder = self.runtime.device.create_command_encoder()
        self.record(problem, encoder, iterations, active_frames=active_frames)
        self.runtime.device.submit_command_buffer(encoder.finish())
