"""GPU projective point-to-plane ICP, with Caspar derivatives and GPU SE(3) updates."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (
            not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 4
            or self.height < 4
            or not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all()
            or self.fx <= 0
            or self.fy <= 0
        ):
            raise ValueError(
                "camera requires positive dimensions and finite positive focal lengths"
            )


@dataclass
class TrackingResult:
    pose: np.ndarray
    success: bool
    correspondences: int
    rmse: float
    iterations: int
    elapsed_ms: float

    @property
    def reference_from_current(self):
        return self.pose


class VulkanTracker:
    """Fixed iterations (coarsest first) in one submission per tracked frame.

    Depth is float32 metres, zero marks missing depth. RMSE is Huber-weighted
    point-to-plane error immediately before the final pose update. A failed
    frame restores the supplied initial pose, even after earlier valid updates.
    """

    def __init__(
        self,
        camera,
        levels=3,
        iterations=(8, 5, 4),
        *,
        runtime=None,
        distance_threshold=0.1,
        normal_threshold=0.7,
        huber_delta=0.02,
        damping=1e-4,
        min_correspondences=20,
        depth_jump=0.1,
        _compile_association=True,
        _reduction="subgroup",
    ):
        from caspar_slang.gn import GaussNewton
        from caspar_slang.runtime import VulkanRuntime

        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if (
            not isinstance(levels, int)
            or levels < 1
            or len(iterations) != levels
            or any(not isinstance(n, int) or n < 1 for n in iterations)
        ):
            raise ValueError("iterations must contain one positive integer per pyramid level")
        if camera.width % 2 ** (levels - 1) or camera.height % 2 ** (levels - 1):
            raise ValueError("dimensions must be divisible by the pyramid scale")
        if min(camera.width, camera.height) // 2 ** (levels - 1) < 4:
            raise ValueError("coarsest camera must have at least four pixels per dimension")
        values = [distance_threshold, normal_threshold, huber_delta, damping, depth_jump]
        if (
            not np.isfinite(values).all()
            or distance_threshold <= 0
            or huber_delta <= 0
            or damping < 0
            or depth_jump <= 0
            or not 0 <= normal_threshold <= 1
            or not isinstance(min_correspondences, int)
            or min_correspondences < 6
        ):
            raise ValueError("invalid ICP filtering, damping, or correspondence parameters")
        self.camera, self.iterations = camera, tuple(iterations)
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.gn = GaussNewton(self.runtime, reduction=_reduction)
        self.damping = damping
        self.distance, self.normal_cos, self.huber = (
            distance_threshold,
            normal_threshold,
            huber_delta,
        )
        self.minimum, self.depth_jump = min_correspondences, depth_jump
        from caspar_slang.primitives import primitives_source

        source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("tracker.slang").read_text()
        )
        self.kernels = {
            name: self.runtime.compile(source, entry=name)
            for name in (
                "downsample",
                "makeVertices",
                "makeNormals",
                "resetCount",
                "associate",
                "updatePose",
            )
            if name != "associate" or _compile_association
        }
        self.levels = []
        for level in range(levels):
            scale = 2**level
            cam = Camera(
                camera.width // scale,
                camera.height // scale,
                camera.fx / scale,
                camera.fy / scale,
                (camera.cx + 0.5) / scale - 0.5,
                (camera.cy + 0.5) / scale - 0.5,
            )
            count = cam.width * cam.height
            maps = {
                role: {
                    "depth": self.runtime.buffer(np.zeros(count, np.float32)),
                    "vertices": self.runtime.buffer(np.zeros(count * 4, np.float32)),
                    "normals": self.runtime.buffer(np.zeros(count * 4, np.float32)),
                }
                for role in ("reference", "source")
            }
            zeros = np.zeros((count, 3), np.float32)
            self.levels.append(
                dict(
                    camera=cam,
                    maps=maps,
                    problem=self.gn.prepare(zeros, zeros, zeros, np.arange(count, dtype=np.uint32)),
                )
            )
        self.pose = self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
        self.initial_pose = self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
        self.status = self.runtime.buffer(np.zeros(2, np.uint32))
        self.diagnostics = self.runtime.buffer(np.zeros(1, np.float32))
        self.has_reference = False

    def _depth(self, depth):
        array = np.asarray(depth)
        if array.dtype != np.float32 or array.shape != (self.camera.height, self.camera.width):
            raise ValueError("depth must be float32 HxW matching camera")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("depth must contain finite nonnegative metres")
        return np.ascontiguousarray(array).ravel()

    def _map_params(self, camera):
        return dict(
            width=camera.width,
            height=camera.height,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            depthJump=self.depth_jump,
        )

    def _record_maps(self, role, encoder):
        for i, level in enumerate(self.levels):
            camera, maps = level["camera"], level["maps"][role]
            params = self._map_params(camera)
            threads = [camera.width, camera.height, 1]
            if i:
                self.kernels["downsample"].dispatch(
                    thread_count=threads,
                    vars=dict(
                        inputDepth=self.levels[i - 1]["maps"][role]["depth"],
                        outputDepth=maps["depth"],
                        mapParams=params,
                    ),
                    command_encoder=encoder,
                )
            self.kernels["makeVertices"].dispatch(
                thread_count=threads,
                vars=dict(inputDepth=maps["depth"], vertices=maps["vertices"], mapParams=params),
                command_encoder=encoder,
            )
            self.kernels["makeNormals"].dispatch(
                thread_count=threads,
                vars=dict(vertices=maps["vertices"], normals=maps["normals"], mapParams=params),
                command_encoder=encoder,
            )

    def set_reference(self, depth):
        self.levels[0]["maps"]["reference"]["depth"].copy_from_numpy(self._depth(depth))
        encoder = self.runtime.device.create_command_encoder()
        self._record_maps("reference", encoder)
        self.runtime.device.submit_command_buffer(encoder.finish())
        self.has_reference = True

    def _record_track(self, command_encoder):
        self._record_maps("source", command_encoder)
        for level, iterations in zip(reversed(self.levels), self.iterations):
            camera, problem = level["camera"], level["problem"]
            params = dict(
                count=problem.count,
                width=camera.width,
                height=camera.height,
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                distance=self.distance,
                normalCos=self.normal_cos,
                huber=self.huber,
                minCorrespondences=self.minimum,
            )
            source, reference = level["maps"]["source"], level["maps"]["reference"]
            for _ in range(iterations):
                self.kernels["resetCount"].dispatch(
                    thread_count=[1, 1, 1],
                    vars=dict(trackingStatus=self.status),
                    command_encoder=command_encoder,
                )
                self.kernels["associate"].dispatch(
                    thread_count=[problem.count, 1, 1],
                    vars=dict(
                        sourceVertices=source["vertices"],
                        sourceNormals=source["normals"],
                        referenceVertices=reference["vertices"],
                        referenceNormals=reference["normals"],
                        points=problem.buffers["points"],
                        targets=problem.buffers["targets"],
                        weightedNormals=problem.buffers["normals"],
                        pose=self.pose,
                        trackingStatus=self.status,
                        trackParams=params,
                    ),
                    command_encoder=command_encoder,
                )
                self.gn.dispatch(problem, self.damping, command_encoder=command_encoder)
                self.kernels["updatePose"].dispatch(
                    thread_count=[1, 1, 1],
                    vars=dict(
                        pose=self.pose,
                        initialPose=self.initial_pose,
                        delta=problem.delta,
                        solveStatus=problem.status,
                        normalTerms=problem.partials[-1][1],
                        trackingStatus=self.status,
                        diagnostics=self.diagnostics,
                        trackParams=params,
                    ),
                    command_encoder=command_encoder,
                )

    def track(self, depth, initial_pose=None):
        if not self.has_reference:
            raise RuntimeError("set_reference must precede track")
        data = self._depth(depth)
        pose = (
            np.eye(4, dtype=np.float32)
            if initial_pose is None
            else np.asarray(initial_pose, dtype=np.float32)
        )
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
        ):
            raise ValueError("initial_pose must be a finite rigid 4x4 transform")
        start = perf_counter()
        self.levels[0]["maps"]["source"]["depth"].copy_from_numpy(data)
        self.pose.copy_from_numpy(np.ascontiguousarray(pose).ravel())
        self.initial_pose.copy_from_numpy(np.ascontiguousarray(pose).ravel())
        self.status.copy_from_numpy(np.zeros(2, np.uint32))
        encoder = self.runtime.device.create_command_encoder()
        self._record_track(encoder)
        self.runtime.device.submit_command_buffer(encoder.finish())
        result_pose = self.pose.to_numpy().view(np.float32).reshape(4, 4).copy()
        status = self.status.to_numpy().view(np.uint32).ravel()
        cost = float(self.diagnostics.to_numpy().view(np.float32).ravel()[0])
        return TrackingResult(
            result_pose,
            status[1] == 0,
            int(status[0]),
            float(np.sqrt(cost / max(int(status[0]), 1))),
            sum(self.iterations),
            (perf_counter() - start) * 1000,
        )
