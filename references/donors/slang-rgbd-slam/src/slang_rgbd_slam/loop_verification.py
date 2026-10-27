"""GPU RGB-D geometric RANSAC and calibrated refinement for loop candidates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import Camera


def rigid_from_three(source, target):
    """Independent rigid basis alignment; rejects repeated/collinear samples."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.shape != (3, 3) or target.shape != (3, 3) or not np.isfinite([source, target]).all():
        raise ValueError("samples must be two finite 3x3 arrays")
    bases = []
    for points in (source, target):
        first, second = points[1] - points[0], points[2] - points[0]
        normal = np.cross(first, second)
        if np.linalg.norm(first) <= 1e-5 or np.linalg.norm(normal) <= 1e-5:
            raise ValueError("degenerate three-point sample")
        first, normal = first / np.linalg.norm(first), normal / np.linalg.norm(normal)
        bases.append(np.column_stack((first, np.cross(normal, first), normal)))
    result = np.eye(4)
    result[:3, :3] = bases[1] @ bases[0].T
    result[:3, 3] = target[0] - result[:3, :3] @ source[0]
    return result


def _hash(value):
    value = int(value) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def ransac_reference(source, target, matches, *, hypotheses=128, max_distance=0.05, min_matches=20):
    """CPU oracle over calibrated RGB-D landmarks, with no pose initialization."""
    matches = np.asarray(matches, np.uint32)
    a, b = np.asarray(source["landmarks"]), np.asarray(target["landmarks"])
    indices = np.flatnonzero((matches < len(b)) & (np.arange(len(matches)) < len(a)))
    indices = indices[(a[indices, 3] > 0) & (b[matches[indices], 3] > 0)]
    best, count, cost = None, 0, np.inf
    if len(indices) >= 3:
        for hypothesis in range(hypotheses):
            first = _hash(hypothesis * 3 + 1) % len(indices)
            second = _hash(hypothesis * 3 + 2) % (len(indices) - 1)
            second += second >= first
            third = _hash(hypothesis * 3 + 3) % (len(indices) - 2)
            third += third >= min(first, second)
            third += third >= max(first, second)
            selected = indices[[first, second, third]]
            try:
                pose = rigid_from_three(a[selected, :3], b[matches[selected], :3])
            except ValueError:
                continue
            transformed = a[indices, :3] @ pose[:3, :3].T + pose[:3, 3]
            squared = np.sum((transformed - b[matches[indices], :3]) ** 2, axis=1)
            inlier = squared < max_distance**2
            candidate_count, candidate_cost = int(inlier.sum()), float(squared[inlier].sum())
            if candidate_count > count or (candidate_count == count and candidate_cost < cost):
                best, count, cost = pose, candidate_count, candidate_cost
    return dict(
        success=count >= min_matches,
        inliers=count,
        target_from_source=best,
        rmse_m=float(np.sqrt(cost / max(count, 1))) if best is not None else None,
    )


@dataclass
class LoopVerificationProblem:
    buffers: dict
    refinement: object

    @property
    def pose(self):
        return self.buffers["outputPose"]

    @property
    def stats(self):
        return self.buffers["verificationStats"]

    @property
    def cost(self):
        return self.buffers["verificationCost"]


class RGBDLoopVerifier:
    """Bounded all-GPU RANSAC + calibrated 2D/3D refinement of a loop candidate."""

    def __init__(
        self,
        camera,
        runtime=None,
        max_features=1024,
        *,
        hypotheses=128,
        max_distance=0.05,
        min_matches=20,
        iterations=8,
        max_rmse=3.0,
    ):
        from caspar_slang.primitives import primitives_source
        from caspar_slang.reprojection import ReprojectionGN
        from caspar_slang.runtime import VulkanRuntime

        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if (
            not isinstance(max_features, int)
            or not 20 <= max_features <= 4096
            or not isinstance(hypotheses, int)
            or not 1 <= hypotheses <= 1024
            or not isinstance(min_matches, int)
            or not 6 <= min_matches <= max_features
            or not isinstance(iterations, int)
            or iterations < 1
            or not np.isfinite([max_distance, max_rmse]).all()
            or min(max_distance, max_rmse) <= 0
        ):
            raise ValueError("invalid loop verification configuration")
        self.camera, self.max_features = camera, max_features
        self.hypotheses, self.max_distance, self.minimum = hypotheses, max_distance, min_matches
        self.iterations, self.max_rmse = iterations, max_rmse
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.refiner = ReprojectionGN(self.runtime, (camera.fx, camera.fy, camera.cx, camera.cy))
        source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("loop_verification.slang").read_text()
        )
        self.kernels = {
            name: self.runtime.compile(source, entry=name)
            for name in (
                "resetVerification",
                "compactPairs",
                "scoreHypotheses",
                "selectHypothesis",
                "buildRefinement",
                "finishVerification",
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

    def prepare(self):
        return LoopVerificationProblem(
            dict(
                pairIndices=self._buffer(self.max_features),
                pairCount=self._buffer(1),
                hypothesisPoses=self._buffer(self.hypotheses * 16),
                hypothesisScores=self._buffer(self.hypotheses),
                hypothesisCosts=self._buffer(self.hypotheses),
                bestRigid=self._buffer(16),
                candidatePose=self._buffer(16, np.eye(4, dtype=np.float32)),
                outputPose=self._buffer(16, np.eye(4, dtype=np.float32)),
                verificationStats=self._buffer(2),
                verificationCost=self._buffer(1),
                worldPoints=self._buffer(self.max_features * 3),
                observations=self._buffer(self.max_features * 2),
                validMask=self._buffer(self.max_features),
            ),
            self.refiner.prepare(self.max_features),
        )

    def _params(self):
        return dict(
            maxFeatures=self.max_features,
            hypotheses=self.hypotheses,
            maxDistance=self.max_distance,
            minMatches=self.minimum,
            maxRMSE=self.max_rmse,
        )

    def _dispatch(self, name, count, encoder, buffers):
        self.kernels[name].dispatch(
            thread_count=[count, 1, 1],
            vars=dict(cfg=self._params(), **buffers),
            command_encoder=encoder,
        )

    def record(self, problem, source_set, target_set, matches, target_world_pose, encoder):
        """Verify current->keyframe matches; output camera_from_world only on success.

        Failure leaves the previously committed output pose unchanged. All
        compaction, hypotheses, score selection, refinement, and commit are GPU
        operations. No feature arrays, pose matrices, or equations return to CPU.
        """
        b = problem.buffers
        self._dispatch(
            "resetVerification",
            1,
            encoder,
            {k: b[k] for k in ("pairCount", "verificationStats", "verificationCost")},
        )
        encoder.global_barrier()
        shared = dict(
            sourceLandmarks=source_set["landmarks"],
            targetLandmarks=target_set["landmarks"],
            sourceCount=source_set["count"],
            targetCount=target_set["count"],
            matches=matches,
        )
        self._dispatch(
            "compactPairs",
            self.max_features,
            encoder,
            {**shared, "pairIndices": b["pairIndices"], "pairCount": b["pairCount"]},
        )
        encoder.global_barrier()
        self._dispatch(
            "scoreHypotheses",
            self.hypotheses * 64,
            encoder,
            {
                **shared,
                **{
                    k: b[k]
                    for k in (
                        "pairIndices",
                        "pairCount",
                        "hypothesisPoses",
                        "hypothesisScores",
                        "hypothesisCosts",
                    )
                },
            },
        )
        encoder.global_barrier()
        self._dispatch(
            "selectHypothesis",
            1,
            encoder,
            {
                **{
                    k: b[k]
                    for k in (
                        "hypothesisPoses",
                        "hypothesisScores",
                        "hypothesisCosts",
                        "bestRigid",
                        "candidatePose",
                        "verificationStats",
                    )
                },
                "targetWorldPose": target_world_pose,
            },
        )
        encoder.global_barrier()
        self._dispatch(
            "buildRefinement",
            self.max_features,
            encoder,
            {
                **shared,
                "sourcePositions": source_set["positions"],
                "targetWorldPose": target_world_pose,
                **{
                    k: b[k]
                    for k in (
                        "bestRigid",
                        "verificationStats",
                        "worldPoints",
                        "observations",
                        "validMask",
                    )
                },
            },
        )
        encoder.global_barrier()
        self.refiner.reset(problem.refinement, encoder)
        for _ in range(self.iterations):
            self.refiner.record(
                problem.refinement,
                b["worldPoints"],
                b["observations"],
                b["validMask"],
                b["candidatePose"],
                encoder,
                min_matches=self.minimum,
            )
        self.refiner.record_evaluate(
            problem.refinement,
            b["worldPoints"],
            b["observations"],
            b["validMask"],
            b["candidatePose"],
            encoder,
            min_matches=self.minimum,
        )
        encoder.global_barrier()
        self._dispatch(
            "finishVerification",
            1,
            encoder,
            {
                **{
                    k: b[k]
                    for k in (
                        "candidatePose",
                        "outputPose",
                        "verificationStats",
                        "verificationCost",
                    )
                },
                "refinementStats": problem.refinement.stats,
                "refinementCost": problem.refinement.cost,
            },
        )
        encoder.global_barrier()
