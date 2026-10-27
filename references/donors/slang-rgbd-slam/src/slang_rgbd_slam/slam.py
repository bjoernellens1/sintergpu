"""Run dense RGB-D SLAM and write trajectories, maps, and validation reports."""

import json
from pathlib import Path

import numpy as np

from .mapping import VoxelMap
from .trajectory import absolute_trajectory_error, write_trajectory


def _initial_geometry_constrained(depth, camera):
    """Keyframe-only host check: six independent point-to-plane pose directions."""
    v, u = np.indices(depth.shape)
    points = np.stack(
        ((u - camera.cx) * depth / camera.fx, (v - camera.cy) * depth / camera.fy, depth), axis=-1
    )
    centers = points[1:-1, 1:-1]
    normals = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
    lengths = np.linalg.norm(normals, axis=-1)
    valid = (
        (depth[1:-1, 1:-1] > 0)
        & (depth[1:-1, 2:] > 0)
        & (depth[1:-1, :-2] > 0)
        & (depth[2:, 1:-1] > 0)
        & (depth[:-2, 1:-1] > 0)
        & (lengths > 1e-10)
    )
    if valid.sum() < 6:
        return False
    n = normals[valid] / lengths[valid, None]
    p = centers[valid]
    jacobian = np.concatenate((np.cross(p, n), n), axis=1)
    return np.linalg.matrix_rank(jacobian, tol=1e-5) == 6


class LegacyRGBDSLAM:
    def __init__(
        self,
        camera,
        *,
        tracker=None,
        keyframe_interval=5,
        keyframe_translation=0.10,
        voxel_size=0.03,
        max_voxels=200000,
        **tracker_options,
    ):
        if keyframe_interval < 1 or keyframe_translation <= 0:
            raise ValueError("Invalid keyframe limits")
        if tracker is None:
            from .tracker import VulkanTracker

            tracker = VulkanTracker(camera, **tracker_options)
        self.camera, self.tracker = camera, tracker
        self.map = VoxelMap(voxel_size, max_voxels)
        self.keyframe_interval, self.keyframe_translation = keyframe_interval, keyframe_translation
        self.reference_pose = None
        self.reference_index = 0
        self.index = 0
        self.trajectory = []
        self.results = []
        self.keyframes = 0
        self.last_timestamp = None

    def process(self, depth, rgb=None, *, timestamp=0.0):
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (self.camera.height, self.camera.width) or not np.isfinite(timestamp):
            raise ValueError("Invalid frame")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("Depth must be finite nonnegative metres")
        if rgb is not None and (
            np.asarray(rgb).shape != (*depth.shape, 3) or np.asarray(rgb).dtype != np.uint8
        ):
            raise ValueError("RGB must be HxWx3 uint8")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Timestamps must increase")
        self.last_timestamp = float(timestamp)
        if self.reference_pose is None:
            if not _initial_geometry_constrained(depth, self.camera):
                result = {
                    "timestamp": float(timestamp),
                    "success": False,
                    "status": "lost",
                    "world_from_camera": None,
                    "reason": "unconstrained_initial_geometry",
                }
                self.index += 1
                self.results.append(result)
                return result
            self.tracker.set_reference(depth)
            pose = np.eye(4)
            self.reference_pose = pose.copy()
            self.reference_index = self.index
            self.map.integrate(depth, self.camera, pose, rgb)
            self.keyframes += 1
            result = {
                "timestamp": float(timestamp),
                "success": True,
                "status": "initialized",
                "world_from_camera": pose,
                "keyframe": True,
            }
        else:
            tracking = self.tracker.track(depth)
            if not tracking.success:
                result = {
                    "timestamp": float(timestamp),
                    "success": False,
                    "status": "lost",
                    "world_from_camera": None,
                    "keyframe": False,
                    "correspondences": tracking.correspondences,
                    "rmse": float(tracking.rmse) if np.isfinite(tracking.rmse) else None,
                }
                self.index += 1
                self.results.append(result)
                return result
            relative = np.asarray(tracking.pose)
            pose = self.reference_pose @ relative
            promote = (
                self.index - self.reference_index >= self.keyframe_interval
                or np.linalg.norm(relative[:3, 3]) >= self.keyframe_translation
            )
            result = {
                "timestamp": float(timestamp),
                "success": True,
                "status": "tracked",
                "world_from_camera": pose,
                "keyframe": bool(promote),
                "correspondences": tracking.correspondences,
                "rmse": float(tracking.rmse) if np.isfinite(tracking.rmse) else None,
                "elapsed_ms": float(tracking.elapsed_ms),
            }
            if promote:
                self.tracker.set_reference(depth)
                self.reference_pose = pose.copy()
                self.reference_index = self.index
                self.map.integrate(depth, self.camera, pose, rgb)
                self.keyframes += 1
        self.trajectory.append((float(timestamp), pose.copy()))
        self.results.append(result)
        self.index += 1
        return result


def run_sequence(
    dataset, camera, output, max_frames=None, *, backend="gpu", engine="hybrid", **options
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if engine not in {"hybrid", "baseline"}:
        raise ValueError("engine must be hybrid or baseline")
    if backend == "legacy":
        slam = LegacyRGBDSLAM(camera, **options)
    elif backend == "gpu":
        if engine == "hybrid":
            from .hybrid import HybridSLAM as Pipeline
        else:
            from .gpu_slam import GPUSLAM as Pipeline
        slam = Pipeline(camera, **options)
    elif backend == "cpu":
        if engine == "hybrid":
            from .cpu_hybrid import CPUHybridSLAM as Pipeline
        else:
            from .cpu import CPUSLAM as Pipeline
        slam = Pipeline(camera, **options)
    else:
        raise ValueError("backend must be gpu, cpu, or legacy")
    truth = []
    processed = 0
    for frame in dataset:
        if max_frames is not None and processed >= max_frames:
            break
        slam.process(frame.depth, frame.rgb, timestamp=frame.timestamp)
        if frame.world_from_camera is not None:
            truth.append((frame.timestamp, frame.world_from_camera))
        processed += 1
    if getattr(dataset, "ground_truth", None) is not None:
        truth = dataset.ground_truth
    trajectory = (
        slam.corrected_trajectory() if hasattr(slam, "corrected_trajectory") else slam.trajectory
    )
    write_trajectory(output / "trajectory.txt", trajectory)
    slam.map.export_ply(output / "map.ply")
    report = {
        "frames": processed,
        "tracked_frames": len(trajectory),
        "lost_frames": processed - len(trajectory),
        "mapping_failed_frames": sum(not r.get("mapping_success", True) for r in slam.results),
        "map_statistics": slam.map.stats() if backend != "legacy" else None,
        "keyframes": getattr(slam, "keyframes", len(slam.trajectory)),
        "voxels": len(slam.map),
        "mapping_backend": "CPU sparse voxel prototype"
        if backend == "legacy"
        else backend + " sparse hashed TSDF",
        "tracking_backend": "CPU NumPy" if backend == "cpu" else "Vulkan",
        "ground_truth": "available" if truth else "missing",
        "ate": absolute_trajectory_error(trajectory, truth) if truth else None,
    }
    report["engine"] = engine if backend != "legacy" else "legacy"
    if engine == "hybrid" and backend != "legacy":
        report["local_ba_events"] = slam.ba_events
        report["loop_events"] = slam.loop_events
        report["active_keyframes"] = len(slam.active_keyframes)
        report["map_rebuild_policy"] = (
            "retained bounded keyframes; evicted history is not retained in rebuilt maps"
        )
    report["tracking"] = [
        {k: v for k, v in r.items() if k != "world_from_camera"} for r in slam.results
    ]
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def RGBDSLAM(camera, **options):
    """Construct dense GPU SLAM with local BA and sparse loop support."""
    from .hybrid import HybridSLAM

    return HybridSLAM(camera, **options)
