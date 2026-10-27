"""Matched full-frame CPU/GPU comparison, with decoding outside timing.

Neither backend substitutes for the other. Dataset frames are decoded once and
retained so both backends process identical arrays, poses, and timestamps.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tomllib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .cpu_hybrid import CPUHybridSLAM
from .trajectory import absolute_trajectory_error, match_timestamps, write_trajectory


def _gpu_class():
    from .hybrid import HybridSLAM

    return HybridSLAM


def _versions():
    result = {}
    for package in ("numpy", "slangpy", "symforce", "caspar-slang", "slang-rgbd-slam"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _provenance():
    """Portable installed-package hash plus bounded local Git/project metadata."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        path for path in package.rglob("*") if path.is_file() and path.suffix in (".py", ".slang")
    )
    for path in files:
        relative = path.relative_to(package).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    result = dict(
        source_git_revision=None,
        source_dirty=None,
        source_tree_sha256=digest.hexdigest(),
        source_tree_scope="installed slang_rgbd_slam package *.py/*.slang; sorted relative paths and raw bytes with length prefixes",
        backend_version=_versions().get("caspar-slang"),
        backend_dependency_sha=None,
    )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(package), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()

    try:
        repository = Path(git("rev-parse", "--show-toplevel"))
        result["source_git_revision"] = git("rev-parse", "HEAD")
        result["source_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
        project = repository / "pyproject.toml"
        if project.exists():
            data = tomllib.loads(project.read_text())
            source = data.get("tool", {}).get("uv", {}).get("sources", {}).get("caspar-slang", {})
            result["backend_dependency_sha"] = (
                source.get("rev") if isinstance(source, dict) else None
            )
    except (OSError, subprocess.SubprocessError, tomllib.TOMLDecodeError):
        pass
    return result


def _hardware(gpu):
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    adapter = None
    runtime = getattr(gpu, "runtime", None)
    if runtime is None:
        runtime = getattr(getattr(gpu, "tracker", None), "runtime", None)
    device = getattr(runtime, "device", None)
    info = getattr(device, "info", None)
    if info is not None:
        # Slang RHI device info is a typed object rather than a Python dict.
        adapter = {}
        for name in ("adapter_name", "device_name", "api_name", "device_type"):
            value = getattr(info, name, None)
            if value is not None:
                adapter[name] = str(value)
        if not adapter:
            adapter = str(info)
    return dict(
        cpu=cpu,
        gpu=adapter,
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python=platform.python_version(),
        versions=_versions(),
    )


def _summary(times):
    array = np.asarray(times, dtype=float)
    if not len(array):
        return dict(count=0, total=0.0, median=None, p95=None, mean=None)
    return dict(
        count=len(array),
        total=float(array.sum()),
        median=float(np.median(array)),
        p95=float(np.percentile(array, 95)),
        mean=float(array.mean()),
    )


def _run(slam, frames, truth):
    times, results = [], []
    for frame in frames:
        start = perf_counter()
        result = slam.process(frame.depth, frame.rgb, timestamp=frame.timestamp)
        times.append((perf_counter() - start) * 1000)
        results.append(result)
    corrected = getattr(slam, "corrected_trajectory", None)
    trajectory = corrected() if callable(corrected) else list(slam.trajectory)
    slam._comparison_trajectory = trajectory
    final_poses = {float(timestamp): pose for timestamp, pose in trajectory}
    for frame, result in zip(frames, results):
        if result["success"] and float(frame.timestamp) in final_poses:
            result["world_from_camera"] = final_poses[float(frame.timestamp)].copy()
    tracked = sum(bool(result["success"]) for result in results)
    summary = dict(
        frames=len(frames),
        tracked_frames=tracked,
        lost_frames=len(frames) - tracked,
        mapping_failed_frames=sum(not result.get("mapping_success", True) for result in results),
        wall_ms=_summary(times),
        subsequent_frames_wall_ms=_summary(times[1:]),
        map=slam.map.stats(),
        ate=absolute_trajectory_error(trajectory, truth) if truth else None,
    )
    summary["active_keyframes"] = len(slam.active_keyframes)
    summary["local_ba_events"] = list(slam.ba_events)
    summary["loop_events"] = list(slam.loop_events)
    summary["frame_results"] = [
        dict(
            timestamp=float(frame.timestamp),
            success=bool(result["success"]),
            mapping_success=bool(result.get("mapping_success", True)),
            wall_ms=float(time),
            joint_weighted_rmse=float(result["rmse"])
            if result.get("rmse") is not None and np.isfinite(result["rmse"])
            else None,
        )
        for frame, result, time in zip(frames, results, times)
    ]
    return summary, results


def compare_hybrid_backends(
    frames,
    camera,
    output,
    *,
    tracker_options=None,
    map_options=None,
    max_evaluation_poses=None,
    hybrid_options=None,
):
    """Run both implementations and save JSON plus TUM trajectories.

    Timed work is process(): input transfer for GPU, model raycasting, pyramid
    geometry, projective correspondence, residual/Jacobian, robust normal
    equations, solve, pose update, sparse TSDF allocation and integration, and
    small final diagnostics. Constructor/compiler, frame decode, map readback,
    map quality evaluation, trajectory export, and JSON export are outside it.
    CPU input/pyramid geometry and TSDF storage/fusion are float32; ICP pose
    transforms, projection, normal equations and solve use float64. NumPy
    raycast arithmetic is mixed precision. GPU computation is float32. CPU posegraph uses dense anchored solves; GPU uses bounded PCG.
    """
    if max_evaluation_poses is not None and (
        not isinstance(max_evaluation_poses, int)
        or isinstance(max_evaluation_poses, bool)
        or max_evaluation_poses < 1
    ):
        raise ValueError("max_evaluation_poses must be a positive integer or None")
    ground_truth = getattr(frames, "ground_truth", None)
    frames = list(frames)
    if not frames:
        raise ValueError("comparison requires a nonempty frame sequence")
    truth = (
        ground_truth
        if ground_truth is not None
        else [
            (frame.timestamp, frame.world_from_camera)
            for frame in frames
            if frame.world_from_camera is not None
        ]
    )
    truth = sorted(truth, key=lambda item: item[0])
    tracker_options, map_options = dict(tracker_options or {}), dict(map_options or {})
    # Construct the GPU backend first so a missing device never produces a
    # misleading CPU-only "comparison" output.
    hybrid_options = dict(hybrid_options or {})
    gpu = _gpu_class()(
        camera, tracker_options=tracker_options, map_options=map_options, **hybrid_options
    )
    cpu = CPUHybridSLAM(
        camera, tracker_options=tracker_options, map_options=map_options, **hybrid_options
    )
    cpu_report, cpu_results = _run(cpu, frames, truth)
    gpu_report, gpu_results = _run(gpu, frames, truth)
    common = [
        i for i, (a, b) in enumerate(zip(cpu_results, gpu_results)) if a["success"] and b["success"]
    ]
    cpu_common = [(frames[i].timestamp, cpu_results[i]["world_from_camera"]) for i in common]
    gpu_common = [(frames[i].timestamp, gpu_results[i]["world_from_camera"]) for i in common]
    cpu_report["common_frames_ate"] = (
        absolute_trajectory_error(cpu_common, truth) if truth else None
    )
    gpu_report["common_frames_ate"] = (
        absolute_trajectory_error(gpu_common, truth) if truth else None
    )
    pose_errors = [
        np.linalg.norm(
            cpu_results[i]["world_from_camera"][:3, 3] - gpu_results[i]["world_from_camera"][:3, 3]
        )
        for i in common
    ]
    # Evaluate both maps at identical world poses. Ground truth, when supplied,
    # provides an independent camera path. Otherwise use the CPU trajectory.
    ground_truth_pairs = match_timestamps([(frame.timestamp, None) for frame in frames], truth)
    paired_ground_truth = {i: truth[j][1] for i, j in ground_truth_pairs}
    first_success = next((i for i, result in enumerate(cpu_results) if result["success"]), None)
    evaluation_poses, pose_source = cpu_common, "cpu_common_tracked_poses"
    if first_success in paired_ground_truth:
        # The reconstructed world is anchored at the first tracked camera, while
        # dataset ground truth can have any global coordinate frame.
        world_from_truth = np.linalg.inv(paired_ground_truth[first_success])
        evaluation_poses = [
            (frames[i].timestamp, world_from_truth @ pose)
            for i, pose in sorted(paired_ground_truth.items())
        ]
        pose_source = "ground_truth_in_initial_camera"
    available_evaluation_poses = len(evaluation_poses)
    if max_evaluation_poses is not None and len(evaluation_poses) > max_evaluation_poses:
        selected = np.linspace(0, len(evaluation_poses) - 1, max_evaluation_poses, dtype=int)
        evaluation_poses = [evaluation_poses[i] for i in selected]
    squared_error, paired, cpu_pixels, gpu_pixels, total_pixels = 0.0, 0, 0, 0, 0
    frame_depth = {frame.timestamp: frame.depth for frame in frames}
    cpu_truth_error, gpu_truth_error, cpu_truth_pixels, gpu_truth_pixels = 0.0, 0.0, 0, 0
    for timestamp, pose in evaluation_poses:
        a, b = cpu.map.raycast(pose), gpu.map.raycast(pose)
        av, bv = a > 0, b > 0
        joint = av & bv
        squared_error += float(np.sum((a[joint].astype(np.float64) - b[joint]) ** 2))
        paired += int(joint.sum())
        cpu_pixels, gpu_pixels = cpu_pixels + int(av.sum()), gpu_pixels + int(bv.sum())
        total_pixels += a.size
        measured = frame_depth[timestamp]
        for values, valid, backend in ((a, av, "cpu"), (b, bv, "gpu")):
            valid = valid & (measured > 0)
            # Measured depth belongs to the evaluation pose only with GT input.
            if pose_source == "ground_truth_in_initial_camera":
                error = float(np.sum((values[valid].astype(np.float64) - measured[valid]) ** 2))
                if backend == "cpu":
                    cpu_truth_error, cpu_truth_pixels = (
                        cpu_truth_error + error,
                        cpu_truth_pixels + int(valid.sum()),
                    )
                else:
                    gpu_truth_error, gpu_truth_pixels = (
                        gpu_truth_error + error,
                        gpu_truth_pixels + int(valid.sum()),
                    )
    cpu_median, gpu_median = cpu_report["wall_ms"]["median"], gpu_report["wall_ms"]["median"]
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.float64(frame.timestamp).tobytes())
        digest.update(np.ascontiguousarray(frame.depth).tobytes())
        if frame.rgb is not None:
            digest.update(np.ascontiguousarray(frame.rgb).tobytes())
    report = dict(
        schema_version=1,
        input=dict(
            frames=len(frames),
            decoded_once=True,
            sha256=digest.hexdigest(),
            decoded_bytes=sum(
                frame.depth.nbytes + (0 if frame.rgb is None else frame.rgb.nbytes)
                for frame in frames
            ),
        ),
        configuration=dict(
            camera=asdict(camera),
            tracker_options=tracker_options,
            map_options=map_options,
            hybrid_options=hybrid_options,
            keyframe_map_policy="normal-frame fusion until transactional rebuild from retained bounded FIFO keyframes",
            dense_ba_support="current associations, >=80% initial factor count, final robust cost transaction",
        ),
        timing_boundary="full process wall time; transfers, raycast, pyramid, association, "
        "residual/Jacobian, reduction, solve, update, sparse allocation/fusion, "
        "keyframe feature extraction/retrieval, local coupled inverse-depth BA, loop verification/pose graph/map rebuild when scheduled, small final diagnostics included; decoding, constructor/compile, "
        "quality evaluation, readback/export excluded",
        backend_order=["cpu", "gpu"],
        trajectory_boundary="final corrected trajectories read back after timed frames; paired pose errors/ATE/map evaluation use final corrections",
        hardware=_hardware(gpu),
        provenance=_provenance(),
        arithmetic=dict(
            cpu="float32 input/pyramid geometry and TSDF storage/fusion; float64 joint geometric/photometric transforms, Schur BA and posegraph solves; "
            "projection, normal equations and solve; graph uses anchored dense solve with central-difference Jacobians; raycast uses NumPy mixed precision",
            gpu="float32 geometry/map/normal equations/solve; graph uses generated Jacobians and bounded matrix-free PCG",
        ),
        cpu=cpu_report,
        gpu=gpu_report,
        common_tracked_frames=len(common),
        median_wall_speedup=cpu_median / gpu_median if gpu_median and cpu_median else None,
        max_cpu_gpu_translation_difference_m=float(max(pose_errors)) if pose_errors else None,
        map_comparison=dict(
            evaluation_pose_source=pose_source,
            evaluation_poses=len(evaluation_poses),
            available_evaluation_poses=available_evaluation_poses,
            evaluation_pose_limit=max_evaluation_poses,
            evaluation_timestamps=[float(timestamp) for timestamp, _ in evaluation_poses],
            paired_pixels=paired,
            depth_rmse_m=float(np.sqrt(squared_error / paired)) if paired else None,
            cpu_coverage=cpu_pixels / total_pixels if total_pixels else None,
            gpu_coverage=gpu_pixels / total_pixels if total_pixels else None,
            cpu_measured_depth_rmse_m=float(np.sqrt(cpu_truth_error / cpu_truth_pixels))
            if cpu_truth_pixels
            else None,
            gpu_measured_depth_rmse_m=float(np.sqrt(gpu_truth_error / gpu_truth_pixels))
            if gpu_truth_pixels
            else None,
        ),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_trajectory(output / "cpu_trajectory.txt", cpu._comparison_trajectory)
    write_trajectory(output / "gpu_trajectory.txt", gpu._comparison_trajectory)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    (output / "comparison.json").write_text(encoded + "\n")
    return json.loads(encoded)
