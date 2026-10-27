"""Deterministic controls and inventory for the public reference kernels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

from vkl.backend import get_backend_info
from vkl.bench.triton_qualification import (
    KernelCase,
    compare_outputs,
    measure_case,
    serialize_report,
)
from vkl.kernels import reference
from vkl.kernels.triton import argmin as _argmin_module
from vkl.kernels.triton import select_best_inverse_depth


def classifications() -> dict[str, dict[str, str]]:
    """Return the reviewed Triton disposition for every public reference kernel."""
    groups = {
        "candidate": {
            "project_depth_evidence": "Measured fixed-capacity projection/depth fusion; transaction gate remains.",
        },
        "deferred": {
            "aggregate_depth_cost_volume": "Dynamic-programming scan needs a separate cost-volume study.",
            "dense_pose_depth": "Do not materialize or custom-port the dense dynamic cost volume yet.",
            "direct_pose_depth_dso": "Direct depth has dynamic hypotheses and selected-point policy.",
            "direct_pose_depth_tensorized": "Direct depth has dynamic hypotheses and selected-point policy.",
            "rasterize_reprojection": "Visibility rasterization is backend-specialized scatter semantics.",
            "select_compact": "Stable compact ordering requires dynamic prefix/counter semantics.",
            "spatial_bin_reduce": "Lexicographic unique/reduction is dynamic and explicitly out of scope.",
        },
        "retain_vendor": {
            "camera_rectify": "Vendor sampling primitive already owns interpolation semantics.",
            "canonical_rescale": "Vendor interpolation preserves the public camera/image contract.",
            "depth_pyramid": "Vendor pooling/interpolation is the measured primitive boundary.",
            "depth_downsample": "Validity-aware pyramid pooling is the vendor primitive boundary.",
            "depth_upsample": "Nearest-neighbour propagation is a vendor interpolation primitive.",
            "resolution_pyramid": "Vendor pooling/interpolation is the measured primitive boundary.",
        },
        "retain_compile": {
            "align_depth": "Tensor reductions are compile-friendly without a fixed fusion boundary.",
            "depth_convert": "Small elementwise conversion is compile-friendly.",
            "depth_evidence": "Mixed pooling/reduction semantics need a transaction profile first.",
            "depth_geometry": "Reusable geometry is composeable PyTorch with no measured launch opportunity.",
            "dense_track_reduce": "Dense tracking reduction needs a composed tracking transaction profile before fusion.",
            "sparse_pose_reduce": "Fixed-capacity sparse reduction needs a composed sparse+dense transaction profile before fusion.",
            "depth_register": "Registration relies on vendor sampling/scatter composition.",
            "depth_reproject": "Reprojection relies on vendor sampling composition.",
            "deterministic_reduce_solve": "Small fixed solve is compile-friendly until transaction profiling justifies fusion.",
            "dso_select_points": "Selection includes top-k policy and is not a fixed kernel boundary.",
            "fast_select": "Corner selection includes NMS/top-k policy.",
            "fuse_depth": "Confidence fusion is compile-friendly elementwise/reduction work.",
            "depth_densify": "RGB-guided local propagation is compile-friendly until a mapping transaction profile justifies fusion.",
            "keyframe_filter": "Frame policy is intentionally outside kernel specialization.",
            "keypoint_evidence": "Bounded evidence reductions need transaction profiling first.",
            "landmark_update": "State/identity update is intentionally not a stateless Triton target.",
            "lk_track": "Grid sampling and iterative tracking remain vendor/compiled work.",
            "photometric_isp": "Autograd-sensitive photometry requires forward/backward qualification first.",
            "pose_observable_select": "Selection includes top-k policy and is not a fixed kernel boundary.",
            "reproject_rgbd": "Reprojection relies on vendor sampling composition.",
            "residual_evidence": "Per-pixel global atomics were measured and rejected on ROCm.",
            "rigid_align_3d": "Small batched SVD solve is compile-friendly until relocalization profiling justifies fusion.",
            "rgbd_evidence": "Mixed pooling/reduction semantics need a transaction profile first.",
            "shi_tomasi_select": "Corner selection includes NMS/top-k policy.",
            "spatial_select_points": "Spatial selection includes top-k policy.",
            "temporal_best_observation": "Temporal ranking is reduction/policy work.",
            "temporal_depth_fuse": "Temporal fusion is compile-friendly tensor reduction work.",
            "temporal_frame_compare": "Temporal alignment uses vendor sampling composition.",
            "temporal_reproject_rgb": "Temporal reprojection uses vendor sampling composition.",
            "temporal_rgbd_reference": "D4/T0/D9 composition needs a mapper-transaction profile before fusion.",
            "temporal_restore_rgb": "Temporal restoration uses vendor sampling composition.",
            "temporal_restore_rgbd": "Temporal restoration uses vendor sampling composition.",
            "tracking_prepare": "Composed tracker preparation needs a transaction profile before any fusion decision.",
            "triangulate_depth": "Small linear algebra remains compile-friendly.",
            "uniform_select": "Uniform selection has no demonstrated compute bottleneck.",
            "validate_frame_batch": "Boundary validation is deliberately not a GPU kernel.",
            "world_transform": "Small batched matrix transform is compile-friendly.",
        },
    }
    result = {
        name: {"family": "triton", "disposition": disposition, "reason": reason}
        for disposition, entries in groups.items()
        for name, reason in entries.items()
    }
    if set(result) != set(reference.__all__):
        raise RuntimeError("Triton classifications must cover every public reference export")
    return result


def _scene(device: torch.device, batch: int) -> dict[str, torch.Tensor]:
    h = w = 8
    rgb = torch.linspace(0.1, 0.9, batch * 3 * h * w, device=device, dtype=torch.float32).reshape(batch, 3, h, w)
    depth = torch.full((batch, 1, h, w), 2.0, device=device)
    valid = torch.ones_like(depth, dtype=torch.bool)
    confidence = torch.full_like(depth, 0.8)
    intrinsics = torch.eye(3, device=device).expand(batch, -1, -1).clone()
    intrinsics[:, 0, 0] = 6.0
    intrinsics[:, 1, 1] = 6.0
    intrinsics[:, 0, 2] = 3.5
    intrinsics[:, 1, 2] = 3.5
    pose = torch.eye(4, device=device).expand(batch, -1, -1).clone()
    neighbor_pose = pose.clone()
    neighbor_pose[:, 0, 3] = 0.1
    return {"rgb": rgb, "depth": depth, "valid": valid, "confidence": confidence, "intrinsics": intrinsics, "pose": pose, "neighbor_pose": neighbor_pose}


def _build(name: str, device: torch.device, batch: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    s = _scene(device, batch)
    rgb, depth, valid, conf, k, pose, neighbor_pose = (s[key] for key in ("rgb", "depth", "valid", "confidence", "intrinsics", "pose", "neighbor_pose"))
    rgb_n = rgb[:, None].expand(-1, 1, -1, -1, -1)
    pose_n = neighbor_pose[:, None]
    if name == "aggregate_depth_cost_volume": return (torch.ones(batch, 3, 8, 8, device=device), rgb), {}
    if name == "align_depth": return (depth, depth * 1.1, valid, conf), {}
    if name == "camera_rectify": return (rgb, k, k, (8, 8)), {}
    if name == "canonical_rescale": return (), {"rgb": rgb, "depth": depth, "intrinsics": k, "rgb_valid": valid, "depth_valid": valid, "target_pixels": 64, "min_width": 4, "min_height": 4}
    if name == "dense_pose_depth": return (rgb, rgb_n, pose, pose_n, k), {"num_hypotheses": 2, "patch_radius": 0, "min_views": 1}
    if name == "depth_convert": return (), {"values": depth, "representation": "DEPTH", "valid": valid, "confidence": conf, "scale": 1.0, "metric_status": "METRIC_ANCHORED"}
    if name == "depth_densify": return (depth, valid, conf, rgb), {"steps": 1}
    if name == "depth_downsample": return (depth, valid, conf), {"scale": 2}
    if name == "depth_evidence": return (depth, valid, conf), {"grid_size": (2, 2)}
    if name == "depth_geometry": return (depth, k, pose, conf), {}
    if name == "dense_track_reduce":
        return (), {
            "source_depth": depth,
            "source_intrinsics": k,
            "source_pose_c2w": pose,
            "reference_depth": depth,
            "reference_intrinsics": k,
            "reference_pose_c2w": pose,
            "source_rgb": rgb,
            "reference_rgb": rgb,
        }
    if name == "sparse_pose_reduce":
        xyz = torch.tensor(
            [[[-0.2, -0.1, 2.0], [0.3, -0.2, 2.2], [-0.1, 0.4, 1.8]]],
            device=device,
        ).expand(batch, -1, -1).clone()
        return (xyz, xyz.clone(), pose), {"partial_count": 2}
    if name == "depth_pyramid": return (), {"depth": depth, "valid": valid, "confidence": conf}
    if name == "depth_register": return (depth, k, k, torch.eye(4, device=device).expand(batch, -1, -1).clone(), (8, 8)), {}
    if name == "depth_reproject": return (depth, k, pose, k, neighbor_pose), {"target_hw": (8, 8)}
    if name == "depth_upsample": return (depth[:, :, :4, :4], valid[:, :, :4, :4], conf[:, :, :4, :4]), {"size": (8, 8)}
    if name == "deterministic_reduce_solve":
        partial_H = torch.eye(6, device=device).expand(batch, 2, -1, -1).clone()
        partial_b = torch.zeros(batch, 2, 6, device=device)
        return (partial_H, partial_b, torch.ones(batch, 2, dtype=torch.bool, device=device)), {}
    if name == "direct_pose_depth_dso": return (), {"rgb_reference": rgb, "rgb_neighbors": rgb_n, "pose_reference": pose, "pose_neighbors": pose_n, "intrinsics": k, "num_points": 2, "num_hypotheses": 2, "patch_radius": 1}
    if name == "direct_pose_depth_tensorized": return (), {"rgb_reference": rgb, "rgb_neighbors": rgb_n, "pose_reference": pose, "pose_neighbors": pose_n, "intrinsics": k, "num_points": 2, "num_hypotheses": 2, "patch_radius": 1}
    if name == "dso_select_points": return (rgb,), {"num_points": 2, "patch_radius": 1}
    if name == "fast_select": return (rgb,), {"num_points": 2, "margin": 3}
    if name == "fuse_depth": return ([depth, depth * 1.01], [valid, valid], [conf, conf]), {}
    if name == "keyframe_filter": return (), {"frame_score": torch.full((batch,), 0.8, device=device), "tile_score": torch.full((batch, 2, 2), 0.8, device=device), "pose": pose, "reference_pose": torch.eye(4, device=device).expand(batch, -1, -1).clone(), "frame_gap": torch.full((batch,), 30, dtype=torch.int64, device=device), "rgb_novelty": torch.full((batch,), 0.2, device=device)}
    if name == "keypoint_evidence": return (), {"selected": valid, "valid": valid, "confidence": conf, "parallax_px": torch.ones_like(depth), "photometric_error": torch.full_like(depth, 0.001), "gradient": torch.ones_like(depth), "tile_rows": 2, "tile_cols": 2}
    if name == "fit_local_planes":
        xyz = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]], device=device).expand(batch, -1, -1).clone()
        return (xyz, torch.arange(4, device=device, dtype=torch.int64)[None, None].expand(batch, 4, -1)), {}
    if name == "landmark_update":
        xyz = torch.zeros(batch, 3, 3, device=device)
        return (xyz, torch.ones(batch, 3, dtype=torch.int64, device=device), torch.ones(batch, 3, dtype=torch.int64, device=device), torch.ones(batch, 3, dtype=torch.bool, device=device), xyz + 0.01, torch.ones(batch, 3, dtype=torch.bool, device=device)), {}
    if name == "lk_track":
        track_pixels = torch.tensor([[[2, 2], [5, 5]]], dtype=torch.int64, device=device).expand(batch, -1, -1).clone()
        return (rgb, rgb * 0.99, track_pixels, track_pixels.to(rgb.dtype)), {"patch_radius": 1, "iterations": 1}
    if name == "photometric_isp": return (rgb,), {"exposure_ev": torch.zeros(batch, 1, 1, 1, device=device), "rgb_gain": torch.ones(batch, 3, 1, 1, device=device), "evidence": False}
    if name == "point_splat_render":
        xyz = torch.tensor([[[0.0, 0.0, 2.0], [0.1, 0.0, 1.0]]], device=device).expand(batch, -1, -1).clone()
        colors = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device).expand(batch, -1, -1).clone()
        return (xyz, colors, k, pose), {"image_hw": (8, 8)}
    if name == "pose_observable_select": return (rgb, k, pose, neighbor_pose), {"num_points": 2, "margin": 3}
    if name == "project_depth_evidence": return (torch.tensor([[[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]], device=device).expand(batch, -1, -1), torch.ones(batch, 2, dtype=torch.bool, device=device), torch.ones(batch, 2, dtype=torch.bool, device=device), torch.ones(batch, 2, dtype=torch.bool, device=device), torch.eye(4, device=device).expand(batch, -1, -1).clone(), k, depth), {"support_threshold": 0.1, "conflict_threshold": 0.2}
    if name == "rigid_align_3d":
        xyz = torch.tensor(
            [[[-0.3, -0.2, 1.0], [0.4, -0.1, 1.2], [-0.2, 0.5, 1.4], [0.3, 0.4, 1.1], [0.1, -0.4, 1.6], [-0.5, 0.2, 1.3]]],
            device=device,
        ).expand(batch, -1, -1).clone()
        return (xyz, xyz + torch.tensor([0.1, -0.2, 0.3], device=device)), {}
    if name == "rasterize_reprojection": return (depth, conf, k, pose, k, neighbor_pose), {"target_hw": (8, 8)}
    if name == "reproject_rgbd": return (depth, k, pose, k, neighbor_pose), {"target_hw": (8, 8)}
    if name == "residual_evidence": return (rgb, rgb * 0.99), {"observed_depth": depth, "predicted_depth": depth * 1.01, "valid": valid, "tile_rows": 2, "tile_cols": 2}
    if name == "resolution_pyramid": return (), {"rgb": rgb}
    if name == "rgbd_evidence": return (), {"rgb": rgb, "depth": depth, "rgb_valid": valid, "depth_valid": valid, "depth_measurement_valid": valid, "tile_rows": 2, "tile_cols": 2}
    if name == "select_compact": return (rgb.permute(0, 2, 3, 1).reshape(batch, 64, 3), valid.reshape(batch, 64)), {"capacity": 4, "action_codes": torch.ones(batch, 64, dtype=torch.int32, device=device)}
    if name == "shi_tomasi_select": return (rgb,), {"num_points": 2, "margin": 3, "window": 3}
    pixels = torch.tensor([[[2.0, 2.0], [5.0, 5.0], [3.0, 4.0]]], device=device).expand(batch, -1, -1).clone()
    if name == "spatial_select_points": return (pixels.to(torch.int64), torch.ones(batch, 3, device=device), torch.ones(batch, 3, dtype=torch.bool, device=device)), {"num_points": 2, "height": 8, "width": 8, "tiles": (2, 2)}
    if name == "spatial_bin_reduce": return (torch.tensor([[[0.1, 0.1, 2.0], [0.2, 0.1, 2.0], [1.1, 0.1, 2.0]]], device=device).expand(batch, -1, -1), torch.ones(batch, 3, dtype=torch.bool, device=device)), {"voxel_size": 1.0, "origin": torch.zeros(3, device=device), "capacity": 4}
    if name == "temporal_best_observation": return (torch.ones(batch, 1, dtype=torch.bool, device=device), torch.full((batch,), 0.5, device=device), torch.full((batch, 1), 0.6, device=device)), {}
    if name == "temporal_depth_fuse": return (depth, [depth * 1.01], valid, [valid], conf, [conf]), {}
    if name == "temporal_frame_compare": return (rgb, rgb_n), {"alignment_confidence": conf[:, None], "aligned_valid": valid[:, None].bool(), "tile_size": 4}
    if name == "temporal_reproject_rgb": return (rgb, depth, rgb_n, k, pose, k[:, None], pose_n), {}
    if name == "temporal_rgbd_reference": return (), {"target_depth": depth, "source_rgb": rgb_n, "source_depth": depth[:, None], "target_intrinsics": k, "target_pose_c2w": pose, "source_intrinsics": k[:, None], "source_pose_c2w": pose[:, None]}
    if name == "temporal_restore_rgb": return (rgb, rgb_n), {"alignment_confidence": conf[:, None], "aligned_valid": valid[:, None].bool(), "tile_size": 4}
    if name == "temporal_restore_rgbd": return (rgb, rgb_n), {"context_valid": valid[:, None].bool(), "context_confidence": conf[:, None], "alignment": "rgb_aligned"}
    if name == "tracking_prepare": return (), {"rgb": rgb, "depth": depth, "intrinsics": k, "pose_c2w": pose, "depth_confidence": conf, "target_pixels": 64, "min_width": 4, "min_height": 4}
    if name == "triangulate_depth": return (pixels, pixels + torch.tensor([0.2, 0.0], device=device), k, k, pose, neighbor_pose), {}
    if name == "uniform_select": return (rgb,), {"num_points": 2, "margin": 1}
    if name == "validate_frame_batch": return ({"rgb": rgb, "depth": depth, "intrinsics": k, "pose_c2w": pose},), {}
    if name == "world_transform": return (torch.cat((torch.zeros(batch, 2, 8, 8, device=device), depth), dim=1), pose), {}
    if name in {"voxelize_points", "world_voxel_reduce"}:
        xyz = torch.tensor([[[0.1, 0.1, 2.0], [0.2, 0.1, 2.0], [1.1, 0.1, 2.0]]], device=device).expand(batch, -1, -1).clone()
        if name == "voxelize_points":
            return (xyz, 1.0), {"rgb": torch.full_like(xyz, 0.5), "confidence": torch.full((batch, 3), 0.8, device=device)}
        return (xyz, torch.tensor([[1, 1, 2]], dtype=torch.int64, device=device).expand(batch, -1), 1.0), {"rgb": torch.full_like(xyz, 0.5), "min_source_support": 2}
    if name == "voxel_surface_mesh":
        voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64, device=device).repeat(batch, 1)
        return (voxels, 1.0), {"batch_index": torch.arange(batch, dtype=torch.int64, device=device).repeat_interleave(2)}
    if name == "select_best_inverse_depth": return (torch.arange(batch * 3 * 4, device=device, dtype=torch.float32).reshape(batch, 3, 4) / 10, torch.linspace(0.1, 1.0, 4, device=device)), {}
    raise KeyError(name)


def _argmin_reference(cost: torch.Tensor, inverse_depths: torch.Tensor) -> dict[str, torch.Tensor]:
    best_two, indices = cost.topk(2, dim=-1, largest=False)
    index = indices[..., 0]
    return {
        "best_cost": best_two[..., 0],
        "second_cost": best_two[..., 1],
        "best_index": index,
        "inverse_depth": inverse_depths[index],
    }


def _argmin_triton(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    output = select_best_inverse_depth(*args, **kwargs)
    actual = "triton" if args[0].device.type == "cuda" and _argmin_module.triton is not None else "eager_fallback"
    _argmin_triton.dispatch_metadata = {"requested": "triton", "actual": actual}
    return output


def _workload(name: str) -> dict[str, Any]:
    workload: dict[str, Any] = {
        "batches": (1, 2, 4, 8),
        "deterministic": True,
        "image_hw": (8, 8),
        "channels": 3,
        "neighbors": 1,
    }
    workload["algorithm_dimensions"] = {
        "operation": name,
        "batches": [1, 2, 4, 8],
        "image_hw": [8, 8],
        "channels": 3,
        "neighbors": 1,
    }
    if name in {"select_compact", "spatial_bin_reduce"}:
        workload["capacity"] = 4
        workload["algorithm_dimensions"]["capacity"] = 4
    if name in {"dense_pose_depth", "direct_pose_depth_dso", "direct_pose_depth_tensorized"}:
        workload.update({"num_hypotheses": 2, "patch_radius": 1 if name != "dense_pose_depth" else 0})
        workload["algorithm_dimensions"].update({"num_hypotheses": 2, "patch_radius": workload["patch_radius"]})
    if name == "dso_select_points" or name.endswith("_select") or name == "spatial_select_points":
        workload["num_points"] = 2
        workload["algorithm_dimensions"]["num_points"] = 2
    if name == "lk_track":
        workload.update({"patch_radius": 1, "iterations": 1})
        workload["algorithm_dimensions"].update({"patch_radius": 1, "iterations": 1})
    if name in {"temporal_restore_rgb", "temporal_frame_compare"}:
        workload["tile_size"] = 4
        workload["algorithm_dimensions"]["tile_size"] = 4
    if name in {"depth_evidence", "keypoint_evidence", "rgbd_evidence", "residual_evidence"}:
        workload.update({"tile_rows": 2, "tile_cols": 2})
        workload["algorithm_dimensions"].update({"tile_rows": 2, "tile_cols": 2})
    if name == "select_best_inverse_depth":
        workload.update({"depth_hypotheses": 4, "triton_backend_only": True})
        workload["algorithm_dimensions"]["depth_hypotheses"] = 4
    return workload


def _describe_control(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _describe_control(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe_control(item) for item in value]
    return value


def inventory() -> tuple[KernelCase, ...]:
    cases: list[KernelCase] = []
    for name in (*reference.__all__, "select_best_inverse_depth"):
        implementation = _argmin_reference if name == "select_best_inverse_depth" else getattr(reference, name)
        triton = _argmin_triton if name == "select_best_inverse_depth" else None
        workload = _workload(name)
        args, kwargs = _build(name, torch.device("cpu"), 1)
        workload["algorithm_dimensions"]["control"] = {
            "args": _describe_control(args),
            "kwargs": _describe_control(kwargs),
        }
        cases.append(KernelCase(name, "inverse_depth" if name == "select_best_inverse_depth" else "reference", lambda device, batch, operation=name: _build(operation, device, batch), implementation, triton, compare_outputs, workload))
    return tuple(cases)


def build_report_metadata(
    device: torch.device, batches: tuple[int, ...], command: str
) -> dict[str, Any]:
    backend = get_backend_info(device).as_dict()
    return {
        "source_revision": os.getenv("VKL_SOURCE_REVISION", "unknown"),
        "image_revision": os.getenv("VKL_IMAGE_REVISION", "unknown"),
        "image_reference": os.getenv("VKL_IMAGE_REFERENCE", "unknown"),
        "torch_version": torch.__version__,
        "triton_version": getattr(_argmin_module.triton, "__version__", "unavailable"),
        "backend": backend,
        "device_name": backend["device_name"],
        "command": command,
        "algorithm_dimensions": {"batches": list(batches), "image_hw": [8, 8]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/triton/inventory.json")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batches", default="1,2,4,8", help="comma-separated batch sizes")
    parser.add_argument("--variants", default="eager,compiled,triton", help="comma-separated variants (eager,compiled,triton)")
    args = parser.parse_args()
    batches = tuple(int(item.strip()) for item in args.batches.split(",") if item.strip())
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    reports = [measure_case(case, variants, device=args.device, warmup=args.warmup, iterations=args.iterations, batches=batches) for case in inventory()]
    serialize_report(
        {
            "device": args.device,
            "metadata": build_report_metadata(torch.device(args.device), batches, " ".join(sys.argv)),
            "cases": reports,
        },
        args.output,
    )
    print(Path(args.output))


if __name__ == "__main__":
    main()
