from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from vkl.data import FrameSample

ROOT = Path(__file__).resolve().parents[1]


def _recipe_module():
    path = ROOT / "recipes/splatograph/bench_prepared_ingress.py"
    spec = importlib.util.spec_from_file_location("bench_prepared_ingress", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_ingress_recipe_records_cpu_completion_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("VKL_SOURCE_REVISION", "test-source")
    monkeypatch.setenv("VKL_IMAGE_REVISION", "test-image")
    monkeypatch.setenv("VKL_IMAGE_REFERENCE", "localhost/vkl:test")
    output = tmp_path / "prepared_ingress.json"

    result = _recipe_module().run_prepared_ingress(
        output=output,
        device="cpu",
        ring_sizes=(1,),
        batches=(1,),
        height=8,
        width=8,
        warmup=0,
        iterations=2,
    )

    recorded = json.loads(output.read_text())
    assert recorded == result
    assert result["execution_modes"] == ["serialized", "queued"]
    assert result["source_revision"] == "test-source"
    assert result["image_revision"] == "test-image"
    assert result["image_reference"] == "localhost/vkl:test"
    assert {record["completion_event"] for record in result["records"]} == {None}
    assert {
        "max_pending_results",
        "incremental_peak_allocated_bytes",
        "transaction_p50_ms",
        "transaction_p95_ms",
        "transaction_p99_ms",
        "wall_mean_ms",
        "sustained_frames_per_s",
        "h2d_mean_ms",
        "h2d_p50_ms",
        "h2d_p95_ms",
        "h2d_p99_ms",
    } <= result["records"][0].keys()
    assert result["records"][0]["wall_mean_ms"] > 0
    assert result["records"][0]["sustained_frames_per_s"] > 0


def test_prepared_ingress_recipe_records_distinct_rgb_and_depth_shapes(tmp_path):
    result = _recipe_module().run_prepared_ingress(
        output=tmp_path / "prepared_ingress_mixed.json",
        device="cpu",
        ring_sizes=(1,),
        batches=(1,),
        height=8,
        width=10,
        depth_height=4,
        depth_width=5,
        warmup=0,
        iterations=1,
    )

    assert result["rgb_hw"] == [8, 10]
    assert result["depth_hw"] == [4, 5]
    assert {tuple(record["rgb_hw"]) for record in result["records"]} == {(8, 10)}
    assert {tuple(record["depth_hw"]) for record in result["records"]} == {(4, 5)}


def test_prepared_ingress_recipe_uses_real_tum_reader_shapes_and_provenance(tmp_path, monkeypatch):
    module = _recipe_module()
    calls = []

    def source(root, *, calibration, drop_unmatched):
        calls.append((root, calibration, drop_unmatched))
        intrinsics = torch.eye(3)
        pose = torch.eye(4)
        return iter(
            FrameSample(
                rgb=torch.full((3, 4, 5), 0.5),
                depth=torch.ones((1, 2, 3)),
                pose=pose if index else None,
                intrinsics=intrinsics,
                depth_intrinsics=intrinsics,
                target_intrinsics=intrinsics,
                transform_rgb_depth=pose,
                timestamp=float(index),
                frame_id=str(index),
            )
            for index in range(5)
        )

    monkeypatch.setattr(module, "TumRGBDLoader", source)
    result = module.run_prepared_ingress(
        output=tmp_path / "prepared_ingress_tum.json",
        device="cpu",
        ring_sizes=(1,),
        batches=(1,),
        warmup=0,
        iterations=1,
        tum_root=Path("/datasets/tum"),
        calibration="freiburg3",
    )

    assert result["source"] == {
        "kind": "tum_rgbd",
        "root": "/datasets/tum",
        "calibration": "freiburg3",
        "camera_conditioning": "single_camera_identity",
    }
    assert result["rgb_hw"] == [4, 5]
    assert result["depth_hw"] == [2, 3]
    assert calls and all(call == (Path("/datasets/tum"), "freiburg3", True) for call in calls)
