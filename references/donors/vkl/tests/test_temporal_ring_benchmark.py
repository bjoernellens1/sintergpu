from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_path = Path(__file__).resolve().parents[1] / "recipes/temporal/bench_temporal_ring.py"
_spec = importlib.util.spec_from_file_location("temporal_ring_benchmark", _path)
assert _spec is not None and _spec.loader is not None
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)


def test_summary_interpolates_percentiles_and_has_required_schema():
    assert _recipe._summary([1.0, 3.0]) == pytest.approx(
        {
            "mean_ms": 2.0,
            "p50_ms": 2.0,
            "p95_ms": 2.9,
            "p99_ms": 2.98,
            "min_ms": 1.0,
            "max_ms": 3.0,
        }
    )


def test_cpu_synthetic_ring_records_timing_scope_and_stages(tmp_path: Path):
    result = _recipe.run_temporal_ring(
        output=tmp_path / "temporal.json",
        device="cpu",
        batches=(1,),
        neighbors=(2,),
        height=16,
        width=16,
        warmup=0,
        iterations=1,
    )

    assert result["timing_scope"] == "preallocated_gpu_resident_temporal_ring"
    assert result["source_revision"] == "unknown"
    assert [record["mode"] for record in result["records"]] == ["latency", "saturation"]
    record = result["records"][0]
    assert record["compile_actual"] == "eager"
    assert record["compile_error"] is None
    assert record["stage_timing_scope"] == "eager_stage_events"
    assert set(record["stage_ms"]) == {"t0_reproject", "t1_compare", "t2_best", "t3_restore", "transaction"}


def test_tum_ring_skips_frames_without_matched_pose(monkeypatch: pytest.MonkeyPatch):
    intrinsics = torch.eye(3)
    valid = SimpleNamespace(
        rgb=torch.rand(3, 16, 16), depth=torch.ones(1, 16, 16), intrinsics=intrinsics, pose=torch.eye(4)
    )
    missing_pose = SimpleNamespace(rgb=torch.rand(3, 16, 16), depth=torch.ones(1, 16, 16), intrinsics=intrinsics, pose=None)
    monkeypatch.setattr(_recipe, "TumRGBDLoader", lambda *args, **kwargs: [missing_pose, valid, valid, valid])

    ring = _recipe._tum_ring(
        root=Path("/unused"), calibration="freiburg3", batch=1, neighbors=2, ring_size=1,
        height=16, width=16, device=torch.device("cpu"),
    )

    assert len(ring) == 1
