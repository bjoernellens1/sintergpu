from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_spec = importlib.util.spec_from_file_location(
    "camera_registration_bench",
    Path(__file__).resolve().parents[1] / "recipes/splatograph/bench_camera_registration.py",
)
assert _spec is not None and _spec.loader is not None
_bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bench
_spec.loader.exec_module(_bench)


def test_camera_registration_benchmark_summary_is_serialized_rate():
    summary = _bench._summary([2.0, 2.0, 2.0], pixels=2_000_000)
    assert summary["mean_ms"] == 2.0
    assert summary["serialized_megapixels_per_s"] == 1000.0


def test_camera_registration_benchmark_parity_quantifies_errors():
    eager = {"valid": torch.tensor([[[[True, False]]]]), "image": torch.tensor([[[[0.0, 1.0]]]])}
    compiled = {"valid": torch.tensor([[[[True, False]]]]), "image": torch.tensor([[[[0.0, 1.01]]]])}

    parity = _bench._parity(eager, compiled, "image")

    assert parity["valid_exact"]
    assert parity["valid_mismatch_count"] == 0
    assert not parity["value_allclose"]
    assert parity["value_mismatch_count"] == 1
    assert parity["value_max_abs_error"] > 0


def test_camera_rectify_benchmark_parity_allows_equivalent_sampler_rounding():
    eager = {"valid": torch.tensor([[[[True]]]]), "image": torch.tensor([[[[0.0]]]])}
    compiled = {"valid": torch.tensor([[[[True]]]]), "image": torch.tensor([[[[1.8e-4]]]])}

    parity = _bench._parity(eager, compiled, "image", atol=2e-4)

    assert parity["value_allclose"]


def test_camera_registration_benchmark_cpu_memory_is_none():
    assert _bench._memory(lambda: None, torch.device("cpu")) is None


def test_camera_registration_benchmark_records_compile_unsupported_without_dropping_eager_measurement():
    eager = {"valid": torch.tensor([[[[True]]]]), "image": torch.tensor([[[[0.5]]]])}

    record = _bench._record(
        "camera_rectify",
        lambda: eager,
        None,
        eager,
        None,
        compile_error="Unsupported: cudnn_grid_sampler",
        warmup=1,
        iterations=1,
        pixels=1,
        device=torch.device("cpu"),
    )

    assert record["compile_requested"]
    assert record["compile_actual"] == "unsupported"
    assert record["compile_error"] == "Unsupported: cudnn_grid_sampler"
    assert record["compiled"] is None
    assert record["eager"]["mean_ms"] >= 0
