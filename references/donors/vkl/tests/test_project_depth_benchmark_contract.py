from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from vkl.kernels import project_depth_evidence


_spec = importlib.util.spec_from_file_location(
    "project_depth_bench",
    Path(__file__).resolve().parents[1] / "recipes/splatograph/bench_project_depth_evidence.py",
)
assert _spec is not None and _spec.loader is not None
_bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bench
_spec.loader.exec_module(_bench)


def test_project_depth_benchmark_inputs_are_reproducible_under_recorded_seed():
    torch.manual_seed(7)
    first = _bench._inputs(2, 16, 12, 16, torch.device("cpu"))
    torch.manual_seed(7)
    second = _bench._inputs(2, 16, 12, 16, torch.device("cpu"))

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_project_depth_benchmark_reports_workload_shape_and_serialized_rate():
    torch.manual_seed(0)
    values = _bench._inputs(1, 32, 12, 16, torch.device("cpu"))
    output = _bench._invoke(project_depth_evidence, values)
    workload = _bench._workload(output)
    summary = _bench._summary([2.0, 2.0, 2.0], points=100)

    for name in ("eligible", "projected_valid", "sensor_valid", "support", "conflict"):
        assert 0.0 <= workload[f"{name}_fraction"] <= 1.0
    assert workload["sensor_residual_mean"] >= 0
    assert workload["sensor_residual_p95"] >= 0
    assert summary["mean_ms"] == 2.0
    assert summary["serialized_points_per_s"] == 50_000.0


def test_project_depth_benchmark_parity_quantifies_boundary_differences():
    eager = {
        "support": torch.tensor([[True, False]]),
        "depth_residual": torch.tensor([[0.0, 1.0]]),
    }
    compiled = {
        "support": torch.tensor([[True, False]]),
        "depth_residual": torch.tensor([[0.0, 1.01]]),
    }

    parity = _bench._parity(eager, compiled)

    assert parity["boolean_exact"]
    assert parity["boolean_mismatch_count"] == 0
    assert not parity["residual_allclose"]
    assert parity["residual_mismatch_count"] == 1
    assert parity["residual_mismatch_fraction"] == 0.5
    assert parity["residual_max_abs_error"] > 0


def test_project_depth_benchmark_memory_is_cpu_none():
    torch.manual_seed(0)
    values = _bench._inputs(1, 8, 8, 8, torch.device("cpu"))
    assert _bench._memory(project_depth_evidence, values) is None
