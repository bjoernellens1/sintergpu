import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vkl.kernels import photometric_isp


_spec = importlib.util.spec_from_file_location(
    "photometric_bench",
    Path(__file__).resolve().parents[1] / "recipes/photometric/bench_photometric_isp.py",
)
assert _spec is not None and _spec.loader is not None
_bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bench
_spec.loader.exec_module(_bench)


def test_ev_wb_benchmark_uses_three_latent_scalars_and_backpropagates():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    state = _bench._state(rgb, "ev_wb", train=True)

    output = photometric_isp(
        rgb,
        validate=False,
        evidence=False,
        **_bench._arguments(state),
    )["rgb_observed"]
    output.square().mean().backward()

    assert tuple(state) == ("exposure_ev", "chroma_rg_bg")
    assert sum(value.numel() for value in state.values()) == 3
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in state.values())


def test_full_benchmark_uses_constrained_latents_and_backpropagates():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    state = _bench._state(rgb, "full_current", train=True)

    output = photometric_isp(
        rgb,
        validate=False,
        evidence=False,
        **_bench._arguments(state),
    )["rgb_observed"]
    output.square().mean().backward()

    assert tuple(state) == (
        "exposure_ev",
        "chroma_rg_bg",
        "color_offdiag",
        "vignette_raw",
        "response_raw",
    )
    assert sum(value.numel() for value in state.values()) == 19
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in state.values())

    arguments = _bench._arguments(state)
    assert torch.allclose(
        torch.diagonal(arguments["color_matrix"], dim1=-2, dim2=-1),
        torch.ones(1, 3),
    )
    assert torch.all(arguments["vignette"] >= 0)
    assert torch.all(arguments["response_gamma"] > 0)


def test_latency_summary_does_not_claim_sustained_throughput():
    summary = _bench._summary([2.0, 2.0], batch=4)

    assert summary["latency_derived_frames_per_s"] == pytest.approx(2000.0)
    assert "frames_per_s" not in summary


def test_cpu_benchmark_memory_helpers_are_noops():
    device = torch.device("cpu")
    _bench._reset_peak(device)
    assert _bench._peak(device) == {}


def test_benchmark_rejects_invalid_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_photometric_isp.py", "--output", str(tmp_path / "result.json"), "--height", "0"],
    )

    with pytest.raises(ValueError, match="resolution"):
        _bench.main()


def test_benchmark_rejects_unknown_variant_before_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_photometric_isp.py",
            "--output",
            str(tmp_path / "result.json"),
            "--variants",
            "not_a_variant",
        ],
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("allocated")))

    with pytest.raises(ValueError, match="variants"):
        _bench.main()
