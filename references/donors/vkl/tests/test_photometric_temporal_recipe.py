import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vkl.bench.photometric import fit_online_exposure_gain, fit_online_exposure_gain_response


_spec = importlib.util.spec_from_file_location(
    "photometric_temporal_recipe",
    Path(__file__).resolve().parents[1] / "recipes/photometric/run_temporal_recovery.py",
)
assert _spec is not None and _spec.loader is not None
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)


def test_temporal_recipe_unsupported_frame_preserves_ev_wb_state():
    device = torch.device("cpu")
    truth = _recipe.synthetic_sequence(frames=5, height=4, width=4, device=device)
    control, index = _recipe._unsupported_observation(truth, index=2)

    fitted = fit_online_exposure_gain(
        control["clean"],
        control["observed"],
        updates_per_frame=2,
        initialization="log_ratio",
    )
    metrics = _recipe._state_preservation(fitted, unsupported_frame=index)

    assert metrics["trusted_fraction"] == 0
    assert metrics["initializer_applied"] is False
    assert metrics["refinement_applied"] is False
    assert metrics["ev_state_delta"] == pytest.approx(0)
    assert metrics["wb_log_state_delta"] == pytest.approx(0)


def test_temporal_recipe_unsupported_frame_preserves_shared_response_state():
    device = torch.device("cpu")
    truth = _recipe.synthetic_sequence(
        frames=5,
        height=4,
        width=4,
        device=device,
        response_gamma=0.82,
    )
    control, index = _recipe._unsupported_observation(truth, index=2)

    fitted = fit_online_exposure_gain_response(
        control["clean"],
        control["observed"],
        updates_per_frame=1,
    )
    metrics = _recipe._state_preservation(fitted, unsupported_frame=index)

    assert metrics["trusted_fraction"] == 0
    assert metrics["response_refinement_applied"] is False
    assert metrics["ev_state_delta"] == pytest.approx(0)
    assert metrics["wb_log_state_delta"] == pytest.approx(0)
    assert metrics["response_state_delta"] == pytest.approx(0)


def test_temporal_recipe_timing_is_end_to_end_per_frame_wall_time():
    fitted, timing = _recipe._timed_fit(
        lambda: {"value": torch.tensor(1.0)},
        device=torch.device("cpu"),
        frames=4,
    )

    assert fitted["value"].item() == 1
    assert timing["wall_ms_total"] >= 0
    assert timing["wall_ms_per_frame"] == pytest.approx(timing["wall_ms_total"] / 4)
