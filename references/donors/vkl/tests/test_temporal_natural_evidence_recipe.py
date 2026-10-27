from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


_spec = importlib.util.spec_from_file_location(
    "temporal_natural_evidence_recipe",
    Path(__file__).resolve().parents[1] / "recipes/temporal/run_tum_natural_evidence.py",
)
assert _spec is not None and _spec.loader is not None
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)


def test_natural_summary_reports_evidence_without_clean_oracle_claim():
    compared = {
        "temporal_rgb_consistency": torch.tensor([[0.8, 0.6]]),
        "motion_blur_evidence": torch.tensor([[0.2, 0.4]]),
        "restoration_support": torch.tensor([[0.3, 0.5]]),
    }
    restored = {
        "restore_confidence": torch.tensor([[[[0.5, 0.7]]]]),
        "restore_valid": torch.tensor([[[[True, False]]]]),
        "restore_gain": torch.tensor([[[[0.02, 0.04]]]]),
    }
    summary = _recipe._natural_summary(compared, restored, alignment_valid_fraction=0.75)

    assert summary == pytest.approx(
        {
            "alignment_valid_fraction": 0.75,
            "temporal_rgb_consistency": 0.7,
            "motion_blur_evidence": 0.3,
            "restoration_support": 0.4,
            "restore_confidence": 0.6,
            "restore_valid_fraction": 0.5,
            "restore_gain": 0.03,
        }
    )
