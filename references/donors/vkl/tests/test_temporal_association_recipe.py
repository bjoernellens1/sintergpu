from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


_spec = importlib.util.spec_from_file_location(
    "temporal_association_recipe",
    Path(__file__).resolve().parents[1]
    / "recipes/keypoints/benchmark_temporal_association.py",
)
assert _spec is not None and _spec.loader is not None
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)


def test_world_cell_count_deduplicates_valid_observations_across_references():
    points = [
        torch.tensor(
            [[[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [0.21, 0.01, 0.01]]]
        ),
        torch.tensor(
            [[[0.03, 0.02, 0.02], [0.22, 0.01, 0.01], [float("nan"), 4.0, 4.0]]]
        ),
    ]
    valid = [
        torch.tensor([[True, True, True]]),
        torch.tensor([[True, True, False]]),
    ]

    assert _recipe._world_cell_count(points, valid, voxel_size=0.1) == 2


def test_world_cell_count_rejects_nonfinite_or_nonpositive_voxel_size():
    with pytest.raises(ValueError, match="voxel_size"):
        _recipe._world_cell_count([], [], voxel_size=float("nan"))
    with pytest.raises(ValueError, match="voxel_size"):
        _recipe._world_cell_count([], [], voxel_size=0.0)


def test_projection_consistency_masks_depth_matches_outside_pose_projection_tolerance():
    out = _recipe._projection_consistent_valid(
        torch.tensor([[[10.2, 8.1], [18.0, 9.0]]]),
        torch.tensor([[[10.0, 8.0], [10.0, 8.0]]]),
        torch.tensor([[True, True]]),
        tolerance_px=1.0,
    )

    assert out.tolist() == [[True, False]]


def test_projection_consistency_rejects_nonfinite_tolerance():
    with pytest.raises(ValueError, match="finite positive"):
        _recipe._projection_consistent_valid(
            torch.tensor([[[10.0, 8.0]]]),
            torch.tensor([[[10.0, 8.0]]]),
            torch.tensor([[True]]),
            tolerance_px=float("nan"),
        )


def test_depth_consistency_is_finite_and_respects_association_validity():
    consistent, relative_error = _recipe._depth_consistency(
        torch.tensor([[2.0, 2.2, float("nan")]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[2.0, 2.0, 2.0]]),
        torch.tensor([[True, False, True]]),
        max_relative_depth_error=0.05,
    )

    assert consistent.tolist() == [[True, False, False]]
    assert relative_error[0, 0].item() == pytest.approx(0.0)
    assert torch.isnan(relative_error[0, 2])
