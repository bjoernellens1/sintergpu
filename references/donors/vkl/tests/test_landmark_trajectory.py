from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "landmark_trajectory_recipe",
    Path(__file__).resolve().parents[1] / "recipes/keypoints/benchmark_landmark_trajectory.py",
)
assert _spec is not None and _spec.loader is not None
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)

from vkl.data.consistent import wall_floor_boxes
from vkl.landmarks import PersistentLandmarkStore


def test_summarize_separates_forced_dropout_from_lifetime_totals():
    records = [
        {"frame": 0, "created": 3, "matched": 0, "missed": 0, "retired": 0, "forced_created": 0, "unforced_created": 3, "forced_matched": 0, "unforced_matched": 0, "forced_missed": 0, "unforced_missed": 0, "forced_retired": 0, "unforced_retired": 0, "active": 3, "dropout_control_active": False},
        {"frame": 1, "created": 0, "matched": 2, "missed": 1, "retired": 0, "forced_created": 0, "unforced_created": 0, "forced_matched": 0, "unforced_matched": 2, "forced_missed": 0, "unforced_missed": 1, "forced_retired": 0, "unforced_retired": 0, "active": 3, "dropout_control_active": False},
        {"frame": 2, "created": 1, "matched": 0, "missed": 2, "retired": 1, "forced_created": 0, "unforced_created": 1, "forced_matched": 0, "unforced_matched": 0, "forced_missed": 1, "unforced_missed": 1, "forced_retired": 1, "unforced_retired": 0, "active": 2, "dropout_control_active": True},
    ]
    summary = _recipe._summarize(records, dropout_period=2)
    assert summary["frames"] == 3
    assert summary["total"] == {"created": 4, "matched": 2, "missed": 3, "retired": 1}
    assert summary["forced_control"]["missed"] == 1
    assert summary["forced_control"]["retired"] == 1
    assert summary["forced_control"]["retirement_per_created"] is None
    assert summary["unforced"]["created"] == 4
    assert summary["unforced"]["matched"] == 2
    assert summary["unforced"]["missed"] == 2
    assert summary["unforced"]["retired"] == 0
    assert summary["total"]["missed"] == summary["forced_control"]["missed"] + summary["unforced"]["missed"]
    assert summary["control_dropout_frames"] == 1
    assert summary["final_active"] == 2
    assert summary["diagnostics"] == {
        "depth_rejected": 0,
        "duplicate_suppressed": 0,
        "capacity_dropped": 0,
    }


def test_summarize_rejects_empty_trajectory():
    with pytest.raises(ValueError, match="no records"):
        _recipe._summarize([], dropout_period=2)


def test_summarize_reports_lifecycle_stage_wall_clock_profile():
    records = [
        {
            "frame": index,
            "created": 0,
            "matched": 0,
            "missed": 0,
            "retired": 0,
            "forced_created": 0,
            "unforced_created": 0,
            "forced_matched": 0,
            "unforced_matched": 0,
            "forced_missed": 0,
            "unforced_missed": 0,
            "forced_retired": 0,
            "unforced_retired": 0,
            "active": 0,
            "dropout_control_active": False,
            "timing_ms": {
                "association": 1.0 + index,
                "store_update": 2.0 + index,
                "retire": 3.0 + index,
                "creation": 4.0 + index,
                "step": 10.0 + index,
            },
        }
        for index in range(3)
    ]

    profile = _recipe._summarize(records, dropout_period=2)["timing_ms"]

    assert profile["association"] == {"mean": 2.0, "p50": 2.0, "p95": 3.0, "p99": 3.0}
    assert profile["store_update"]["p50"] == 3.0
    assert profile["retire"]["p95"] == 5.0
    assert profile["creation"]["p99"] == 6.0
    assert profile["step"]["mean"] == 11.0


def test_stage_timing_synchronizes_only_accelerated_device(monkeypatch):
    calls = []
    monkeypatch.setattr(_recipe.torch.cuda, "synchronize", lambda device: calls.append(device))

    _recipe._synchronize(torch.device("cpu"))
    _recipe._synchronize(torch.device("cuda"))

    assert calls == [torch.device("cuda")]


def test_depth_consistency_rejects_foreground_background_alias():
    measured = torch.tensor([[3.0, 2.02]])
    projected = torch.tensor([[2.0, 2.0]])
    valid = torch.tensor([[True, True]])

    out = _recipe._depth_consistent(
        measured,
        projected,
        valid,
        max_relative_depth_error=0.05,
    )

    assert out.tolist() == [[False, True]]


def test_new_landmark_creation_suppresses_existing_and_same_frame_duplicates():
    store = PersistentLandmarkStore(batch_size=1, capacity=6)
    store.create(
        torch.tensor([[[1.0, 0.0, 2.0]]]),
        valid=torch.tensor([[True]]),
    )
    candidates = torch.tensor(
        [[[1.01, 0.0, 2.0], [1.20, 0.0, 2.0], [1.21, 0.0, 2.0], [1.40, 0.0, 2.0]]]
    )
    valid = torch.tensor([[True, True, True, True]])

    novel = _recipe._novel_creation_mask(
        store,
        candidates,
        valid,
        min_distance_m=0.05,
    )

    assert novel.tolist() == [[False, True, False, True]]


def test_pose_observable_creation_requires_a_previous_pose_baseline():
    frame = wall_floor_boxes(height=32, width=48)
    without_baseline = _recipe._select(
        "pose_observable",
        frame["rgb"],
        frame["intrinsics"],
        frame["pose_c2w"],
        None,
        8,
    )
    neighbor_pose = frame["pose_c2w"].clone()
    neighbor_pose[:, 0, 3] += 0.08
    with_baseline = _recipe._select(
        "pose_observable",
        frame["rgb"],
        frame["intrinsics"],
        frame["pose_c2w"],
        neighbor_pose,
        8,
    )

    assert without_baseline["pixels_xy"].shape == (1, 8, 2)
    assert not without_baseline["valid"].any()
    assert with_baseline["valid"].any()


def test_step_exercises_create_update_miss_and_retire_on_control_fixture():
    frame = wall_floor_boxes(height=32, width=48)
    store = PersistentLandmarkStore(batch_size=1, capacity=16)
    records = []
    for index in range(4):
        records.append(_recipe._step(
            store,
            frame_index=index,
            rgb=frame["rgb"],
            depth=frame["depth"],
            intrinsics=frame["intrinsics"],
            pose=frame["pose_c2w"],
            selector="dso",
            points=8,
            max_missed=1,
            dropout_period=2,
            create_interval=1,
        ))
    summary = _recipe._summarize(records, dropout_period=2)
    assert summary["forced_control"]["missed"] > 0
    assert summary["forced_control"]["retired"] > 0
    for key in ("created", "matched", "missed", "retired"):
        assert summary["total"][key] == summary["forced_control"][key] + summary["unforced"][key]
    assert summary["control_dropout_frames"] == 2
    assert torch.equal(store.active, store.landmark_id >= 0)
