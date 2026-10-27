from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _recipe_module():
    path = ROOT / "recipes/tracking/run_tum_rgbd_odometry_map.py"
    spec = importlib.util.spec_from_file_location("run_tum_rgbd_odometry_map", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_groundtruth_association_does_not_require_frame_decode() -> None:
    first = torch.eye(4)
    second = torch.eye(4)
    second[0, 3] = 1.0

    out = _recipe_module()._associate_groundtruth_frame_ids(
        ["1341847980.700000", "1341847980.900000", "unparseable"],
        {1341847980.71: first, 1341847980.91: second},
        max_dt=0.02,
    )

    assert out[0] is first
    assert out[1] is second
    assert out[2] is None


def test_reason_histogram_preserves_compact_reason_counts() -> None:
    out = _recipe_module()._reason_histogram(torch.tensor([0, 16, 16, 4], dtype=torch.int64))

    assert out == {"0": 1, "4": 1, "16": 2}
