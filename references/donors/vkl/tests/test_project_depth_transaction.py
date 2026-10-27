import runpy
from pathlib import Path

import torch


def test_make_output_buffers_matches_project_depth_output_contract():
    recipe = runpy.run_path(Path(__file__).parents[1] / "recipes/triton/bench_project_depth_transaction.py")
    outputs = {
        "eligible": torch.empty((2, 3), dtype=torch.bool),
        "projected_valid": torch.empty((2, 3), dtype=torch.bool),
        "sensor_valid": torch.empty((2, 3), dtype=torch.bool),
        "support": torch.empty((2, 3), dtype=torch.bool),
        "conflict": torch.empty((2, 3), dtype=torch.bool),
        "depth_residual": torch.empty((2, 3)),
    }

    actual = recipe["make_output_buffers"](outputs)

    assert set(actual) == set(outputs)
    assert all(actual[name].shape == value.shape for name, value in outputs.items())
    assert all(actual[name].dtype == value.dtype for name, value in outputs.items())
    assert all(actual[name] is not value for name, value in outputs.items())
