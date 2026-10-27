from __future__ import annotations

import importlib.util
import struct
import sys
from io import BytesIO
from pathlib import Path

import torch

_path = Path(__file__).resolve().parents[1] / "recipes/splatograph/run_scannet_registration_control.py"
_spec = importlib.util.spec_from_file_location("scannet_registration_control", _path)
assert _spec and _spec.loader
_recipe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _recipe
_spec.loader.exec_module(_recipe)


def test_parse_sens_header_preserves_distinct_calibrated_grids():
    matrix = torch.eye(4).flatten().tolist()
    blob = b"".join(
        (
            struct.pack("<I", 4),
            struct.pack("<Q", 3),
            b"rgb",
            *(struct.pack("<16f", *matrix) for _ in range(4)),
            struct.pack("<II", 2, 1),
            struct.pack("<IIII", 1296, 968, 640, 480),
            struct.pack("<f", 1000.0),
            struct.pack("<Q", 2),
        )
    )
    header = _recipe.parse_sens_header(BytesIO(blob))

    assert header["sensor_name"] == "rgb"
    assert header["color_hw"] == (968, 1296)
    assert header["depth_hw"] == (480, 640)
    assert header["depth_shift"] == 1000.0
    assert header["frame_count"] == 2
    assert torch.equal(header["color_intrinsics"], torch.eye(3))
