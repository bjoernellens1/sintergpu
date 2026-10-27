import pytest
import torch

from vkl.kernels.reference.contract import validate_frame_batch


def test_frame_batch_requires_rgb_and_preserves_optional_capabilities():
    batch = {
        "rgb": torch.zeros(2, 3, 4, 5),
        "timestamp": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "valid_pose": torch.tensor([True, False]),
    }
    out = validate_frame_batch(batch)
    assert out is batch
    assert set(out) == {"rgb", "timestamp", "valid_pose"}


def test_frame_batch_accepts_mixed_resolution_raw_rgbd_and_camera_calibration():
    batch = {
        "rgb": torch.zeros(2, 3, 8, 10),
        "depth": torch.ones(2, 1, 4, 5),
        "intrinsics": torch.eye(3).repeat(2, 1, 1),
        "depth_intrinsics": torch.eye(3).repeat(2, 1, 1),
        "target_intrinsics": torch.eye(3).repeat(2, 1, 1),
        "transform_rgb_depth": torch.eye(4).repeat(2, 1, 1),
    }
    assert validate_frame_batch(batch) is batch


@pytest.mark.parametrize("field, value", [
    ("depth", torch.zeros(2, 2, 4, 5)),
    ("intrinsics", torch.zeros(2, 4, 3)),
    ("pose_c2w", torch.zeros(2, 3, 4)),
    ("timestamp", torch.zeros(1, dtype=torch.float64)),
])
def test_frame_batch_rejects_wrong_optional_shape(field, value):
    with pytest.raises(ValueError):
        validate_frame_batch({"rgb": torch.zeros(2, 3, 4, 5), field: value})


def test_frame_batch_rejects_nonfinite_intrinsics_and_missing_rgb():
    with pytest.raises(ValueError, match="rgb"):
        validate_frame_batch({"depth": torch.ones(1, 1, 2, 2)})
    with pytest.raises(ValueError, match="finite"):
        validate_frame_batch({"rgb": torch.zeros(1, 3, 2, 2), "intrinsics": torch.full((1, 3, 3), float("nan"))})


def test_frame_batch_timestamp_is_finite_float64_independent_of_rgb_dtype():
    rgb = torch.zeros(2, 3, 2, 2, dtype=torch.float32)
    timestamp = torch.tensor([1_305_031_102.500, 1_305_031_102.533], dtype=torch.float64)
    assert validate_frame_batch({"rgb": rgb, "timestamp": timestamp})["timestamp"].dtype == torch.float64
    with pytest.raises(ValueError, match="float64"):
        validate_frame_batch({"rgb": rgb, "timestamp": timestamp.float()})
    bad = timestamp.clone(); bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_frame_batch({"rgb": rgb, "timestamp": bad})


def test_frame_batch_checks_metadata_and_c2w_rotation():
    base = {"rgb": torch.zeros(1, 3, 2, 2), "depth_scale": 0.001, "pose_c2w": torch.eye(4).unsqueeze(0)}
    assert validate_frame_batch(base) is base
    with pytest.raises(ValueError):
        validate_frame_batch({**base, "depth_scale": -1})
    with pytest.raises(ValueError):
        validate_frame_batch({**base, "timestamp": [1.0]})


def test_frame_batch_rejects_non_pinhole_or_non_rigid_cameras():
    rgb = torch.zeros(1, 3, 2, 2); eye = torch.eye(4).unsqueeze(0)
    bad_k = torch.eye(3).unsqueeze(0); bad_k[:, 0, 0] = 0
    with pytest.raises(ValueError): validate_frame_batch({"rgb": rgb, "intrinsics": bad_k})
    bad_pose = eye.clone(); bad_pose[:, 0, 0] = 2
    with pytest.raises(ValueError): validate_frame_batch({"rgb": rgb, "pose_c2w": bad_pose})
    for k22 in (0.99, 1.01):
        noncanonical_k = torch.eye(3).unsqueeze(0)
        noncanonical_k[0, 2, 2] = k22
        with pytest.raises(ValueError):
            validate_frame_batch({"rgb": rgb, "intrinsics": noncanonical_k})


def test_frame_batch_rejects_depth_calibration_without_depth():
    rgb = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError, match="depth calibration"):
        validate_frame_batch({"rgb": rgb, "depth_intrinsics": torch.eye(3).unsqueeze(0)})
