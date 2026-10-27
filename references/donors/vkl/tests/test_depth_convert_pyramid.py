import pytest
import torch

from vkl.kernels import REGISTRY
from vkl.kernels.reference import depth_convert, depth_pyramid


def test_depth_convert_metric_depth_and_inverse_depth_preserve_values_and_status():
    raw = torch.tensor([[[[1000.0, 2000.0], [0.0, float("nan")]]]])

    result = depth_convert(values=raw, representation="DEPTH", scale=0.001)

    assert torch.allclose(result["depth"][0, 0, :1], torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(result["inverse_depth"][0, 0, :1], torch.tensor([[1.0, 0.5]]))
    assert result["valid"].tolist() == [[[[True, True], [False, False]]]]
    assert result["metric_status"] == "METRIC_ANCHORED"


def test_depth_convert_disparity_requires_named_metric_calibration():
    disparity = torch.tensor([[[[20.0, 10.0]]]])

    result = depth_convert(values=disparity, representation="DISPARITY", focal_length=100.0, baseline=0.1)

    assert torch.allclose(result["depth"], torch.tensor([[[[0.5, 1.0]]]]))
    assert result["metric_status"] == "METRIC_ANCHORED"
    with pytest.raises(ValueError, match="focal_length and baseline"):
        depth_convert(values=disparity, representation="DISPARITY")


def test_depth_convert_rejects_unanchored_metric_claims():
    with pytest.raises(ValueError, match="metric"):
        depth_convert(values=torch.ones(1, 1, 1, 1), representation="DEPTH", metric_status="METRIC_ANCHORED")


def test_depth_convert_preserves_input_support_and_metric_estimate_provenance():
    values = torch.tensor([[[[2.0, 4.0, float("nan")]]]])
    valid = torch.tensor([[[[True, False, True]]]])
    confidence = torch.tensor([[[[0.2, 0.8, 0.6]]]])

    result = depth_convert(
        values=values,
        representation="DEPTH",
        valid=valid,
        confidence=confidence,
        metric_status="METRIC_ESTIMATED",
    )

    assert result["metric_status"] == "METRIC_ESTIMATED"
    assert result["valid"].tolist() == [[[[True, False, False]]]]
    assert torch.allclose(result["confidence"], torch.tensor([[[[0.2, 0.0, 0.0]]]]))
    with pytest.raises(TypeError, match="floating"):
        depth_convert(values=torch.ones(1, 1, 1, 1, dtype=torch.int16), representation="DEPTH")
    with pytest.raises(ValueError, match="scale"):
        depth_convert(values=torch.ones(1, 1, 1, 1), representation="INVERSE_DEPTH", metric_status="METRIC_ANCHORED")


def test_depth_pyramid_does_not_average_holes_or_depth_edges():
    depth = torch.tensor([[[[1.0, 1.0, 9.0, 9.0], [1.0, 0.0, 9.0, 9.0], [1.0, 1.0, 9.0, 9.0]]]])
    valid = depth != 0
    confidence = valid.to(torch.float32)

    levels = depth_pyramid(depth=depth, valid=valid, confidence=confidence)

    assert set(levels) == {"/1", "/2", "/4"}
    half = levels["/2"]
    assert half["valid"].tolist() == [[[[True, True], [True, True]]]]
    assert half["depth"][0, 0, 0, 0] == 1.0
    assert half["depth"][0, 0, 0, 1] == 9.0
    assert half["confidence"][0, 0, 0, 0] == 1.0


def test_depth_pyramid_bounded_odd_shapes_and_registry():
    levels = depth_pyramid(depth=torch.ones(1, 1, 5, 9), valid=torch.ones(1, 1, 5, 9, dtype=torch.bool))

    assert [levels[key]["depth"].shape[-2:] for key in levels] == [(5, 9), (3, 5), (2, 3), (1, 2)]
    assert REGISTRY["depth_convert"] is depth_convert
    assert REGISTRY["depth_pyramid"] is depth_pyramid


def test_depth_pyramid_preserves_selected_support_and_rejects_invalid_confidence():
    depth = torch.tensor([[[[10.0, 1.0]]]])
    valid = torch.tensor([[[[True, True]]]])
    confidence = torch.tensor([[[[0.9, 0.2]]]])

    half = depth_pyramid(depth=depth, valid=valid, confidence=confidence)["/2"]

    assert half["depth"].item() == 1.0
    assert half["confidence"].item() == pytest.approx(0.2)
    holes = depth_pyramid(
        depth=torch.tensor([[[[float("nan"), 2.0]]]]),
        valid=torch.tensor([[[[True, False]]]]),
        confidence=torch.tensor([[[[0.2, 1.0]]]]),
    )["/2"]
    assert not holes["valid"].item()
    assert holes["depth"].item() == 0.0
    assert holes["confidence"].item() == 0.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        depth_pyramid(depth=depth, valid=valid, confidence=torch.full_like(depth, 2.0))
