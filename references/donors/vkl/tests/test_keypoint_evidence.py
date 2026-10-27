import pytest
import torch

from vkl.kernels.reference.keypoint_evidence import FRAME_FEATURES, keypoint_evidence


def _inputs(dtype=torch.float32):
    selected = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    selected[0, 0, 0, 0] = selected[0, 0, 0, 3] = True
    selected[0, 0, 3, 0] = selected[0, 0, 3, 3] = True
    valid = torch.zeros_like(selected)
    valid[0, 0, 0, 0] = valid[0, 0, 0, 3] = True
    confidence = torch.zeros(1, 1, 4, 4, dtype=dtype)
    confidence[0, 0, 0, 0], confidence[0, 0, 0, 3] = 0.8, 0.4
    parallax = torch.where(valid, torch.tensor(2.0, dtype=dtype), torch.tensor(0.0, dtype=dtype))
    error = torch.zeros_like(confidence)
    gradient = selected.to(dtype)
    return selected, valid, confidence, parallax, error, gradient


def test_keypoint_evidence_reports_anchor_yield_and_spatial_coverage():
    selected, valid, confidence, parallax, error, gradient = _inputs()

    out = keypoint_evidence(
        selected=selected,
        valid=valid,
        confidence=confidence,
        parallax_px=parallax,
        photometric_error=error,
        gradient=gradient,
        tile_rows=2,
        tile_cols=2,
    )
    frame = out["frame_features"][0]
    assert out["counts"].tolist() == [[4.0, 2.0, 1.0]]
    assert frame[FRAME_FEATURES.index("anchor_yield")] == 0.5
    assert frame[FRAME_FEATURES.index("spatial_coverage")] == 0.5
    assert frame[FRAME_FEATURES.index("high_confidence_coverage")] == 0.25
    assert out["point_confidence"].shape == (1, 5, 4, 4)
    assert out["tile_features"].shape == (1, 8, 2, 2)


def test_keypoint_evidence_preserves_float_dtype():
    selected, valid, confidence, parallax, error, gradient = _inputs(torch.float64)
    out = keypoint_evidence(
        selected=selected,
        valid=valid,
        confidence=confidence,
        parallax_px=parallax,
        photometric_error=error,
        gradient=gradient,
    )
    assert out["point_confidence"].dtype == torch.float64
    assert out["tile_features"].dtype == torch.float64
    assert out["frame_features"].dtype == torch.float64
    assert out["counts"].dtype == torch.float64


def test_keypoint_evidence_rejects_non_bool_selection():
    selected, valid, confidence, parallax, error, gradient = _inputs()
    with pytest.raises(ValueError):
        keypoint_evidence(
            selected=selected.float(),
            valid=valid,
            confidence=confidence,
            parallax_px=parallax,
            photometric_error=error,
            gradient=gradient,
        )
