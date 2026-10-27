import pytest
import torch

from vkl.kernels import REGISTRY, align_depth, fuse_depth


def test_align_scale_only_recovers_metric_scale():
    pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    anchor = pred * 2.5
    out = align_depth(pred, anchor, torch.ones_like(pred, dtype=torch.bool))
    assert torch.allclose(out["depth"], anchor)
    assert out["scale"].item() == pytest.approx(2.5)
    assert out["shift"].item() == 0
    assert out["metric_status"] == "METRIC_ANCHORED"


def test_align_affine_depth_and_inverse_depth():
    pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    anchor = 3 * pred + 0.5
    out = align_depth(pred, anchor, torch.ones_like(pred, dtype=torch.bool), mode="scale_shift_depth")
    assert torch.allclose(out["depth"], anchor, atol=1e-5)
    inverse_anchor = 1 / (2 * (1 / pred) + 0.1)
    inv_out = align_depth(pred, inverse_anchor, torch.ones_like(pred, dtype=torch.bool), mode="scale_shift_inverse_depth")
    assert torch.allclose(inv_out["depth"], inverse_anchor, atol=1e-5)


def test_align_excludes_invalid_anchors_and_confidence():
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    anchor = torch.tensor([[[[2.0, 100.0, 6.0]]]])
    mask = torch.tensor([[[[True, False, True]]]])
    out = align_depth(pred, anchor, mask, confidence=torch.ones_like(pred))
    assert torch.allclose(out["depth"], pred * 2, atol=1e-5)
    assert out["valid"].all()
    assert (out["confidence"] > 0).all()


def test_align_uses_anchor_weights_without_zeroing_dense_prediction_confidence():
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    anchors = torch.tensor([[[[2.0, 0.0, 0.0]]]])
    anchor_mask = anchors > 0
    out = align_depth(pred, anchors, anchor_mask, confidence=torch.tensor([[[[0.2, 0.0, 0.0]]]]))
    assert out["valid"].all()
    assert (out["confidence"] > 0).all()
    assert out["inlier_fraction"].item() == pytest.approx(1.0)


def test_align_robust_diagnostics_ignore_zero_weight_outliers():
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    anchors = torch.tensor([[[[2.0, 200.0, 300.0]]]])
    confidence = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    out = align_depth(pred, anchors, torch.ones_like(pred, dtype=torch.bool), confidence=confidence)
    assert out["scale"].item() == pytest.approx(2.0)
    assert out["fit_error"].item() == pytest.approx(0.0)
    assert out["inlier_fraction"].item() == pytest.approx(1.0)


def test_align_preserves_predicted_validity_in_dense_output():
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    anchors = pred * 2
    predicted_valid = torch.tensor([[[[True, False, True]]]])
    out = align_depth(pred, anchors, torch.ones_like(pred, dtype=torch.bool), predicted_valid=predicted_valid)
    assert torch.equal(out["valid"], predicted_valid)
    assert out["depth"][0, 0, 0, 1].item() == 0.0
    assert out["confidence"][0, 0, 0, 1].item() == 0.0


def test_align_scale_shift_rejects_degenerate_anchor_geometry():
    pred = torch.ones(1, 1, 1, 2)
    anchors = torch.tensor([[[[2.0, 3.0]]]])
    with pytest.raises(ValueError):
        align_depth(pred, anchors, torch.ones_like(pred, dtype=torch.bool), mode="scale_shift_depth")


def test_fuse_weights_consistent_and_selects_conflict_winner():
    a = torch.tensor([[[[2.0, 2.0]]]])
    b = torch.tensor([[[[2.02, 5.0]]]])
    valid = torch.ones_like(a, dtype=torch.bool)
    out = fuse_depth([a, b], [valid, valid], [torch.full_like(a, 0.8), torch.full_like(a, 0.2)])
    assert out["depth"].shape == a.shape
    assert out["depth"][0, 0, 0, 0].item() == pytest.approx(2.004)
    assert out["depth"][0, 0, 0, 1].item() == pytest.approx(2.0)
    assert out["winning_source"][0, 0, 0, 1].item() == 0


def test_fuse_invalid_pixel_has_no_winning_source():
    depth = torch.ones(1, 1, 1, 1)
    invalid = torch.zeros_like(depth, dtype=torch.bool)
    out = fuse_depth([depth, depth], [invalid, invalid], [depth, depth])
    assert not out["valid"].item()
    assert out["winning_source"].item() == -1


def test_fuse_rejects_mixed_metric_status_and_registers():
    d = torch.ones(1, 1, 1, 1)
    v = torch.ones_like(d, dtype=torch.bool)
    with pytest.raises(ValueError):
        fuse_depth([d, d], [v, v], [d, d], metric_status=["RELATIVE", "METRIC_ANCHORED"])
    assert fuse_depth([d, d], [v, v], [d, d], metric_status=["METRIC_ESTIMATED", "METRIC_ANCHORED"])["metric_status"] == "METRIC_ANCHORED"
    assert REGISTRY["align_depth"] is align_depth
    assert REGISTRY["fuse_depth"] is fuse_depth
