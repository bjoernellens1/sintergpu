import pytest
import torch

from vkl.kernels.reference.depth_evidence import depth_evidence


def test_depth_evidence_frame_and_grid_metadata():
    depth = torch.tensor([[[[1., 0., 9.], [1., 1., 9.], [1., 1., 9.]]]])
    valid = depth > 0
    out = depth_evidence(depth, valid, grid_size=(2, 2))
    assert out["valid_fraction"].shape == (1,)
    assert torch.allclose(out["valid_fraction"], torch.tensor([8 / 9]))
    assert out["hole_fraction"].item() == pytest.approx(1 / 9)
    assert out["grid_evidence"].shape[:3] == (1, 2, 2)
    assert torch.isfinite(out["grid_evidence"]).all()


def test_depth_evidence_confidence_and_invalid_pixels_are_zero_safe():
    depth = torch.ones(2, 1, 4, 4); valid = torch.ones_like(depth, dtype=torch.bool); valid[1] = False
    confidence = torch.full_like(depth, .5)
    out = depth_evidence(depth, valid, confidence=confidence)
    assert out["confidence_mean"].tolist() == [.5, 0.]
    assert out["stable_fraction"][1] == 0


def test_depth_evidence_rejects_mismatched_contract():
    with pytest.raises(ValueError): depth_evidence(torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2, 2, dtype=torch.bool))
    with pytest.raises(ValueError): depth_evidence(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool), confidence=torch.ones(1, 1, 2, 2, dtype=torch.float64))
    with pytest.raises(ValueError): depth_evidence(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool), confidence=torch.full((1, 1, 2, 2), 2.))

def test_evidence_uses_support_denominators_and_bounds_grid():
    d = torch.tensor([[[[1., 0.], [1., 0.]]]]); v = d > 0
    out = depth_evidence(d, v, grid_size=(8, 8))
    assert out["near_fraction"].item() == 0 and out["stable_fraction"].item() == 1
    assert out["grid_evidence"].shape[1:3] == (2, 2)

def test_holes_do_not_create_edges_and_variance_is_local():
    d = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]]); v = d > 0
    out = depth_evidence(d, v)
    assert out["edge_fraction"].item() == 0
    assert out["local_variance"].item() == 0


def test_local_variance_is_population_variance_of_valid_neighbor_pairs():
    depth = torch.tensor([[[[1., 3.]]]])
    out = depth_evidence(depth, torch.ones_like(depth, dtype=torch.bool))

    assert out["local_variance"].item() == pytest.approx(1.)


def test_grid_uses_disjoint_pixel_bins_and_requires_integer_shape():
    depth = torch.tensor([[[[0.], [2.], [0.]]]])
    valid = depth > 0
    out = depth_evidence(depth, valid, grid_size=(2, 1))

    support = out["grid_evidence"][0, :, 0, 0]
    assert (support > 0).sum().item() == 1
    with pytest.raises(ValueError):
        depth_evidence(depth, valid, grid_size=(1.5, 1))
