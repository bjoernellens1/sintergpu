import pytest
import torch

from vkl.kernels import regularize_dense_depth


def test_regularize_dense_depth_keeps_consistent_plane_direct():
    depth = torch.full((1, 1, 5, 5), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    out = regularize_dense_depth(depth, valid, torch.ones_like(depth), torch.zeros(1, 3, 5, 5))
    assert torch.allclose(out["depth"], depth)
    assert out["direct_valid"].all()


def test_regularize_dense_depth_replaces_isolated_depth_speckle():
    depth = torch.full((1, 1, 5, 5), 2.0)
    depth[:, :, 2, 2] = 8.0
    out = regularize_dense_depth(
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        torch.ones_like(depth),
        torch.zeros(1, 3, 5, 5),
    )
    assert torch.isclose(out["depth"][0, 0, 2, 2], torch.tensor(2.0))
    assert not out["direct_valid"][0, 0, 2, 2]


def test_regularize_dense_depth_uses_neighbor_confidence_for_filled_center():
    depth = torch.full((1, 1, 5, 5), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:, :, 2, 2] = False
    confidence = torch.full_like(depth, 0.9)
    confidence[:, :, 2, 2] = 0.0

    out = regularize_dense_depth(depth, valid, confidence, torch.zeros(1, 3, 5, 5))

    assert out["valid"][0, 0, 2, 2]
    assert out["depth"][0, 0, 2, 2] == 2.0
    assert out["confidence"][0, 0, 2, 2] > 0.5


def test_regularize_dense_depth_preserves_rgb_depth_edge():
    depth = torch.ones(1, 1, 5, 6)
    depth[:, :, :, 3:] = 4.0
    rgb = torch.zeros(1, 3, 5, 6)
    rgb[:, :, :, 3:] = 1.0
    out = regularize_dense_depth(
        depth,
        torch.ones_like(depth, dtype=torch.bool),
        torch.ones_like(depth),
        rgb,
    )
    assert torch.isclose(out["depth"][0, 0, 2, 2], torch.tensor(1.0))
    assert torch.isclose(out["depth"][0, 0, 2, 3], torch.tensor(4.0))


def test_regularize_dense_depth_rejects_mismatched_confidence_dtype():
    depth = torch.ones(1, 1, 3, 3)
    with pytest.raises(ValueError):
        regularize_dense_depth(
            depth,
            torch.ones_like(depth, dtype=torch.bool),
            torch.ones_like(depth, dtype=torch.float64),
            torch.zeros(1, 3, 3, 3),
        )
