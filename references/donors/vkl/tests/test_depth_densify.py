import pytest
import torch

from vkl.kernels import depth_densify, depth_downsample, depth_upsample


@pytest.mark.parametrize("batch", [1, 8])
def test_depth_downsample_uses_validity_aware_pyramid(batch: int):
    depth = torch.tensor([[[[4.0, 2.0], [0.0, 0.0]]]]).repeat(batch, 1, 1, 1)
    valid = depth > 0

    out = depth_downsample(depth, valid, torch.ones_like(depth), scale=2)

    assert out["depth"].shape == (batch, 1, 1, 1)
    assert torch.equal(out["valid"], torch.ones_like(out["valid"]))
    assert torch.allclose(out["depth"], torch.full_like(out["depth"], 2.0))
    assert out["direct_valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_depth_upsample_marks_expanded_values_propagated(batch: int):
    depth = torch.tensor([[[[2.0]]]]).repeat(batch, 1, 1, 1)
    valid = torch.ones_like(depth, dtype=torch.bool)

    out = depth_upsample(depth, valid, torch.ones_like(depth), size=(3, 5))

    assert out["depth"].shape == (batch, 1, 3, 5)
    assert torch.allclose(out["depth"], torch.full_like(out["depth"], 2.0))
    assert out["valid"].all()
    assert not out["direct_valid"].any()
    assert out["propagated_valid"].all()
    assert torch.equal(out["provenance"], torch.full_like(out["provenance"], 2))


@pytest.mark.parametrize("batch", [1, 8])
def test_depth_densify_fills_rgb_consistent_hole_without_claiming_direct(batch: int):
    depth = torch.full((batch, 1, 5, 5), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:, :, 2, 2] = False
    depth[:, :, 2, 2] = 0

    out = depth_densify(depth, valid, torch.ones_like(depth), torch.zeros(batch, 3, 5, 5))

    assert torch.allclose(out["depth"], torch.full_like(depth, 2.0))
    assert out["valid"].all()
    assert not out["direct_valid"][:, :, 2, 2].any()
    assert out["propagated_valid"][:, :, 2, 2].all()
    assert (out["provenance"][:, :, 2, 2] == 2).all()


def test_depth_densify_does_not_cross_rgb_or_depth_discontinuity():
    depth = torch.ones(1, 1, 5, 6)
    valid = torch.ones_like(depth, dtype=torch.bool)
    depth[:, :, :, 3:] = 4.0
    rgb = torch.zeros(1, 3, 5, 6)
    rgb[:, :, :, 3:] = 1.0
    valid[:, :, 1:4, 2:5] = False
    depth[~valid] = 0

    out = depth_densify(depth, valid, torch.ones_like(depth), rgb, max_rgb_difference=0.1)

    assert not out["valid"][:, :, 2, 3].item()
    assert out["provenance"][:, :, 2, 3].item() == 0


def test_depth_densify_multi_step_expands_only_from_supported_neighbours():
    depth = torch.zeros(1, 1, 7, 7)
    valid = torch.zeros_like(depth, dtype=torch.bool)
    depth[:, :, 3, 3] = 2.0
    valid[:, :, 3, 3] = True

    out = depth_densify(
        depth,
        valid,
        torch.ones_like(depth),
        torch.zeros(1, 3, 7, 7),
        min_support=1,
        steps=2,
    )

    assert out["valid"][:, :, 1:6, 1:6].all()
    assert not out["valid"][:, :, 0, :].any()


def test_downsample_upsample_then_densify_retains_propagated_provenance():
    depth = torch.full((1, 1, 4, 4), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)

    coarse = depth_downsample(depth, valid, torch.ones_like(depth), scale=2)
    lifted = depth_upsample(coarse["depth"], coarse["valid"], coarse["confidence"], size=(4, 4))
    out = depth_densify(
        lifted["depth"],
        lifted["valid"],
        lifted["confidence"],
        torch.zeros(1, 3, 4, 4),
        direct_valid=lifted["direct_valid"],
    )

    assert out["valid"].all()
    assert not out["direct_valid"].any()
    assert out["propagated_valid"].all()
