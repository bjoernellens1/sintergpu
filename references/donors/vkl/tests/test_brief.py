from __future__ import annotations

import pytest
import torch

from vkl.kernels import brief_describe
from vkl.kernels.reference.brief import _default_pairs


def test_brief_descriptor_is_deterministic_and_tracks_local_intensity_order():
    image = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7) / 48
    pixels = torch.tensor([[[3, 3], [0, 0]]])
    pairs = torch.tensor([[[-1, 0], [1, 0]], [[0, -1], [0, 1]]], dtype=torch.int64)
    out = brief_describe(image, pixels, pairs=pairs)
    assert out["valid"].tolist() == [[True, False]]
    assert out["descriptor"].dtype == torch.bool
    assert out["descriptor"][0, 0].tolist() == [True, True]
    assert not out["descriptor"][0, 1].any()


def test_brief_preserves_upstream_fixed_capacity_validity():
    image = torch.rand(1, 1, 7, 7)
    pixels = torch.tensor([[[3, 3], [3, 3]]])
    pairs = torch.tensor([[[-1, 0], [1, 0]]], dtype=torch.int64)
    out = brief_describe(image, pixels, pixels_valid=torch.tensor([[True, False]]), pairs=pairs)
    assert out["valid"].tolist() == [[True, False]]
    assert not out["descriptor"][0, 1].any()


def test_brief_requires_valid_center_and_every_comparison_sample():
    image = torch.rand(1, 1, 7, 7)
    pixels = torch.tensor([[[3, 3]]])
    pairs = torch.tensor([[[-1, 0], [1, 0]]], dtype=torch.int64)
    rgb_valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)

    rgb_valid[0, 0, 3, 3] = False
    invalid_center = brief_describe(image, pixels, rgb_valid=rgb_valid, pairs=pairs)
    assert not invalid_center["valid"].item()
    assert not invalid_center["descriptor"].any()

    rgb_valid.fill_(True)
    rgb_valid[0, 0, 3, 4] = False
    invalid_sample = brief_describe(image, pixels, rgb_valid=rgb_valid, pairs=pairs)
    assert not invalid_sample["valid"].item()
    assert not invalid_sample["descriptor"].any()

    image[0, 0, 3, 4] = float("nan")
    finite_guard = brief_describe(image, pixels, pairs=pairs)
    assert not finite_guard["valid"].item()
    assert not finite_guard["descriptor"].any()


def test_brief_outside_center_is_invalid_even_if_custom_offsets_land_inside():
    image = torch.rand(1, 1, 7, 7)
    pixels = torch.tensor([[[-1, 3]]])
    pairs = torch.tensor([[[1, 0], [2, 0]]], dtype=torch.int64)
    out = brief_describe(image, pixels, pairs=pairs)
    assert not out["valid"].item()
    assert not out["descriptor"].any()


def test_brief_rejects_malformed_rgb_valid():
    image = torch.rand(1, 1, 7, 7)
    pixels = torch.tensor([[[3, 3]]])
    pairs = torch.tensor([[[-1, 0], [1, 0]]], dtype=torch.int64)
    with pytest.raises(ValueError):
        brief_describe(image, pixels, rgb_valid=torch.ones(1, 7, 7, dtype=torch.bool), pairs=pairs)


def test_brief_default_256_bits_are_distinct_nonself_comparisons():
    pairs = _default_pairs(256, torch.device("cpu"))
    assert torch.unique(pairs.reshape(256, 4), dim=0).shape[0] == 256
    assert not (pairs[:, 0] == pairs[:, 1]).all(-1).any()


def test_brief_rejects_empty_or_self_custom_pattern():
    image = torch.zeros(1, 1, 7, 7)
    pixels = torch.tensor([[[3, 3]]])
    with pytest.raises(ValueError):
        brief_describe(image, pixels, pairs=torch.empty(0, 2, 2, dtype=torch.int64))
    with pytest.raises(ValueError):
        brief_describe(image, pixels, pairs=torch.tensor([[[0, 0], [0, 0]]], dtype=torch.int64))
