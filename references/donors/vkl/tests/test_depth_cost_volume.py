from __future__ import annotations

import torch

from vkl.kernels import aggregate_depth_cost_volume


def test_edge_aware_cost_aggregation_removes_weak_isolated_label_flips():
    """Weak one-pixel wins do not survive smooth-region cost aggregation."""
    cost = torch.tensor([[[[0.10, 0.10, 0.105, 0.10, 0.10]], [[0.11, 0.11, 0.10, 0.11, 0.11]]]])
    rgb = torch.zeros((1, 3, 1, 5))

    aggregated = aggregate_depth_cost_volume(cost, rgb, smoothness_penalty=0.05)

    assert torch.equal(aggregated.argmin(1), torch.zeros((1, 1, 5), dtype=torch.long))


def test_edge_aware_cost_aggregation_preserves_label_change_at_rgb_edge():
    cost = torch.tensor([[[[0.10, 0.10, 0.11, 0.11]], [[0.11, 0.11, 0.10, 0.10]]]])
    rgb = torch.zeros((1, 3, 1, 4))
    rgb[:, :, :, 2:] = 1

    aggregated = aggregate_depth_cost_volume(cost, rgb, smoothness_penalty=0.05)

    assert torch.equal(aggregated.argmin(1), torch.tensor([[[0, 0, 1, 1]]]))


def test_cost_aggregation_can_operate_on_a_coarser_grid():
    cost = torch.rand((1, 3, 8, 12))
    rgb = torch.rand((1, 3, 8, 12))

    aggregated = aggregate_depth_cost_volume(cost, rgb, aggregation_scale=4)

    assert aggregated.shape == cost.shape
    assert torch.isfinite(aggregated).all()
