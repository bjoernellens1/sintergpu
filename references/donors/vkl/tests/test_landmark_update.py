from __future__ import annotations

import torch
import pytest

from vkl.kernels import landmark_update


def test_landmark_update_fuses_matched_world_observation_and_preserves_unmatched_state():
    xyz = torch.tensor([[[1.0, 0.0, 2.0], [4.0, 0.0, 2.0]]])
    support = torch.tensor([[2, 3]], dtype=torch.int64)
    age = torch.tensor([[5, 7]], dtype=torch.int64)
    active = torch.tensor([[True, True]])
    observation = torch.tensor([[[1.3, 0.0, 2.0], [0.0, 0.0, 0.0]]])
    matched = torch.tensor([[True, False]])

    out = landmark_update(xyz, support, age, active, observation, matched)

    assert torch.allclose(out["xyz_world"], torch.tensor([[[1.1, 0.0, 2.0], [4.0, 0.0, 2.0]]]))
    assert out["support_count"].tolist() == [[3, 3]]
    assert out["age"].tolist() == [[6, 8]]
    assert out["matched"].tolist() == [[True, False]]
    assert out["unmatched"].tolist() == [[False, True]]


@pytest.mark.parametrize("batch", [1, 8])
def test_landmark_update_preserves_bounded_batch_contract(batch: int):
    xyz = torch.ones(batch, 3, 3)
    out = landmark_update(
        xyz,
        torch.zeros(batch, 3, dtype=torch.int64),
        torch.zeros(batch, 3, dtype=torch.int64),
        torch.ones(batch, 3, dtype=torch.bool),
        xyz * 2,
        torch.ones(batch, 3, dtype=torch.bool),
    )

    assert out["xyz_world"].shape == (batch, 3, 3)
    assert out["support_count"].shape == (batch, 3)
