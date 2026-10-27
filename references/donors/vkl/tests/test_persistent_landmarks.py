from __future__ import annotations

import pytest
import torch

from vkl.landmarks import PersistentLandmarkStore


def test_store_allocates_stable_ids_updates_matched_tracks_and_retires_lost_tracks():
    store = PersistentLandmarkStore(batch_size=1, capacity=3, descriptor_bits=4)
    observations = torch.tensor([[[1.0, 0.0, 2.0], [4.0, 0.0, 2.0]]])
    descriptors = torch.tensor([[[True, False, False, True], [False, True, True, False]]])

    created = store.create(observations, valid=torch.tensor([[True, True]]), descriptors=descriptors)

    assert created["slot"].tolist() == [[0, 1]]
    assert created["id"].tolist() == [[0, 1]]
    assert created["valid"].tolist() == [[True, True]]
    assert store.active.tolist() == [[True, True, False]]

    matched = torch.tensor([[True, False, False]])
    updated = store.update(
        torch.tensor([[[1.3, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        matched,
        descriptors=torch.tensor([[[False, True, False, True], [False, False, False, False], [False, False, False, False]]]),
    )

    assert torch.allclose(store.xyz_world[0, 0], torch.tensor([1.15, 0.0, 2.0]))
    assert store.support_count.tolist() == [[2, 1, 0]]
    assert store.missed_count.tolist() == [[0, 1, 0]]
    assert updated["matched"].tolist() == [[True, False, False]]
    assert store.descriptor[0, 0].tolist() == [False, True, False, True]

    # max_missed is the number of consecutive misses that may be retained.
    assert not store.retire(max_missed=1).any()
    store.update(torch.zeros(1, 3, 3), torch.tensor([[True, False, False]]))
    retired = store.retire(max_missed=1)

    assert retired.tolist() == [[False, True, False]]
    assert store.active.tolist() == [[True, False, False]]
    assert store.landmark_id.tolist() == [[0, -1, -1]]

    replacement = store.create(
        torch.tensor([[[8.0, 0.0, 2.0]]]),
        valid=torch.tensor([[True]]),
        descriptors=torch.tensor([[[True, True, False, False]]]),
    )
    assert replacement["slot"].tolist() == [[1]]
    assert replacement["id"].tolist() == [[2]]


@pytest.mark.parametrize("batch", [1, 8])
def test_store_has_fixed_capacity_bounded_batch_state(batch: int):
    store = PersistentLandmarkStore(batch_size=batch, capacity=2, descriptor_bits=8)
    created = store.create(
        torch.ones(batch, 3, 3),
        valid=torch.tensor([[True, True, True]] * batch),
        descriptors=torch.ones(batch, 3, 8, dtype=torch.bool),
    )

    assert store.xyz_world.shape == (batch, 2, 3)
    assert store.landmark_id.shape == (batch, 2)
    assert created["valid"].tolist() == [[True, True, False]] * batch
    assert created["dropped"].tolist() == [[False, False, True]] * batch


def test_store_preserves_descriptor_validity_independently_from_geometry():
    store = PersistentLandmarkStore(batch_size=1, capacity=2, descriptor_bits=4)
    descriptors = torch.tensor([[[True, True, False, False], [False, True, False, True]]])
    store.create(
        torch.ones(1, 2, 3),
        valid=torch.tensor([[True, True]]),
        descriptors=descriptors,
        descriptor_valid=torch.tensor([[True, False]]),
    )
    assert store.descriptor_valid.tolist() == [[True, False]]
    assert not store.descriptor[0, 1].any()

    old = store.descriptor.clone()
    store.update(
        torch.ones(1, 2, 3) * 2,
        torch.tensor([[True, True]]),
        descriptors=torch.logical_not(descriptors),
        descriptor_valid=torch.tensor([[False, True]]),
    )
    assert torch.equal(store.descriptor[0, 0], old[0, 0])
    assert torch.equal(store.descriptor[0, 1], torch.logical_not(descriptors)[0, 1])
    assert store.descriptor_valid.tolist() == [[True, True]]

    with pytest.raises(ValueError, match="descriptor_valid requires descriptors"):
        store.update(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), descriptor_valid=torch.ones(1, 2, dtype=torch.bool))


def test_zero_miss_budget_retires_only_after_first_miss():
    store = PersistentLandmarkStore(batch_size=1, capacity=1, descriptor_bits=4)
    store.create(torch.ones(1, 1, 3), valid=torch.ones(1, 1, dtype=torch.bool))
    assert not store.retire(max_missed=0).any()
    store.update(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.bool))
    assert store.retire(max_missed=0).item()


def test_store_masks_inactive_matches_on_fast_path_and_keeps_strict_validation_available():
    store = PersistentLandmarkStore(batch_size=1, capacity=2, descriptor_bits=4)
    store.create(torch.ones(1, 1, 3), valid=torch.ones(1, 1, dtype=torch.bool))

    with pytest.raises(ValueError, match="descriptor"):
        store.create(
            torch.ones(1, 1, 3),
            valid=torch.ones(1, 1, dtype=torch.bool),
            descriptors=torch.ones(1, 1, 3, dtype=torch.bool),
        )
    out = store.update(torch.ones(1, 2, 3), torch.tensor([[True, True]]))
    assert out["matched"].tolist() == [[True, False]]

    with pytest.raises(ValueError, match="active"):
        store.update(torch.ones(1, 2, 3), torch.tensor([[True, True]]), validate=True)

    store.xyz_world[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        store.update(torch.ones(1, 2, 3), torch.tensor([[True, False]]), validate=True)
