from __future__ import annotations

import pytest
import torch

from vkl.kernels import binary_local_match


def test_binary_local_match_uses_pose_gate_before_hamming_distance():
    landmark = torch.tensor([[[True, False, True], [False, False, False]]])
    candidates = torch.tensor([[[False, False, False], [True, False, True], [True, True, True]]])
    projected = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]])
    pixels = torch.tensor([[[1, 1], [10, 10], [20, 20]]])
    out = binary_local_match(landmark, candidates, projected, pixels, radius_px=2.0, max_hamming_distance=1)
    assert out["matched_index"].tolist() == [[1, 0]]
    assert out["distance"].tolist() == [[0, 0]]
    assert out["valid"].tolist() == [[True, True]]


def test_binary_local_match_masks_invalid_landmarks_and_candidates():
    landmark = torch.zeros((1, 2, 4), dtype=torch.bool)
    candidates = torch.zeros((1, 2, 4), dtype=torch.bool)
    projected = torch.tensor([[[5.0, 5.0], [5.0, 5.0]]])
    pixels = torch.tensor([[[5, 5], [5, 5]]])
    out = binary_local_match(
        landmark, candidates, projected, pixels,
        radius_px=1.0, max_hamming_distance=0,
        landmark_valid=torch.tensor([[True, False]]),
        candidate_valid=torch.tensor([[False, True]]),
    )
    assert out["matched_index"].tolist() == [[1, -1]]
    assert out["valid"].tolist() == [[True, False]]


def test_binary_local_match_rejects_descriptor_poor_local_candidate():
    out = binary_local_match(
        torch.ones((1, 1, 8), dtype=torch.bool),
        torch.zeros((1, 1, 8), dtype=torch.bool),
        torch.tensor([[[5.0, 5.0]]]),
        torch.tensor([[[5, 5]]]),
        radius_px=1.0,
        max_hamming_distance=2,
    )
    assert not out["valid"].item()
    assert out["matched_index"].item() == -1


def test_binary_local_match_is_mutual_one_to_one_by_default():
    landmark = torch.zeros((1, 2, 4), dtype=torch.bool)
    candidates = torch.zeros((1, 1, 4), dtype=torch.bool)
    projected = torch.tensor([[[5.0, 5.0], [5.0, 5.0]]])
    pixels = torch.tensor([[[5, 5]]])
    mutual = binary_local_match(landmark, candidates, projected, pixels, radius_px=1.0, max_hamming_distance=0)
    non_mutual = binary_local_match(landmark, candidates, projected, pixels, radius_px=1.0, max_hamming_distance=0, mutual=False)
    assert mutual["valid"].tolist() == [[True, False]]
    assert non_mutual["valid"].tolist() == [[True, True]]


def test_binary_local_match_candidate_chunking_is_exact():
    generator = torch.Generator().manual_seed(7)
    landmark = torch.rand((2, 5, 16), generator=generator) > 0.5
    candidates = torch.rand((2, 9, 16), generator=generator) > 0.5
    projected = torch.rand((2, 5, 2), generator=generator) * 10
    pixels = torch.randint(0, 10, (2, 9, 2), generator=generator)
    landmark_valid = torch.rand((2, 5), generator=generator) > 0.2
    candidate_valid = torch.rand((2, 9), generator=generator) > 0.2
    kwargs = dict(
        radius_px=4.0,
        max_hamming_distance=8,
        landmark_valid=landmark_valid,
        candidate_valid=candidate_valid,
    )

    full = binary_local_match(landmark, candidates, projected, pixels, **kwargs)
    chunked = binary_local_match(
        landmark, candidates, projected, pixels, candidate_chunk_size=3, **kwargs
    )

    for key in full:
        assert torch.equal(full[key], chunked[key])


def test_binary_local_match_rejects_empty_descriptor_candidate_or_landmark_axes():
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 1, 0), dtype=torch.bool), torch.zeros((1, 1, 0), dtype=torch.bool),
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.int64),
            radius_px=1.0, max_hamming_distance=0,
        )
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 1, 4), dtype=torch.bool), torch.zeros((1, 0, 4), dtype=torch.bool),
            torch.zeros(1, 1, 2), torch.zeros(1, 0, 2, dtype=torch.int64),
            radius_px=1.0, max_hamming_distance=0,
        )
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 0, 4), dtype=torch.bool), torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros(1, 0, 2), torch.zeros(1, 1, 2, dtype=torch.int64),
            radius_px=1.0, max_hamming_distance=0,
        )


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), 0.0, -1.0, True])
def test_binary_local_match_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.int64),
            radius_px=radius,
            max_hamming_distance=0,
        )


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_binary_local_match_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.int64),
            radius_px=1.0,
            max_hamming_distance=0,
            candidate_chunk_size=chunk_size,
        )


def test_binary_local_match_requires_boolean_mutual_flag():
    with pytest.raises(ValueError):
        binary_local_match(
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros((1, 1, 4), dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.int64),
            radius_px=1.0,
            max_hamming_distance=0,
            mutual=1,
        )
