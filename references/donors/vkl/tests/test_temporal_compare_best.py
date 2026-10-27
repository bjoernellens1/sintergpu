from __future__ import annotations

import torch
import pytest

from vkl.kernels import temporal_best_observation, temporal_frame_compare


def _gradient_image(height: int = 8, width: int = 10) -> torch.Tensor:
    return (torch.arange(height)[:, None] + torch.arange(width)[None, :]).remainder(2).float()


def test_temporal_frame_compare_reports_sharp_neighbor_and_rgb_evidence():
    target_gray = _gradient_image()
    sharp = target_gray.repeat(3, 1, 1)[None]
    blurred = torch.nn.functional.avg_pool2d(sharp, 5, stride=1, padding=2)
    target = blurred
    aligned = torch.stack((sharp, sharp), dim=1)
    out = temporal_frame_compare(
        target,
        aligned,
        alignment_confidence=torch.ones((1, 2, 1, 8, 10)),
    )

    assert out["pairwise_quality"].shape == (1, 2)
    assert out["pairwise_equivalence"].shape == (1, 2)
    assert out["temporal_tile_features"].ndim == 4
    assert out["temporal_frame_features"].shape[0] == 1
    assert out["restoration_support_map"].shape == (1, 2, 1, 8, 10)
    assert out["relative_sharpness"][0, 0] > 0.9
    assert torch.isfinite(out["temporal_rgb_consistency"][0, 0])
    assert out["motion_blur_evidence"][0, 0] > 0


def test_blurred_static_target_keeps_sharp_neighbor_equivalent_for_replacement():
    sharp = _gradient_image(24, 28).repeat(3, 1, 1)[None]
    blurred = torch.nn.functional.avg_pool2d(sharp, 5, stride=1, padding=2)
    out = temporal_frame_compare(blurred, sharp[:, None])
    selected = temporal_best_observation(
        out["pairwise_equivalence"], out["target_quality"], out["pairwise_quality"], pairwise_quality=out["pairwise_quality"]
    )

    assert out["pairwise_equivalence"][0, 0]
    assert selected["best_index"].item() == 1


def test_temporal_frame_compare_depth_consistency_is_optional_and_invalid_is_excluded():
    target = torch.ones((1, 3, 4, 4))
    aligned = target[:, None].expand(-1, 2, -1, -1, -1).clone()
    target_depth = torch.ones((1, 1, 4, 4))
    aligned_depth = torch.stack((target_depth, target_depth * 2), dim=1)
    confidence = torch.ones((1, 2, 1, 4, 4))
    confidence[:, 0, :, 0, 0] = 0
    out = temporal_frame_compare(
        target,
        aligned,
        alignment_confidence=confidence,
        target_depth=target_depth,
        aligned_depth=aligned_depth,
    )

    assert out["temporal_depth_consistency"][0, 0] > out["temporal_depth_consistency"][0, 1]
    assert out["pairwise_equivalence"][0, 0]
    assert out["pairwise_equivalence"][0, 1] == 0


def test_temporal_best_observation_selects_equivalent_neighbor_and_falls_back_to_target():
    equivalent = torch.tensor([[True, True], [False, False]])
    target_quality = torch.tensor([0.4, 0.8])
    neighbor_quality = torch.tensor([[0.7, 0.3], [0.1, 0.2]])
    pair_quality = torch.tensor([[0.7, 0.3], [0.1, 0.2]])
    out = temporal_best_observation(equivalent, target_quality, neighbor_quality, pairwise_quality=pair_quality)

    assert torch.equal(out["best_index"], torch.tensor([1, 0]))
    assert torch.allclose(out["best_quality"], torch.tensor([0.7, 0.8]))
    assert out["replaceability"][0] > 0
    assert out["replaceability"][1] == 0
    assert out["replacement_confidence"][0] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_temporal_frame_compare_compiles_without_boundary_validation():
    target = torch.rand(1, 3, 16, 20, device="cuda")
    aligned = target[:, None].repeat(1, 2, 1, 1, 1)
    confidence = torch.ones(1, 2, 1, 16, 20, device="cuda")

    eager = temporal_frame_compare(target, aligned, alignment_confidence=confidence, validate=False)
    compiled = torch.compile(temporal_frame_compare, fullgraph=True)(
        target, aligned, alignment_confidence=confidence, validate=False
    )

    assert torch.allclose(compiled["pairwise_quality"], eager["pairwise_quality"])
