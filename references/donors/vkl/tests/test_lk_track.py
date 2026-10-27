from __future__ import annotations

import pytest
import torch

from vkl.kernels import lk_track


def _textured_image(batch: int = 1) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(12, dtype=torch.float32), torch.arange(12, dtype=torch.float32), indexing="ij")
    image = (x.square() + 0.7 * y.square() + 0.3 * x * y)[None, None]
    return image.expand(batch, -1, -1, -1).contiguous()


def test_lk_track_refines_pose_seeded_translation_with_a_fixed_point_axis():
    reference = _textured_image()
    target = torch.nn.functional.pad(reference[..., :, :-1], (1, 0, 0, 0))
    out = lk_track(
        reference,
        target,
        torch.tensor([[[4, 5]]]),
        torch.tensor([[[4.6, 5.0]]]),
        patch_radius=2,
        iterations=8,
    )

    assert out["valid"].tolist() == [[True]]
    assert torch.allclose(out["pixels_xy"], torch.tensor([[[5.0, 5.0]]]), atol=0.05)
    assert out["residual"].item() < 1e-3


def test_lk_track_masks_invalid_or_out_of_bounds_seed_without_compacting():
    image = _textured_image()
    out = lk_track(
        image,
        image,
        torch.tensor([[[4, 4], [4, 4]]]),
        torch.tensor([[[4.0, 4.0], [0.0, 0.0]]]),
        pixels_valid=torch.tensor([[True, False]]),
        patch_radius=2,
    )

    assert out["pixels_xy"].shape == (1, 2, 2)
    assert out["valid"].tolist() == [[True, False]]
    assert out["pixels_xy"][0, 1].tolist() == [0.0, 0.0]


def test_lk_track_rejects_camera_invalid_or_nonfinite_patch_support():
    reference = _textured_image()
    target = reference.clone()
    reference_valid = torch.ones(1, 1, 12, 12, dtype=torch.bool)
    target_valid = torch.ones_like(reference_valid)
    reference_valid[..., 4, 5] = False
    target_valid[..., 6, 5] = False
    points = torch.tensor([[[5, 5], [8, 8]]])

    out = lk_track(
        reference,
        target,
        points,
        points.to(torch.float32),
        rgb_reference_valid=reference_valid,
        rgb_target_valid=target_valid,
        patch_radius=1,
    )
    assert out["valid"].tolist() == [[False, True]]

    target[0, 0, 8, 9] = float("nan")
    nonfinite = lk_track(reference, target, points[:, 1:], points[:, 1:].to(torch.float32), patch_radius=1)
    assert not nonfinite["valid"].item()
    assert nonfinite["residual"].item() == 0.0


@pytest.mark.parametrize("kwargs", [{"patch_radius": 1.5}, {"iterations": float("nan")}, {"min_determinant": float("inf")}])
def test_lk_track_rejects_invalid_scalar_settings(kwargs):
    image = _textured_image()
    points = torch.tensor([[[4, 4]]], dtype=torch.int64)

    with pytest.raises(ValueError, match="patch_radius, iterations, and min_determinant"):
        lk_track(image, image, points, points.to(torch.float32), **kwargs)


def test_lk_track_rejects_empty_axis_and_cpu_low_precision():
    image = _textured_image()
    with pytest.raises(ValueError, match="at least one"):
        lk_track(image, image, torch.empty(1, 0, 2, dtype=torch.int64), torch.empty(1, 0, 2))
    with pytest.raises(ValueError, match="CPU LK"):
        lk_track(
            image.to(torch.float16),
            image.to(torch.float16),
            torch.tensor([[[4, 4]]]),
            torch.tensor([[[4.0, 4.0]]], dtype=torch.float16),
        )


@pytest.mark.parametrize("batch", [1, 8])
def test_lk_track_preserves_bounded_b1_b8_contract(batch: int):
    image = _textured_image(batch)
    points = torch.tensor([[[4, 4], [6, 6]]], dtype=torch.int64).expand(batch, -1, -1).contiguous()
    out = lk_track(image, image, points, points.to(torch.float32), patch_radius=2)

    assert out["pixels_xy"].shape == (batch, 2, 2)
    assert out["valid"].shape == (batch, 2)
