from __future__ import annotations

import pytest
import torch

from vkl.kernels import rgbd_evidence


def test_rgbd_evidence_preserves_float64_contract():
    rgb = torch.rand(2, 3, 12, 16, dtype=torch.float64)
    depth = torch.ones(2, 1, 12, 16, dtype=torch.float64)

    out = rgbd_evidence(rgb=rgb, depth=depth, tile_rows=3, tile_cols=4)

    assert out["frame_features"].shape == (2, 9)
    assert out["tile_features"].shape == (2, 9, 3, 4)
    assert all(value.dtype == torch.float64 for value in out.values())
    assert all(torch.isfinite(value).all() for value in out.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rgbd_evidence_cpu_low_precision_uses_safe_compute_but_preserves_output(dtype: torch.dtype):
    rgb = torch.rand(1, 3, 8, 8).to(dtype)
    depth = torch.ones(1, 1, 8, 8, dtype=dtype)

    out = rgbd_evidence(rgb=rgb, depth=depth, tile_rows=2, tile_cols=2)

    assert all(value.dtype == dtype for value in out.values())
    assert out["frame_score"].shape == (1,)


def test_rgbd_evidence_masks_nonfinite_rgb_without_fabricating_texture():
    clean_rgb = torch.full((1, 3, 8, 8), 0.5)
    depth = torch.ones(1, 1, 8, 8)
    clean = rgbd_evidence(rgb=clean_rgb, depth=depth, tile_rows=2, tile_cols=2)

    corrupted_rgb = clean_rgb.clone()
    corrupted_rgb[..., 3, 4] = float("nan")
    corrupted = rgbd_evidence(rgb=corrupted_rgb, depth=depth, tile_rows=2, tile_cols=2)

    assert all(torch.isfinite(value).all() for value in corrupted.values())
    assert corrupted["frame_features"][0, 2] <= clean["frame_features"][0, 2]
    assert corrupted["frame_features"][0, 3] <= clean["frame_features"][0, 3]
    assert corrupted["frame_features"][0, 5] > clean["frame_features"][0, 5]
    assert 0 <= corrupted["frame_score"].item() <= 1


def test_rgbd_evidence_keeps_depth_measurement_occupancy_separate_from_domain():
    rgb = torch.full((1, 3, 4, 4), 0.5)
    depth = torch.full((1, 1, 4, 4), 2.0)
    domain = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    domain[..., -1, :] = False
    measurement = torch.zeros_like(domain)
    measurement[..., :2, :] = True
    original_measurement = measurement.clone()

    out = rgbd_evidence(
        rgb=rgb,
        depth=depth,
        depth_valid=domain,
        depth_measurement_valid=measurement,
        tile_rows=2,
        tile_cols=2,
    )

    assert torch.equal(measurement, original_measurement)
    assert out["frame_depth_domain_fraction"].item() == pytest.approx(0.75)
    assert out["frame_depth_measurement_fraction"].item() == pytest.approx(0.5)
    assert out["frame_depth_coverage"].item() == pytest.approx(0.5)
    assert out["tile_depth_measurement_fraction"].shape == (1, 1, 2, 2)


def test_rgbd_evidence_rejects_mismatched_depth_and_invalid_thresholds():
    rgb = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError):
        rgbd_evidence(rgb=rgb, depth=torch.ones(1, 1, 7, 8))
    with pytest.raises(ValueError):
        rgbd_evidence(rgb=rgb, depth=torch.ones(1, 1, 8, 8, dtype=torch.float64))
    with pytest.raises(ValueError):
        rgbd_evidence(
            rgb=rgb,
            depth_measurement_valid=torch.ones(1, 1, 8, 8, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        rgbd_evidence(rgb=rgb, black_clip=0.9, white_clip=0.1)
    with pytest.raises(ValueError):
        rgbd_evidence(rgb=rgb, depth_min=2.0, depth_max=1.0)
