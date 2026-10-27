from __future__ import annotations

import torch

from vkl.kernels import canonical_rescale, rgbd_evidence
from vkl.pipeline import NodeSpec, PipelineExecutor, PipelineSpec


def test_canonical_rescale_masks_invalid_rgb_without_border_bleed():
    rgb = torch.ones(1, 3, 4, 4)
    rgb[..., :2] = 1000.0
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., :2] = False
    out = canonical_rescale(rgb=rgb, rgb_valid=valid, target_pixels=4, min_width=1, min_height=1)
    assert out["rgb"].shape[-2:] == (2, 2)
    assert torch.isfinite(out["rgb"]).all()
    assert out["rgb"].max() <= 1.0001
    assert not out["rgb_valid"][..., 0].any()
    assert not out["rgb"][..., 0].any()


def test_canonical_rescale_propagates_depth_validity_and_zeros_invalid_depth():
    rgb = torch.ones(1, 3, 4, 4)
    depth = torch.full((1, 1, 4, 4), 2.0)
    depth[..., :2] = 50.0
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., :2] = False
    out = canonical_rescale(rgb=rgb, depth=depth, depth_valid=valid, target_pixels=4, min_width=1, min_height=1)
    assert not out["depth_valid"][..., 0].any()
    assert not out["depth"][..., 0].any()
    assert torch.equal(out["depth"][..., 1], torch.full_like(out["depth"][..., 1], 2.0))


def test_canonical_rescale_keeps_explicit_depth_domain_when_missing_measurements_are_nan():
    rgb = torch.ones(1, 3, 4, 4)
    depth = torch.full((1, 1, 4, 4), float("nan"))
    domain = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    measurement = torch.zeros_like(domain)

    out = canonical_rescale(
        rgb=rgb,
        depth=depth,
        depth_valid=domain,
        depth_measurement_valid=measurement,
        target_pixels=16,
        min_width=1,
        min_height=1,
    )

    assert out["depth_valid"].all()
    assert not out["depth_measurement_valid"].any()
    assert not out["depth"].any()


def test_canonical_rescale_infers_positive_depth_measurements_without_shrinking_domain():
    rgb = torch.ones(1, 3, 2, 2)
    depth = torch.tensor([[[[0.0, 2.0], [float("nan"), -1.0]]]])

    out = canonical_rescale(
        rgb=rgb,
        depth=depth,
        target_pixels=4,
        min_width=1,
        min_height=1,
    )

    assert out["depth_valid"].all()
    assert out["depth_measurement_valid"].tolist() == [[[[False, True], [False, False]]]]
    assert torch.equal(out["depth"], torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]]))
    evidence = rgbd_evidence(
        rgb=out["rgb"],
        depth=out["depth"],
        depth_valid=out["depth_valid"],
        depth_measurement_valid=out["depth_measurement_valid"],
        tile_rows=1,
        tile_cols=1,
    )
    assert torch.allclose(evidence["frame_depth_domain_fraction"], torch.tensor([1.0]))
    assert torch.allclose(evidence["frame_depth_measurement_fraction"], torch.tensor([0.25]))
    assert torch.allclose(evidence["frame_depth_coverage"], torch.tensor([0.25]))


def test_rgbd_evidence_ignores_values_outside_conditioning_domain():
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    valid[..., :2, :] = False
    first = torch.rand(1, 3, 8, 8)
    second = first.clone()
    first[..., :2, :] = -1000.0
    second[..., :2, :] = 1000.0
    a = rgbd_evidence(rgb=first, rgb_valid=valid, tile_rows=2, tile_cols=2)
    b = rgbd_evidence(rgb=second, rgb_valid=valid, tile_rows=2, tile_cols=2)
    assert torch.allclose(a["frame_features"], b["frame_features"])
    assert torch.allclose(a["tile_features"], b["tile_features"])
    assert torch.allclose(a["frame_score"], b["frame_score"])
    assert torch.allclose(a["tile_score"], b["tile_score"])
    assert torch.allclose(a["frame_rgb_valid_fraction"], torch.tensor([0.75]))


def test_rgbd_evidence_reports_conditioning_and_sensor_depth_coverage_separately():
    rgb = torch.ones(1, 3, 4, 4)
    depth = torch.ones(1, 1, 4, 4)
    depth_valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    depth_valid[..., :2] = False
    depth[..., 2:, 2:] = 0.0
    out = rgbd_evidence(rgb=rgb, depth=depth, depth_valid=depth_valid, tile_rows=2, tile_cols=2)
    assert torch.allclose(out["frame_depth_domain_fraction"], torch.tensor([0.5]))
    assert torch.allclose(out["frame_depth_coverage"], torch.tensor([0.25]))


def test_pipeline_carries_conditioning_validity_from_k0_into_k1():
    spec = PipelineSpec(
        "validity-fastpath",
        (
            NodeSpec("canonical", "canonical_rescale", {"rgb": "input.rgb", "depth": "input.depth", "rgb_valid": "input.rgb_valid", "depth_valid": "input.depth_valid"}, {"target_pixels": 16, "min_width": 1, "min_height": 1}),
            NodeSpec("evidence", "rgbd_evidence", {"rgb": "canonical.rgb", "depth": "canonical.depth", "rgb_valid": "canonical.rgb_valid", "depth_valid": "canonical.depth_valid"}, {"tile_rows": 2, "tile_cols": 2}),
        ),
        ("evidence.frame_score", "evidence.frame_rgb_valid_fraction"),
    )
    rgb = torch.rand(1, 3, 4, 4)
    depth = torch.ones(1, 1, 4, 4)
    rgb_valid = torch.ones(1, 1, 4, 4, dtype=torch.bool); rgb_valid[..., 0, :] = False
    depth_valid = torch.ones_like(rgb_valid)
    out = PipelineExecutor(spec).run({"rgb": rgb, "depth": depth, "rgb_valid": rgb_valid, "depth_valid": depth_valid})
    assert torch.isfinite(out["evidence.frame_score"]).all()
    assert torch.allclose(out["evidence.frame_rgb_valid_fraction"], torch.tensor([0.75]))
