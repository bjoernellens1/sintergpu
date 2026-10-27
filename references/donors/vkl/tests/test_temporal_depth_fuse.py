import pytest
import torch

from vkl.kernels import REGISTRY, temporal_depth_fuse


def _field(values, *, batch=1):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1).expand(batch, -1, -1, -1).clone()


def test_temporal_depth_fuse_weights_consistent_observations_and_reports_support():
    target = _field([2.0, 2.0, 0.0])
    temporal = _field([2.02, 2.11, 3.0])
    valid = torch.tensor([[[[True, True, False]]]])
    temporal_valid = torch.tensor([[[[True, True, True]]]])
    out = temporal_depth_fuse(
        target,
        [temporal],
        valid,
        [temporal_valid],
        torch.full_like(target, 0.8),
        [torch.full_like(target, 0.2)],
    )
    assert out["depth"][0, 0, 0, 0].item() == pytest.approx(2.004)
    assert out["depth"][0, 0, 0, 1].item() == pytest.approx(0.0)
    assert out["valid"][0, 0, 0, 1].item() is False
    assert out["valid"][0, 0, 0, 2].item() is False
    assert out["support_count"].tolist() == [[[[2, 1, 1]]]]
    assert out["depth_variance"][0, 0, 0, 0].item() == pytest.approx(0.000064, abs=1e-8)
    assert out["conflict_owner_source"].tolist() == [[[[0, -1, -1]]]]
    assert out["dominant_contributor"].tolist() == [[[[0, -1, -1]]]]
    assert out["temporal_weight_fraction"][0, 0, 0, 0].item() == pytest.approx(0.2)
    assert out["source_disagreement"][0, 0, 0, 0].item() == pytest.approx(0.02)


@pytest.mark.parametrize("batch", [1, 8])
def test_temporal_depth_fuse_rejects_target_only_evidence_without_explicit_trust(batch):
    target = _field([2.0], batch=batch)
    temporal = _field([2.0], batch=batch)
    valid = torch.ones_like(target, dtype=torch.bool)
    invalid = torch.zeros_like(target, dtype=torch.bool)

    out = temporal_depth_fuse(
        target,
        [temporal],
        valid,
        [invalid],
        torch.ones_like(target),
        [torch.ones_like(target)],
    )

    assert not out["valid"].any()
    assert torch.equal(out["support_count"], torch.ones_like(out["support_count"]))
    assert torch.equal(out["temporal_support_count"], torch.zeros_like(out["temporal_support_count"]))


def test_temporal_depth_fuse_uses_trusted_geometry_on_conflict_and_evidence_masks():
    target = _field([5.0, 2.0])
    temporal = _field([2.0, 2.01])
    valid = torch.ones_like(target, dtype=torch.bool)
    out = temporal_depth_fuse(
        target,
        [temporal],
        valid,
        [valid],
        torch.full_like(target, 0.9),
        [torch.full_like(target, 0.1)],
        visible=[valid],
        consistent=[valid],
        trusted=[valid],
    )
    assert out["depth"][0, 0, 0, 0].item() == pytest.approx(2.0)
    assert out["support_count"][0, 0, 0, 0].item() == 1
    assert out["depth"][0, 0, 0, 1].item() == pytest.approx(2.001)
    assert REGISTRY["temporal_depth_fuse"] is temporal_depth_fuse


def test_temporal_depth_fuse_uses_strongest_untrusted_agreement_cluster():
    target = _field([5.0])
    temporal_a = _field([2.0])
    temporal_b = _field([2.02])
    valid = torch.ones_like(target, dtype=torch.bool)
    out = temporal_depth_fuse(
        target,
        [temporal_a, temporal_b],
        valid,
        [valid, valid],
        torch.full_like(target, 0.8),
        [torch.full_like(target, 0.6), torch.full_like(target, 0.6)],
    )
    assert out["depth"].item() == pytest.approx(2.01)
    assert out["support_count"].item() == 2


def test_temporal_depth_fuse_invalid_pixel_has_no_source_provenance():
    depth = _field([1.0])
    invalid = torch.zeros_like(depth, dtype=torch.bool)
    out = temporal_depth_fuse(
        depth,
        [depth],
        invalid,
        [invalid],
        torch.ones_like(depth),
        [torch.ones_like(depth)],
    )
    assert not out["valid"].item()
    assert out["winning_source"].item() == -1
    assert out["conflict_owner_source"].item() == -1
    assert out["dominant_contributor"].item() == -1
    assert out["source_disagreement"].item() == 0


def test_temporal_depth_fuse_rejects_mismatched_inputs():
    depth = _field([1.0])
    valid = torch.ones_like(depth, dtype=torch.bool)
    with pytest.raises(ValueError):
        temporal_depth_fuse(depth, [depth], valid, [], depth, [depth])
