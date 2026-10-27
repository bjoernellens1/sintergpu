import pytest
import torch

from vkl.bench.tracking_transaction import run_transaction


def _inputs(batch: int):
    height, width = 12, 16
    x = torch.linspace(0.1, 0.9, width).reshape(1, 1, 1, width)
    y = torch.linspace(0.0, 0.2, height).reshape(1, 1, height, 1)
    rgb = (x + y).expand(batch, 3, height, width).contiguous()
    depth = torch.full((batch, 1, height, width), 2.0)
    intrinsics = torch.tensor([[[16.0, 0.0, 7.5], [0.0, 16.0, 5.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    reference_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    source_pose = reference_pose.clone()
    candidate_pose = source_pose.clone()
    candidate_pose[:, 2, 3] = 0.05
    return rgb, depth, intrinsics, reference_pose, source_pose, candidate_pose


@pytest.mark.parametrize("batch", [1, 8])
def test_transaction_composes_real_dense_and_sparse_evidence(batch: int):
    rgb, depth, intrinsics, reference_pose, _, candidate_pose = _inputs(batch)

    out = run_transaction(
        source_rgb=rgb,
        source_depth=depth,
        source_intrinsics=intrinsics,
        candidate_pose_c2w=candidate_pose,
        reference_rgb=rgb,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        points=16,
        dense_partial_grid=(2, 2),
        sparse_partial_count=2,
    )

    assert out["delta"].shape == (batch, 6)
    assert out["dense_partial_count"].shape == (batch,)
    assert out["sparse_partial_count"].shape == (batch,)
    assert (out["sparse_inlier_count"] > 0).all()
    assert (out["sparse_partial_count"] > 0).all()
    assert torch.allclose(out["H"], out["dense_H"] + out["sparse_H"])
    assert torch.allclose(out["b"], out["dense_b"] + out["sparse_b"])


@pytest.mark.parametrize("batch", [1, 8])
def test_transaction_rejects_zero_confidence_pseudodepth_from_dense_and_sparse_terms(batch: int):
    rgb, depth, intrinsics, reference_pose, _, candidate_pose = _inputs(batch)

    out = run_transaction(
        source_rgb=rgb,
        source_depth=depth,
        source_confidence=torch.zeros_like(depth),
        source_intrinsics=intrinsics,
        candidate_pose_c2w=candidate_pose,
        reference_rgb=rgb,
        reference_depth=depth,
        reference_confidence=torch.ones_like(depth),
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        points=16,
        dense_partial_grid=(2, 2),
        sparse_partial_count=2,
    )

    assert not out["valid"].any()
    assert torch.equal(out["dense_H"], torch.zeros_like(out["dense_H"]))
    assert torch.equal(out["sparse_H"], torch.zeros_like(out["sparse_H"]))


@pytest.mark.parametrize("batch", [1, 8])
def test_transaction_rejects_relative_depth_from_metric_geometric_solve(batch: int):
    rgb, depth, intrinsics, reference_pose, _, candidate_pose = _inputs(batch)

    with pytest.raises(ValueError, match="metric"):
        run_transaction(
            source_rgb=rgb,
            source_depth=depth,
            source_intrinsics=intrinsics,
            source_metric_status="RELATIVE",
            candidate_pose_c2w=candidate_pose,
            reference_rgb=rgb,
            reference_depth=depth,
            reference_intrinsics=intrinsics,
            reference_pose_c2w=reference_pose,
            points=16,
        )
