import pytest
import torch

from vkl.kernels import (
    apply_camera_local_tracking_update,
    dense_track_reduce,
    deterministic_reduce_solve,
    select_tracking_reference,
    sparse_pose_reduce,
)


def _partials(batch: int, value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.eye(6).reshape(1, 1, 6, 6).expand(batch, 2, -1, -1).clone(),
        torch.full((batch, 2, 6), value),
        torch.ones((batch, 2), dtype=torch.bool),
    )


@pytest.mark.parametrize("batch", [1, 8])
def test_reference_selector_uses_f2f_only_when_f2m_evidence_is_unusable(batch: int):
    f2m_H, f2m_b, f2m_valid = _partials(batch, 1.0)
    f2f_H, f2f_b, f2f_valid = _partials(batch, 2.0)
    f2m_usable = torch.ones(batch, dtype=torch.bool)
    f2m_usable[0] = False

    out = select_tracking_reference(
        f2m_partial_H=f2m_H,
        f2m_partial_b=f2m_b,
        f2m_partial_valid=f2m_valid,
        f2m_usable=f2m_usable,
        f2f_partial_H=f2f_H,
        f2f_partial_b=f2f_b,
        f2f_partial_valid=f2f_valid,
    )

    assert out["f2f_selected"].tolist() == [True] + [False] * (batch - 1)
    assert torch.equal(out["partial_b"][0], f2f_b[0])
    if batch > 1:
        assert torch.equal(out["partial_b"][1:], f2m_b[1:])


@pytest.mark.parametrize("batch", [1, 8])
def test_camera_local_tracking_update_corrects_a_sparse_translation_control(batch: int):
    world = torch.tensor(
        [[[-0.2, -0.1, 2.0], [0.3, -0.2, 2.2], [-0.1, 0.4, 1.8], [0.25, 0.3, 2.4]]]
    ).expand(batch, -1, -1).clone()
    candidate = torch.eye(4)[None].expand(batch, -1, -1).clone()
    candidate[:, 2, 3] = 0.2
    reduced = sparse_pose_reduce(world, world, candidate, partial_count=2)
    solved = deterministic_reduce_solve(reduced["partial_H"], reduced["partial_b"], reduced["partial_valid"])

    updated = apply_camera_local_tracking_update(candidate, solved["delta"])

    assert torch.allclose(updated[:, :3, 3], torch.zeros((batch, 3)), atol=2e-4)
    assert torch.allclose(updated[:, :3, :3], torch.eye(3).expand(batch, -1, -1), atol=2e-4)


@pytest.mark.parametrize("batch", [1, 8])
def test_same_dense_reducer_falls_back_to_f2f_and_reanchors_to_recovered_f2m(batch: int):
    height, width = 8, 10
    y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
    depth = (2.0 + 0.15 * x + 0.08 * y + 0.12 * x.square() - 0.06 * y.square()).reshape(1, 1, height, width).expand(batch, -1, -1, -1).contiguous()
    rgb = (0.5 + 0.2 * x + 0.15 * y + 0.1 * x * y).reshape(1, 1, height, width).expand(batch, 3, -1, -1).contiguous()
    intrinsics = torch.tensor([[[10.0, 0.0, 4.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    source_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    source_pose[:, 2, 3] = 0.1
    reference_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    kwargs = {
        "source_depth": depth, "source_intrinsics": intrinsics, "source_pose_c2w": source_pose,
        "reference_intrinsics": intrinsics, "reference_pose_c2w": reference_pose,
        "source_rgb": rgb, "reference_rgb": rgb, "partial_grid": (2, 2),
    }
    f2f = dense_track_reduce(reference_depth=depth, **kwargs)
    degraded_f2m = dense_track_reduce(reference_depth=torch.zeros_like(depth), **kwargs)
    recovered_f2m = dense_track_reduce(reference_depth=depth, **kwargs)
    f2f_solve = deterministic_reduce_solve(f2f["partial_H"], f2f["partial_b"], f2f["partial_valid"])
    degraded_solve = deterministic_reduce_solve(
        degraded_f2m["partial_H"], degraded_f2m["partial_b"], degraded_f2m["partial_valid"]
    )
    recovered_solve = deterministic_reduce_solve(
        recovered_f2m["partial_H"], recovered_f2m["partial_b"], recovered_f2m["partial_valid"]
    )

    fallback = select_tracking_reference(
        f2m_partial_H=degraded_f2m["partial_H"], f2m_partial_b=degraded_f2m["partial_b"],
        f2m_partial_valid=degraded_f2m["partial_valid"], f2m_usable=degraded_solve["valid"],
        f2f_partial_H=f2f["partial_H"], f2f_partial_b=f2f["partial_b"], f2f_partial_valid=f2f["partial_valid"],
    )
    reanchored = select_tracking_reference(
        f2m_partial_H=recovered_f2m["partial_H"], f2m_partial_b=recovered_f2m["partial_b"],
        f2m_partial_valid=recovered_f2m["partial_valid"], f2m_usable=recovered_solve["valid"],
        f2f_partial_H=f2f["partial_H"], f2f_partial_b=f2f["partial_b"], f2f_partial_valid=f2f["partial_valid"],
    )

    assert f2f_solve["valid"].all()
    assert not degraded_solve["valid"].any()
    assert recovered_solve["valid"].all()
    assert fallback["f2f_selected"].all()
    assert torch.equal(fallback["partial_H"], f2f["partial_H"])
    assert not reanchored["f2f_selected"].any()
    assert torch.equal(reanchored["partial_H"], recovered_f2m["partial_H"])


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_f2f_control_reduces_residual_after_one_camera_local_update(batch: int):
    height, width = 16, 20
    y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
    depth = (2.0 + 0.15 * x + 0.08 * y + 0.12 * x.square() - 0.06 * y.square()).reshape(1, 1, height, width).expand(batch, -1, -1, -1).contiguous()
    rgb = (0.5 + 0.2 * x + 0.15 * y + 0.1 * x * y).reshape(1, 1, height, width).expand(batch, 3, -1, -1).contiguous()
    intrinsics = torch.tensor([[[20.0, 0.0, 9.5], [0.0, 20.0, 7.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    reference_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    candidate_pose = reference_pose.clone()
    candidate_pose[:, 2, 3] = 0.1
    kwargs = {
        "source_depth": depth, "source_intrinsics": intrinsics, "reference_depth": depth,
        "reference_intrinsics": intrinsics, "reference_pose_c2w": reference_pose,
        "source_rgb": rgb, "reference_rgb": rgb, "partial_grid": (2, 2),
    }
    before = dense_track_reduce(source_pose_c2w=candidate_pose, **kwargs)
    solved = deterministic_reduce_solve(before["partial_H"], before["partial_b"], before["partial_valid"])
    updated = apply_camera_local_tracking_update(candidate_pose, solved["delta"])
    after = dense_track_reduce(source_pose_c2w=updated, **kwargs)

    assert solved["valid"].all()
    assert (after["geometry_residual_abs_mean"] < before["geometry_residual_abs_mean"]).all()
