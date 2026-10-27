import pytest
import torch

from vkl.bench.tracking_reference import iterate_reference_tracking
from vkl.kernels import dense_track_reduce


def _inputs(batch: int):
    height, width = 16, 20
    y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
    depth = (2.0 + 0.15 * x + 0.08 * y + 0.12 * x.square() - 0.06 * y.square()).reshape(1, 1, height, width).expand(batch, -1, -1, -1).contiguous()
    rgb = (0.5 + 0.2 * x + 0.15 * y + 0.1 * x * y).reshape(1, 1, height, width).expand(batch, 3, -1, -1).contiguous()
    intrinsics = torch.tensor([[[20.0, 0.0, 9.5], [0.0, 20.0, 7.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    candidate = pose.clone()
    candidate[:, 2, 3] = 0.1
    return rgb, depth, intrinsics, pose, candidate


@pytest.mark.parametrize("batch", [1, 8])
def test_reference_iterations_prefer_healthy_f2m_and_reduce_pose_error(batch: int):
    rgb, depth, intrinsics, pose, candidate = _inputs(batch)

    out = iterate_reference_tracking(
        source_rgb=rgb, source_depth=depth, source_intrinsics=intrinsics,
        f2m_rgb=rgb, f2m_depth=depth, f2m_intrinsics=intrinsics, f2m_pose_c2w=pose,
        f2f_rgb=rgb, f2f_depth=depth, f2f_intrinsics=intrinsics, f2f_pose_c2w=pose,
        candidate_pose_c2w=candidate, iterations=1, partial_grid=(2, 2),
    )

    assert out["selected_valid"].all()
    assert not out["f2f_selected"].any()
    after = dense_track_reduce(
        source_depth=depth, source_intrinsics=intrinsics, source_pose_c2w=out["pose_c2w"],
        reference_depth=depth, reference_intrinsics=intrinsics, reference_pose_c2w=pose,
        source_rgb=rgb, reference_rgb=rgb, partial_grid=(2, 2),
    )
    assert (after["geometry_residual_abs_mean"] < 0.1).all()


@pytest.mark.parametrize("batch", [1, 8])
def test_reference_iterations_fall_back_to_f2f_when_f2m_depth_is_degraded(batch: int):
    rgb, depth, intrinsics, pose, candidate = _inputs(batch)

    out = iterate_reference_tracking(
        source_rgb=rgb, source_depth=depth, source_intrinsics=intrinsics,
        f2m_rgb=rgb, f2m_depth=torch.zeros_like(depth), f2m_intrinsics=intrinsics, f2m_pose_c2w=pose,
        f2f_rgb=rgb, f2f_depth=depth, f2f_intrinsics=intrinsics, f2f_pose_c2w=pose,
        candidate_pose_c2w=candidate, iterations=1, partial_grid=(2, 2),
    )

    assert not out["f2m_valid"].any()
    assert out["f2f_selected"].all()
    assert out["selected_valid"].all()
    after = dense_track_reduce(
        source_depth=depth, source_intrinsics=intrinsics, source_pose_c2w=out["pose_c2w"],
        reference_depth=depth, reference_intrinsics=intrinsics, reference_pose_c2w=pose,
        source_rgb=rgb, reference_rgb=rgb, partial_grid=(2, 2),
    )
    assert (after["geometry_residual_abs_mean"] < 0.1).all()
