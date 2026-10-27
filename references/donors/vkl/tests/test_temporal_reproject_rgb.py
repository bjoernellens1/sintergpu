from __future__ import annotations

import pytest
import torch

from vkl.kernels import temporal_reproject_rgb, temporal_rgbd_reference


def test_temporal_reproject_rgb_identity_samples_neighbor_in_target_pixels():
    target_rgb = torch.zeros((1, 3, 4, 5))
    source_rgb = torch.rand((1, 2, 3, 4, 5))
    depth = torch.ones((1, 1, 4, 5))
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4)[None]

    out = temporal_reproject_rgb(target_rgb, depth, source_rgb, intrinsics, pose, intrinsics[:, None].repeat(1, 2, 1, 1), pose[:, None].repeat(1, 2, 1, 1))

    assert torch.allclose(out["aligned_rgb"], source_rgb)
    assert out["aligned_valid"].all()
    assert torch.allclose(out["alignment_confidence"], torch.ones_like(out["alignment_confidence"]))


@pytest.mark.parametrize("batch", [1, 8])
def test_temporal_rgbd_reference_fuses_only_prior_identity_observations(batch: int) -> None:
    rgb = torch.rand((batch, 3, 4, 5))
    depth = torch.full((batch, 1, 4, 5), 2.0)
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()

    out = temporal_rgbd_reference(
        target_depth=depth,
        source_rgb=rgb[:, None].expand(-1, 2, -1, -1, -1).contiguous(),
        source_depth=depth[:, None].expand(-1, 2, -1, -1, -1).contiguous(),
        target_intrinsics=intrinsics,
        target_pose_c2w=pose,
        source_intrinsics=intrinsics[:, None].expand(-1, 2, -1, -1).contiguous(),
        source_pose_c2w=pose[:, None].expand(-1, 2, -1, -1).contiguous(),
    )

    assert torch.allclose(out["rgb"], rgb)
    assert torch.allclose(out["depth"], depth)
    assert out["valid"].all()
    assert torch.equal(out["support_count"], torch.full_like(depth, 2, dtype=torch.int64))
