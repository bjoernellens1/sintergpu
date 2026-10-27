import pytest
import torch

from vkl.kernels import REGISTRY, tracking_prepare


@pytest.mark.parametrize("batch", [1, 8])
def test_tracking_prepare_preserves_bounded_rgbd_pose_levels(batch: int):
    height, width = 8, 12
    rgb = torch.rand(batch, 3, height, width)
    depth = torch.ones(batch, 1, height, width)
    confidence = torch.full_like(depth, 0.75)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 5.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]]
    ).expand(batch, -1, -1).clone()
    pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    pose[:, 0, 3] = 2.0

    out = tracking_prepare(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        pose_c2w=pose,
        depth_confidence=confidence,
        target_pixels=height * width,
        min_width=1,
        min_height=1,
    )

    assert tuple(out["levels"]) == ("/4", "/2", "/1")
    assert out["levels"]["/4"]["rgb"].shape == (batch, 3, 2, 3)
    assert out["levels"]["/2"]["rgb"].shape == (batch, 3, 4, 6)
    assert out["levels"]["/1"]["rgb"].shape == (batch, 3, 8, 12)
    assert torch.allclose(out["levels"]["/2"]["intrinsics"][:, 0, 0], torch.full((batch,), 4.0))
    assert torch.allclose(out["levels"]["/2"]["intrinsics"][:, 0, 2], torch.full((batch,), 2.5))
    assert out["levels"]["/1"]["xyz_world"].shape == (batch, 3, height, width)
    assert torch.allclose(out["levels"]["/1"]["xyz_world"][:, 0, 0, 0], torch.full((batch,), 1.3125))
    assert torch.equal(out["levels"]["/1"]["depth_confidence"], confidence)
    assert out["evidence"]["rgbd"]["frame_features"].shape[0] == batch
    assert out["evidence"]["depth"]["valid_fraction"].shape == (batch,)
    assert REGISTRY["tracking_prepare"] is tracking_prepare


@pytest.mark.parametrize("batch", [1, 8])
def test_tracking_prepare_rgb_only_does_not_fabricate_depth(batch: int):
    out = tracking_prepare(
        rgb=torch.rand(batch, 3, 8, 12),
        target_pixels=96,
        min_width=1,
        min_height=1,
    )

    assert tuple(out["levels"]) == ("/4", "/2", "/1")
    assert all("depth" not in level for level in out["levels"].values())
    assert set(out["evidence"]) == {"rgbd"}


def test_tracking_prepare_rejects_depth_without_intrinsics():
    with pytest.raises(ValueError, match="intrinsics"):
        tracking_prepare(rgb=torch.rand(1, 3, 8, 12), depth=torch.ones(1, 1, 8, 12))


def test_tracking_prepare_rejects_unknown_or_duplicate_scales():
    rgb = torch.rand(1, 3, 8, 12)
    with pytest.raises(ValueError, match="scales"):
        tracking_prepare(rgb=rgb, scales=("/1", "/1"))
    with pytest.raises(ValueError, match="scales"):
        tracking_prepare(rgb=rgb, scales=("/16",))
