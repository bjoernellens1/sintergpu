import pytest
import torch

from vkl.kernels import rigid_align_3d


@pytest.mark.parametrize("batch", [1, 8])
def test_rigid_align_3d_recovers_batched_camera_to_world_transform(batch: int) -> None:
    source = torch.tensor(
        [[[-0.4, -0.2, 1.2], [0.3, -0.1, 1.8], [0.1, 0.5, 1.4], [-0.2, 0.4, 2.1], [0.6, 0.2, 1.6], [-0.5, 0.3, 1.7]]]
    ).expand(batch, -1, -1).clone()
    angle = torch.tensor(0.25)
    rotation = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0], [-torch.sin(angle), 0.0, torch.cos(angle)]]
    ).expand(batch, -1, -1).clone()
    translation = torch.tensor([[0.4, -0.2, 0.7]]).expand(batch, -1).clone()
    target = torch.einsum("bij,bpj->bpi", rotation, source) + translation[:, None]

    out = rigid_align_3d(source, target, min_points=6)

    assert out["valid"].all()
    assert torch.allclose(out["pose_c2w"][:, :3, :3], rotation, atol=1e-5)
    assert torch.allclose(out["pose_c2w"][:, :3, 3], translation, atol=1e-5)
    assert out["rmse_m"].max() < 1e-5


def test_rigid_align_3d_rejects_collinear_correspondences() -> None:
    source = torch.stack((torch.arange(6.0), torch.zeros(6), torch.ones(6)), -1)[None]
    out = rigid_align_3d(source, source, min_points=6)

    assert not out["valid"].any()
    assert torch.equal(out["pose_c2w"], torch.eye(4)[None])


def test_rigid_align_3d_ignores_nonfinite_unmatched_rows() -> None:
    source = torch.tensor(
        [[[-0.4, -0.2, 1.2], [0.3, -0.1, 1.8], [0.1, 0.5, 1.4], [-0.2, 0.4, 2.1], [0.6, 0.2, 1.6], [-0.5, 0.3, 1.7]]]
    )
    target = source + torch.tensor([0.4, -0.2, 0.7])
    source[:, 0] = torch.nan
    target[:, 0] = torch.nan
    matched = torch.tensor([[False, True, True, True, True, True]])

    out = rigid_align_3d(source, target, matched=matched, min_points=5)

    assert out["valid"].all()
    assert torch.allclose(out["pose_c2w"][:, :3, 3], torch.tensor([[0.4, -0.2, 0.7]]), atol=1e-5)
    assert out["rmse_m"].max() < 1e-5
