import pytest
import torch

from vkl.kernels import (
    fast_select,
    pose_observable_select,
    shi_tomasi_select,
    spatial_select_points,
    uniform_select,
)


def test_selectors_share_the_bounded_d6_coordinate_contract():
    rgb = torch.zeros(1, 3, 17, 19)
    rgb[..., 5:12, 6:13] = 1
    uniform = uniform_select(rgb, num_points=12, margin=3)
    assert uniform["valid"].all()
    for selector in (shi_tomasi_select, fast_select):
        out = selector(rgb, num_points=12, margin=3)
        assert out["pixels_xy"].shape == (1, 12, 2)
        assert out["pixels_xy"].dtype == torch.int64
        assert out["score"].shape == (1, 12)
        assert out["valid"].shape == (1, 12)
        assert out["valid"].any()
        assert (out["score"][~out["valid"]] == 0).all()


def test_selectors_do_not_create_features_at_conditioning_mask_boundary():
    rgb = torch.ones(1, 3, 25, 25)
    valid = torch.ones(1, 1, 25, 25, dtype=torch.bool)
    valid[..., :, :12] = False
    for selector in (shi_tomasi_select, fast_select):
        out = selector(rgb, rgb_valid=valid, num_points=8, margin=3)
        assert not out["valid"].any()

    uniform = uniform_select(rgb, rgb_valid=valid, num_points=16, margin=3)
    points = uniform["pixels_xy"][0]
    invalid_points = points[:, 0] < 12
    assert torch.equal(~uniform["valid"][0][invalid_points], torch.ones_like(uniform["valid"][0][invalid_points]))


def test_shi_tomasi_and_fast_prefer_a_real_corner_over_flat_pixels():
    rgb = torch.zeros(1, 3, 21, 21)
    rgb[..., 10:, 10:] = 1
    for selector in (shi_tomasi_select, fast_select):
        out = selector(rgb, num_points=4, margin=3)
        points = out["pixels_xy"][0, out["valid"][0]]
        assert ((points[:, 0] - 10).abs() <= 2).logical_and((points[:, 1] - 10).abs() <= 2).any()


def test_feature_selectors_do_not_fabricate_flat_image_keypoints():
    rgb = torch.zeros(1, 3, 21, 21)
    for selector in (shi_tomasi_select, fast_select):
        out = selector(rgb, num_points=8, margin=3)
        assert not out["valid"].any()
        assert (out["score"] == 0).all()


def test_fast_rejects_margin_smaller_than_ring_radius():
    with pytest.raises(ValueError):
        fast_select(torch.zeros(1, 3, 21, 21), num_points=8, margin=2)


def test_spatial_select_points_enforces_a_common_tile_quota():
    pixels = torch.tensor([[[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [7, 1], [1, 7]]])
    score = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]])
    valid = torch.ones((1, 7), dtype=torch.bool)
    out = spatial_select_points(pixels, score, valid, num_points=4, height=8, width=8, tiles=(2, 2))
    assert out["valid"].sum() == 3


def test_spatial_select_points_masks_nonfinite_and_out_of_bounds_candidates():
    out = spatial_select_points(
        torch.tensor([[[1, 1], [-1, 2], [2, 2], [9, 1]]]),
        torch.tensor([[4.0, 100.0, float("nan"), 90.0]]),
        torch.ones(1, 4, dtype=torch.bool), num_points=4, height=8, width=8, tiles=(2, 2),
    )
    assert out["valid"].sum() == 1
    assert out["pixels_xy"][0, 0].tolist() == [1, 1]
    assert (out["pixels_xy"][0, ~out["valid"][0]] == 0).all()


@pytest.mark.parametrize("batch", [1, 8])
def test_pose_observable_selector_uses_known_baseline_and_common_contract(batch: int):
    rgb = torch.zeros(batch, 3, 21, 21)
    rgb[..., :, 10:] = 1.0
    intrinsics = torch.tensor([[[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]]).expand(batch, -1, -1)
    reference = torch.eye(4)[None].expand(batch, -1, -1)
    neighbor = reference.clone()
    neighbor[:, 0, 3] = 0.1
    out = pose_observable_select(rgb, intrinsics, reference, neighbor, num_points=8, margin=3)
    assert out["pixels_xy"].shape == (batch, 8, 2)
    assert out["valid"].any()


def test_pose_observable_selector_rejects_stationary_baseline():
    rgb = torch.rand(1, 3, 21, 21)
    intrinsics = torch.tensor([[[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4)[None]
    out = pose_observable_select(rgb, intrinsics, pose, pose, num_points=8, margin=3)
    assert not out["valid"].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("batch", [1, 8])
def test_pose_observable_selector_fullgraph_matches_eager_on_gpu(batch: int):
    rgb = torch.rand(batch, 3, 32, 40, device="cuda")
    intrinsics = torch.tensor([[[30.0, 0.0, 19.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]]], device="cuda").expand(batch, -1, -1)
    reference = torch.eye(4, device="cuda")[None].expand(batch, -1, -1)
    neighbor = reference.clone()
    neighbor[:, 0, 3] = 0.1
    valid = torch.ones(batch, 1, 32, 40, dtype=torch.bool, device="cuda")
    eager = pose_observable_select(rgb, intrinsics, reference, neighbor, rgb_valid=valid, num_points=64)
    compiled = torch.compile(pose_observable_select, fullgraph=True)(
        rgb, intrinsics, reference, neighbor, rgb_valid=valid, num_points=64
    )
    assert torch.equal(eager["pixels_xy"], compiled["pixels_xy"])
    assert torch.allclose(eager["score"], compiled["score"], atol=1e-6, rtol=1e-6)
    assert torch.equal(eager["valid"], compiled["valid"])
