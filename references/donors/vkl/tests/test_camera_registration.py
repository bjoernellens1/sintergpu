from __future__ import annotations

import pytest
import torch

from vkl.kernels import (
    camera_rectify,
    camera_rectify_from_grid,
    camera_rectify_grid,
    depth_camera_rays,
    depth_register,
)


def _intrinsics(focal: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor([[[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]])


@pytest.mark.parametrize("batch", [1, 8])
def test_camera_rectify_identity_and_debug_coordinates(batch: int):
    image = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4).expand(batch, -1, -1, -1)
    out = camera_rectify(
        image,
        _intrinsics(2.0, 1.5, 1.0).expand(batch, -1, -1),
        _intrinsics(2.0, 1.5, 1.0).expand(batch, -1, -1),
        [3, 4],
        debug=True,
    )
    assert torch.allclose(out["image"], image)
    assert out["valid"].all()
    assert torch.equal(out["source_uv"][0, 0, :, 0], torch.arange(4, dtype=torch.float32))
    assert torch.equal(out["source_uv"][0, :, 0, 1], torch.arange(3, dtype=torch.float32))


def test_camera_rectify_brown_distortion_exposes_displacement_and_zero_invalid_border():
    image = torch.ones(1, 1, 5, 5)
    out = camera_rectify(
        image,
        _intrinsics(2.0, 2.0, 2.0),
        _intrinsics(2.0, 2.0, 2.0),
        (5, 5),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0]]),
        debug=True,
    )
    assert out["displacement"].max() > 0
    assert not out["valid"][0, 0, 0, 0]
    assert out["image"][0, 0, 0, 0] == 0
    assert (out["image"][~out["valid"].expand_as(out["image"])] == 0).all()


def test_precomputed_rectification_grid_matches_direct_brown_rectification():
    image = torch.rand(1, 3, 13, 17)
    source = _intrinsics(14.0, 8.0, 6.0)
    target = _intrinsics(13.0, 7.5, 5.5)
    coefficients = torch.tensor([[0.07, -0.02, 0.001, -0.002, 0.003]])
    direct = camera_rectify(
        image,
        source,
        target,
        (11, 15),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients,
        debug=True,
    )
    grid = camera_rectify_grid(
        source,
        target,
        (13, 17),
        (11, 15),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients,
    )
    cached = camera_rectify_from_grid(image, grid["source_uv"], grid["valid"])
    assert torch.equal(cached["valid"], direct["valid"])
    assert torch.allclose(cached["image"], direct["image"])
    assert set(grid) == {"source_uv", "valid"}
    debug_grid = camera_rectify_grid(
        source,
        target,
        (13, 17),
        (11, 15),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients,
        debug=True,
    )
    assert torch.allclose(debug_grid["target_ray"], direct["target_ray"])
    assert torch.allclose(debug_grid["displacement"], direct["displacement"])


def test_precomputed_depth_rays_match_direct_brown_depth_registration():
    depth = torch.rand(1, 1, 7, 9).add_(0.25)
    depth_intrinsics = _intrinsics(8.0, 4.0, 3.0)
    rgb_intrinsics = _intrinsics(9.0, 4.5, 3.5)
    coefficients = torch.tensor([[0.03, -0.01, 0.001, -0.002, 0.002]])
    transform = torch.eye(4)[None]
    direct = depth_register(
        depth,
        depth_intrinsics,
        rgb_intrinsics,
        transform,
        (8, 10),
        depth_distortion_model="brown_conrady",
        depth_distortion_coefficients=coefficients,
        debug=True,
    )
    rays = depth_camera_rays(
        depth_intrinsics,
        (7, 9),
        depth_distortion_model="brown_conrady",
        depth_distortion_coefficients=coefficients,
    )
    cached = depth_register(
        depth,
        depth_intrinsics,
        rgb_intrinsics,
        transform,
        (8, 10),
        depth_distortion_model="brown_conrady",
        depth_distortion_coefficients=coefficients,
        depth_rays=rays,
        debug=True,
    )
    assert torch.equal(cached["valid"], direct["valid"])
    assert torch.equal(cached["winner_source_index"], direct["winner_source_index"])
    assert torch.allclose(cached["depth"], direct["depth"])


def test_camera_rectify_brown_conrady_matches_opencv_on_valid_pixels():
    cv2 = pytest.importorskip("cv2")
    image = torch.rand(1, 3, 29, 37)
    source = torch.tensor([[[31.0, 0.0, 18.0], [0.0, 30.0, 14.0], [0.0, 0.0, 1.0]]])
    target = torch.tensor([[[28.0, 0.0, 17.0], [0.0, 29.0, 13.0], [0.0, 0.0, 1.0]]])
    coefficients = torch.tensor([[0.07, -0.02, 0.001, -0.002, 0.003]])
    out = camera_rectify(
        image,
        source,
        target,
        (23, 31),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients,
        debug=True,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        source[0].numpy(),
        coefficients[0].numpy(),
        None,
        target[0].numpy(),
        (31, 23),
        cv2.CV_32FC1,
    )
    expected = cv2.remap(image[0].permute(1, 2, 0).numpy(), map_x, map_y, cv2.INTER_LINEAR)
    valid = out["valid"][0, 0]
    assert torch.allclose(out["source_uv"][0, ..., 0][valid], torch.from_numpy(map_x)[valid], atol=2e-4, rtol=0)
    assert torch.allclose(out["source_uv"][0, ..., 1][valid], torch.from_numpy(map_y)[valid], atol=2e-4, rtol=0)
    # The coordinates above must agree tightly.  OpenCV INTER_LINEAR itself
    # quantizes interpolation fractions (INTER_BITS=5), whereas grid_sample
    # uses floating-point weights, so sampled values only have bounded—not
    # bitwise—agreement even for an identical map.
    assert torch.allclose(out["image"][0].permute(1, 2, 0)[valid], torch.from_numpy(expected)[valid], atol=4e-2, rtol=0)


def test_camera_rectify_rejects_invalid_calibration_by_default():
    image = torch.ones(1, 1, 5, 5)
    intrinsics = _intrinsics(2.0, 2.0, 2.0)
    bad_focal = intrinsics.clone()
    bad_focal[:, 0, 0] = 0
    with pytest.raises(ValueError):
        camera_rectify(image, bad_focal, intrinsics, (5, 5))

    bad_skew = intrinsics.clone()
    bad_skew[:, 0, 1] = 0.1
    with pytest.raises(ValueError):
        camera_rectify(image, bad_skew, intrinsics, (5, 5))

    with pytest.raises(ValueError):
        camera_rectify(
            image,
            intrinsics,
            intrinsics,
            (5, 5),
            source_distortion_model="brown_conrady",
            source_distortion_coefficients=torch.tensor([[float("nan"), 0, 0, 0, 0]]),
        )


@pytest.mark.parametrize("batch", [1, 8])
def test_depth_register_identity_preserves_metric_depth_and_debug_winner(batch: int):
    depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).expand(batch, -1, -1, -1)
    out = depth_register(
        depth,
        _intrinsics(1.0, 0.0, 0.0).expand(batch, -1, -1),
        _intrinsics(1.0, 0.0, 0.0).expand(batch, -1, -1),
        torch.eye(4)[None].expand(batch, -1, -1),
        [2, 2],
        debug=True,
    )
    assert torch.equal(out["depth"], depth)
    assert out["valid"].all()
    assert torch.equal(
        out["winner_source_index"],
        torch.tensor([[[[0, 1], [2, 3]]]]).expand(batch, -1, -1, -1),
    )


def test_depth_register_collision_uses_nearest_rgb_camera_z():
    depth = torch.tensor([[[[1.0, 2.0]]]])
    out = depth_register(
        depth,
        _intrinsics(0.1, 0.0, 0.0),
        _intrinsics(0.01, 0.0, 0.0),
        torch.eye(4)[None],
        (1, 1),
        debug=True,
    )
    assert out["depth"].item() == 1.0
    assert out["winner_source_index"].item() == 0
    assert out["collision_count"].item() == 2


def test_depth_register_rejects_invalid_and_behind_depth_samples():
    depth = torch.tensor([[[[0.0, float("nan"), float("inf"), 1.0]]]])
    transform = torch.eye(4)[None]
    transform[:, 2, 3] = -2.0
    out = depth_register(
        depth,
        _intrinsics(1.0, 0.0, 0.0),
        _intrinsics(1.0, 0.0, 0.0),
        transform,
        (1, 4),
        debug=True,
    )
    assert not out["valid"].any()
    assert (out["winner_source_index"] == -1).all()
    assert not out["source_valid"].any()
    assert torch.isfinite(out["depth"]).all()


def test_depth_register_rejects_nonrigid_transform_and_bad_interval():
    depth = torch.ones(1, 1, 2, 2)
    intrinsics = _intrinsics(1.0, 0.0, 0.0)
    transform = torch.eye(4)[None]
    transform[:, 0, 0] = 2
    with pytest.raises(ValueError):
        depth_register(depth, intrinsics, intrinsics, transform, (2, 2))
    with pytest.raises(ValueError):
        depth_register(depth, intrinsics, intrinsics, torch.eye(4)[None], (2, 2), min_depth=float("nan"))
    with pytest.raises(ValueError):
        depth_register(depth, intrinsics, intrinsics, torch.eye(4)[None], (2, 2), min_depth=2.0, max_depth=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_camera_registration_cpu_gpu_reference_parity():
    image = torch.rand(1, 3, 17, 19)
    intrinsics = _intrinsics(14.0, 9.0, 8.0)
    coefficients = torch.tensor([[0.02, -0.01, 0.001, -0.001, 0.003]])
    cpu = camera_rectify(
        image,
        intrinsics,
        intrinsics,
        (17, 19),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients,
    )
    gpu = camera_rectify(
        image.cuda(),
        intrinsics.cuda(),
        intrinsics.cuda(),
        (17, 19),
        source_distortion_model="brown_conrady",
        source_distortion_coefficients=coefficients.cuda(),
    )
    assert torch.equal(cpu["valid"], gpu["valid"].cpu())
    assert torch.allclose(cpu["image"], gpu["image"].cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("batch", [1, 8])
def test_camera_registration_fullgraph_matches_eager_on_gpu(batch: int):
    image = torch.rand(batch, 3, 32, 40, device="cuda")
    intrinsics = _intrinsics(30.0, 19.5, 15.5).to("cuda").expand(batch, -1, -1)
    target_intrinsics = _intrinsics(31.0, 18.75, 14.25).to("cuda").expand(batch, -1, -1)
    eager = camera_rectify(image, intrinsics, target_intrinsics, (32, 40), validate=False)
    compiled = torch.compile(camera_rectify, fullgraph=True)(
        image,
        intrinsics,
        target_intrinsics,
        (32, 40),
        validate=False,
    )
    assert torch.allclose(eager["image"], compiled["image"], atol=1e-4, rtol=1e-5)
    assert torch.equal(eager["valid"], compiled["valid"])

    depth = torch.rand(batch, 1, 32, 40, device="cuda").add_(0.25)
    transform = torch.eye(4, device="cuda").expand(batch, -1, -1)
    eager = depth_register(depth, intrinsics, intrinsics, transform, (32, 40), validate=False)
    compiled = torch.compile(depth_register, fullgraph=True)(
        depth,
        intrinsics,
        intrinsics,
        transform,
        (32, 40),
        validate=False,
    )
    assert torch.allclose(eager["depth"], compiled["depth"])
    assert torch.equal(eager["valid"], compiled["valid"])
