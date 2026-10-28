"""Independent D4 float32 fixtures and full-output frozen-donor parity."""
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from sintergpu.references.reprojection_numeric import (
    depth_reproject, rasterize_reprojection, reproject_rgbd, sample_pixels,
)


def cameras(b=1):
    return np.tile(np.eye(3, dtype=np.float32), (b, 1, 1)), np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))


def test_invalid_depth_preserves_translated_diagnostics():
    k, p = cameras()
    source = p.copy(); source[0, :3, 3] = [2, 3, 4]
    out = reproject_rgbd(np.array([[[[0, np.nan, np.inf, -1]]]], np.float32), k, source, k, p, target_hw=(3, 4))
    np.testing.assert_array_equal(out['projected_depth'], 4)
    np.testing.assert_array_equal(out['projected_xy'][:, 0], .5)
    np.testing.assert_array_equal(out['projected_xy'][:, 1], .75)
    assert not out['visible'].any()
    np.testing.assert_array_equal(out['overlap'], [0])
    np.testing.assert_array_equal(out['parallax'], [0])
    assert depth_reproject is reproject_rgbd


@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
def test_singleton_sampler_maps_all_finite_coordinates_to_only_pixel(mode):
    xy = np.array([[[[-3, 0, 4]], [[2, 0, -7]]]], np.float32)
    np.testing.assert_array_equal(sample_pixels(np.array([[[[9]]]], np.float32), xy, mode=mode), 9)


def test_sampler_ties_even_padding_and_normalized_roundtrip():
    field = np.array([[[[10, 20, 30, 40, 50]]]], np.float32)
    xy = np.array([[[[-.5, .5, 1.5, 2.5, 3.5, 4.5]], [[0, 0, 0, 0, 0, 0]]]], np.float32)
    np.testing.assert_array_equal(sample_pixels(field, xy), [[[[10, 10, 30, 30, 50, 50]]]])
    np.testing.assert_array_equal(sample_pixels(field, xy, mode='bilinear'), [[[[5, 15, 25, 35, 45, 25]]]])


def test_raster_minimum_max_confidence_and_all_close_support():
    k, p = cameras(); target_k = k.copy(); target_k[:, 0, 0] = .01
    eps = np.finfo(np.float32).eps
    d = np.array([[[[1, 1 + eps, 1 + 2 * eps, 2]]]], np.float32)
    c = np.array([[[[.2, .9, 1, 1]]]], np.float32)
    out = rasterize_reprojection(d, c, k, p, target_k, p, target_hw=(1, 2))
    np.testing.assert_array_equal(out['depth'], [[[[1, 0]]]])
    np.testing.assert_array_equal(out['confidence'], np.array([[[[.9, 0]]]], np.float32))
    np.testing.assert_array_equal(out['support_count'], [[[[2, 0]]]])
    np.testing.assert_array_equal(out['valid'], [[[[True, False]]]])


def frozen():
    package = '_sintergpu_d4_frozen_reference'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parents[1] / 'references/donors/vkl/src/vkl/kernels/reference')]
        sys.modules[package] = module
    return importlib.import_module(package + '.reproject')


def compare_all(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        expected_array = value.numpy()
        assert actual[key].shape == expected_array.shape, key
        assert actual[key].dtype == expected_array.dtype, key
        if expected_array.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual[key], expected_array, err_msg=key)
        else:
            # Separate scalar float32 products/additions vs CPU batched GEMM,
            # plus displacement sqrt and reduction, permit a few rounding ULPs.
            np.testing.assert_allclose(actual[key], expected_array, rtol=2e-6, atol=2e-6, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('b,h,w,th,tw', [(1, 3, 5, 5, 7), (8, 5, 7, 3, 9), (1, 1, 5, 1, 7), (8, 5, 1, 7, 1), (1, 1, 1, 1, 1)])
@pytest.mark.parametrize('with_target', [False, True])
def test_differential_all_reprojection_outputs_asymmetric_poses(b, h, w, th, tw, with_target):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(20261028 + b + h + w)
    d = rng.uniform(.8, 3, (b, 1, h, w)).astype(np.float32)
    # Invalid geometry must not hide the translated raw diagnostic coordinates.
    if d.size > 4:
        d.reshape(-1)[:4] = [0, -1, np.nan, np.inf]
    sk, sp = cameras(b); tk, tp = cameras(b)
    sk[:, 0, 0] = 3.7; sk[:, 1, 1] = 4.1
    sk[:, 0, 2] = .37; sk[:, 1, 2] = .21
    tk[:, 0, 0] = 3.3; tk[:, 1, 1] = 3.9
    tk[:, 0, 2] = .43; tk[:, 1, 2] = .17
    angle = np.float32(.071)
    sp[:, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    tp[:, :3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    sp[:, :3, 3] = [.13, -.07, .21]
    tp[:, :3, 3] = [-.02, .03, -.04]
    args = [d, sk, sp, tk, tp]
    kwargs = dict(target_hw=(th, tw))
    if with_target:
        td = rng.uniform(.5, 3, (b, 1, th, tw)).astype(np.float32)
        td.reshape(-1)[:min(3, td.size)] = [0, np.nan, np.inf][:min(3, td.size)]
        args.append(td)
    expected = frozen().reproject_rgbd(*[torch.from_numpy(a.copy()) for a in args], **kwargs)
    compare_all(reproject_rgbd(*args, **kwargs), expected)
    confidence = rng.uniform(0, 1, d.shape).astype(np.float32)
    raster_args = [d, confidence, sk, sp, tk, tp]
    compare_all(rasterize_reprojection(*raster_args, target_hw=(th, tw)),
                frozen().rasterize_reprojection(*[torch.from_numpy(a.copy()) for a in raster_args], target_hw=(th, tw)))


@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
@pytest.mark.parametrize('h,w', [(1, 1), (1, 7), (5, 1), (3, 5), (5, 7)])
def test_sampler_differential_roundtrip_zero_padding_and_nonfinite(mode, h, w):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(15)
    field = rng.normal(size=(2, 3, h, w)).astype(np.float32)
    xy = rng.uniform(-2, 8, (2, 2, 3, 11)).astype(np.float32)
    xy[0, 0, 0, :8] = [-.5, .5, 1.5, 2.5, np.nan, np.inf, -np.inf, 1e30]
    grid = torch.from_numpy(xy).permute(0, 2, 3, 1).clone()
    grid[..., 0] = 2 * grid[..., 0] / max(w - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(h - 1, 1) - 1
    expected = torch.nn.functional.grid_sample(torch.from_numpy(field), grid, mode=mode, padding_mode='zeros', align_corners=True)
    np.testing.assert_allclose(sample_pixels(field, xy, mode=mode), expected.numpy(), rtol=1e-6, atol=3e-7, equal_nan=True)


@pytest.mark.parametrize('z', [-1, 0, np.finfo(np.float32).eps / 2, np.finfo(np.float32).eps, 1])
def test_raw_z_visibility_and_clamped_projection(z):
    k, p = cameras(); source = p.copy(); source[:, 2, 3] = np.float32(z) - np.float32(1)
    d = np.ones((1, 1, 1, 1), np.float32)
    out = reproject_rgbd(d, k, source, k, p, target_hw=(1, 1))
    assert bool(out['visible'].item()) == (np.float32(1) + source[0, 2, 3] > 0)
    torch = pytest.importorskip('torch')
    compare_all(out, frozen().reproject_rgbd(*[torch.from_numpy(a) for a in [d, k, source, k, p]], target_hw=(1, 1)))


def test_directional_occlusion_is_strict_at_threshold_and_consistency_is_masked():
    k, p = cameras(); d = np.ones((1, 1, 1, 4), np.float32)
    td = np.array([[[[.75, .5, 1.5, np.nan]]]], np.float32)
    out = reproject_rgbd(d, k, p, k, p, td, consistency_threshold=.25)
    np.testing.assert_array_equal(out['depth_consistency'], [[[[.25, .5, .5, 0]]]])
    np.testing.assert_array_equal(out['depth_consistency_valid'], [[[[True, True, True, False]]]])
    np.testing.assert_array_equal(out['occluded'], [[[[False, True, False, False]]]])
    torch = pytest.importorskip('torch')
    compare_all(out, frozen().reproject_rgbd(*[torch.from_numpy(a) for a in [d, k, p, k, p, td]], consistency_threshold=.25))


def test_raster_pre_round_visibility_and_half_ties():
    k, p = cameras(); tk = k.copy(); tk[:, 0, 0] = .5
    d = np.ones((1, 1, 1, 7), np.float32)
    out = rasterize_reprojection(d, d, k, p, tk, p, target_hw=(1, 3))
    # Coordinates 0,.5,1,1.5,2 are visible, 2.5 and3 fail before rounding.
    np.testing.assert_array_equal(out['support_count'], [[[[2, 1, 2]]]])
    torch = pytest.importorskip('torch')
    compare_all(out, frozen().rasterize_reprojection(*[torch.from_numpy(a) for a in [d, d, k, p, tk, p]], target_hw=(1, 3)))


def test_zero_batch_and_empty_raster_outputs():
    k, p = cameras(0); d = np.ones((0, 1, 3, 5), np.float32)
    out = reproject_rgbd(d, k, p, k, p, target_hw=(2, 4))
    assert out['projected_xy'].shape == (0, 2, 3, 5)
    assert out['overlap'].shape == (0,)
    out = rasterize_reprojection(d, d, k, p, k, p, target_hw=(2, 4))
    assert out['support_count'].shape == (0, 1, 2, 4)
    k, p = cameras(); d = np.zeros((1, 1, 2, 3), np.float32)
    for value in rasterize_reprojection(d, d, k, p, k, p, target_hw=(1, 1)).values():
        assert not value.any()


@pytest.mark.parametrize('mutation', ['depth_dtype', 'camera_shape', 'focal', 'target_shape', 'target_hw', 'threshold', 'confidence'])
def test_profile_validation(mutation):
    k, p = cameras(); d = np.ones((1, 1, 2, 3), np.float32)
    args = [d, k, p, k, p]; kwargs = dict(target_hw=(2, 3))
    if mutation == 'depth_dtype': args[0] = d.astype(np.float64)
    if mutation == 'camera_shape': args[1] = k[:, :2]
    if mutation == 'focal': args[1] = k.copy(); args[1][0, 0, 0] = 0
    if mutation == 'target_shape': args.append(d[:, 0])
    if mutation == 'target_hw': args.append(d); kwargs['target_hw'] = (3, 2)
    if mutation == 'threshold': kwargs['consistency_threshold'] = 0
    with pytest.raises(ValueError):
        if mutation == 'confidence':
            rasterize_reprojection(d, d * 2, k, p, k, p, target_hw=(2, 3))
        else:
            reproject_rgbd(*args, **kwargs)


def test_normalized_roundtrip_can_move_a_pixel_half_tie():
    # width4: .5 -> normalized -.6666666 -> unnormalized .50000006.
    # Sampling raw pixel coordinates directly would incorrectly choose pixel0.
    field = np.array([[[[10, 20, 30, 40]]]], np.float32)
    xy = np.array([[[[.5]], [[0]]]], np.float32)
    np.testing.assert_array_equal(sample_pixels(field, xy), [[[[20]]]])
    torch = pytest.importorskip('torch')
    grid = torch.tensor([[[[2 * np.float32(.5) / np.float32(3) - 1, 0]]]], dtype=torch.float32)
    expected = torch.nn.functional.grid_sample(torch.from_numpy(field), grid, mode='nearest', padding_mode='zeros', align_corners=True)
    np.testing.assert_array_equal(expected.numpy(), [[[[20]]]])


def test_projection_clamps_denominator_but_visibility_uses_positive_raw_z():
    k, p = cameras(); source = p.copy()
    eps = np.finfo(np.float32).eps
    source[0, 0, 3] = eps
    source[0, 2, 3] = np.float32(eps / 2) - np.float32(1)
    d = np.ones((1, 1, 1, 1), np.float32)
    out = reproject_rgbd(d, k, source, k, p, target_hw=(1, 2))
    np.testing.assert_array_equal(out['projected_depth'], eps / 2)
    np.testing.assert_array_equal(out['projected_xy'][:, 0], 1)
    assert out['visible'].all()
    torch = pytest.importorskip('torch')
    compare_all(out, frozen().reproject_rgbd(*[torch.from_numpy(a) for a in [d, k, source, k, p]], target_hw=(1, 2)))
