"""T0 literal semantics and complete frozen CPU donor differential tests."""
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb, temporal_rgbd_reference


def fields(b=1, n=1, h=1, w=1):
    return dict(target_depth=np.ones((b, 1, h, w), np.float32),
                source_rgb=np.ones((b, n, 3, h, w), np.float32),
                source_depth=np.ones((b, n, 1, h, w), np.float32),
                target_intrinsics=np.tile(np.eye(3, dtype=np.float32), (b, 1, 1)),
                target_pose_c2w=np.tile(np.eye(4, dtype=np.float32), (b, 1, 1)),
                source_intrinsics=np.tile(np.eye(3, dtype=np.float32), (b, n, 1, 1)),
                source_pose_c2w=np.tile(np.eye(4, dtype=np.float32), (b, n, 1, 1)))


def test_prior_only_depth_never_uses_target_depth():
    values = fields()
    for depth in [2, 3]:
        values['target_depth'].fill(depth)
        out = temporal_rgbd_reference(**values, consistency_threshold=4)
        np.testing.assert_array_equal(out['depth'], 1)
        assert out['valid'].all()


def test_invalid_alignment_retains_unmasked_rgb():
    values = fields(); values['target_depth'].fill(0)
    values['target_rgb'] = np.zeros((1, 3, 1, 1), np.float32)
    out = temporal_reproject_rgb(**values)
    np.testing.assert_array_equal(out['aligned_rgb'], 1)
    assert not out['aligned_valid'].any()
    assert not out['alignment_confidence'].any()


def test_rgb_uses_weights_outside_winning_depth_cluster():
    values = fields(n=3)
    values['source_depth'][0, 2].fill(1.2)
    values['source_rgb'][0, :2].fill(0)
    out = temporal_rgbd_reference(**values, consistency_threshold=.1)
    assert out['valid'].all()
    np.testing.assert_array_equal(out['depth'], 1)
    np.testing.assert_array_equal(out['support_count'], 2)
    weight = np.exp(-np.float32(1.2 - 1) / np.float32(.1))
    np.testing.assert_allclose(out['rgb'], weight / (2 + weight), rtol=1e-6)
    np.testing.assert_allclose(out['source_disagreement'], np.float32(1.2) - 1, rtol=0, atol=0)


def test_final_invalid_rgb_retains_depth_support_and_disagreement():
    values = fields(n=2); values['target_depth'].fill(0)
    values['source_depth'][0, 1].fill(2)
    out = temporal_rgbd_reference(**values)
    for key in ('rgb', 'depth', 'confidence', 'valid'):
        assert not out[key].any()
    np.testing.assert_array_equal(out['support_count'], 1)
    np.testing.assert_array_equal(out['temporal_support_count'], 1)
    np.testing.assert_array_equal(out['source_disagreement'], 1)


def test_zero_confidence_cannot_support_depth():
    values = fields(n=2)
    values['source_confidence'] = np.zeros_like(values['source_depth'])
    out = temporal_rgbd_reference(**values)
    for value in out.values():
        assert not value.any()


def frozen():
    package = '_sintergpu_t0_frozen_reference'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parents[1] / 'references/donors/vkl/src/vkl/kernels/reference')]
        sys.modules[package] = module
    return importlib.import_module(package + '.temporal_reproject')


def compare_all(actual, expected):
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        array = tensor.numpy()
        assert actual[key].shape == array.shape, key
        assert actual[key].dtype == array.dtype, key
        if array.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual[key], array, err_msg=key)
        else:
            # D4 float32 scalar arithmetic and bilinear interpolation can differ
            # a few ULPs from CPU GEMM/interpolation. The exp and normalized RGB
            # sum add rounding; exact visibility/support tests remain mandatory.
            np.testing.assert_allclose(actual[key], array, rtol=3e-6, atol=2e-6, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('b,n,h,w', [(1, 1, 3, 5), (8, 3, 5, 7), (1, 2, 1, 5), (8, 2, 5, 1), (1, 3, 1, 1)])
@pytest.mark.parametrize('depth_option,confidence_option', [(False, False), (True, False), (False, True), (True, True)])
def test_all_temporal_sampling_outputs_frozen_differential(b, n, h, w, depth_option, confidence_option):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(81 + b + n + h)
    values = fields(b, n, h, w)
    values['source_rgb'] = rng.uniform(0, 1, (b, n, 3, h, w)).astype(np.float32)
    values['target_depth'] = rng.uniform(.9, 1.1, (b, 1, h, w)).astype(np.float32)
    values['source_depth'] = rng.uniform(.9, 1.1, (b, n, 1, h, w)).astype(np.float32)
    if values['target_depth'].size > 4:
        values['target_depth'].reshape(-1)[:4] = [0, -1, np.nan, np.inf]
    if values['source_depth'].size > 4:
        values['source_depth'].reshape(-1)[-4:] = [0, -1, np.nan, np.inf]
    values['target_intrinsics'][:, 0, 0] = 4.3
    values['target_intrinsics'][:, 1, 1] = 3.7
    values['target_intrinsics'][:, 0, 2] = .31
    values['target_intrinsics'][:, 1, 2] = .17
    values['source_intrinsics'][..., 0, 0] = 4.1
    values['source_intrinsics'][..., 1, 1] = 3.9
    values['source_intrinsics'][..., 0, 2] = .27
    values['source_intrinsics'][..., 1, 2] = .21
    angle = np.float32(.017)
    values['source_pose_c2w'][..., :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    values['source_pose_c2w'][..., :3, 3] = [.012, -.007, .021]
    values['target_pose_c2w'][:, :3, 3] = [-.004, .003, -.011]
    if not depth_option:
        values.pop('source_depth')
    if confidence_option:
        values['source_confidence'] = rng.uniform(0, 1, (b, n, 1, h, w)).astype(np.float32)
    values['target_rgb'] = rng.uniform(0, 1, (b, 3, h, w)).astype(np.float32)
    expected = frozen().temporal_reproject_rgb(**{k: torch.from_numpy(v.copy()) for k, v in values.items()})
    compare_all(temporal_reproject_rgb(**values), expected)


@pytest.mark.parametrize('b,n,h,w', [(1, 1, 3, 5), (8, 3, 5, 7), (1, 2, 1, 5), (8, 2, 5, 1), (1, 3, 1, 1)])
@pytest.mark.parametrize('confidence_option,min_support', [(False, 1), (True, 1), (True, 2), (True, 4)])
def test_all_reference_outputs_frozen_differential(b, n, h, w, confidence_option, min_support):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(327 + b + n + h)
    values = fields(b, n, h, w)
    values['source_rgb'] = rng.uniform(0, 1, (b, n, 3, h, w)).astype(np.float32)
    values['target_depth'] = rng.uniform(.9, 1.1, (b, 1, h, w)).astype(np.float32)
    values['source_depth'] = rng.uniform(.9, 1.1, (b, n, 1, h, w)).astype(np.float32)
    values['source_pose_c2w'][..., :3, 3] = [.012, -.007, .021]
    values['target_pose_c2w'][:, :3, 3] = [-.004, .003, -.011]
    if values['target_depth'].size > 4:
        values['target_depth'].reshape(-1)[:4] = [0, -1, np.nan, np.inf]
    if values['source_depth'].size > 4:
        values['source_depth'].reshape(-1)[-4:] = [0, -1, np.nan, np.inf]
    if confidence_option:
        values['source_confidence'] = rng.uniform(0, 1, (b, n, 1, h, w)).astype(np.float32)
    expected = frozen().temporal_rgbd_reference(**{k: torch.from_numpy(v.copy()) for k, v in values.items()}, min_support=min_support)
    compare_all(temporal_rgbd_reference(**values, min_support=min_support), expected)


@pytest.mark.parametrize('which', ['prior', 'invalid_rgb', 'conflict', 'zero_confidence', 'min_support'])
def test_literal_fixtures_all_outputs_against_frozen(which):
    torch = pytest.importorskip('torch')
    values = fields(n=3)
    kwargs = dict(consistency_threshold=.1)
    if which == 'prior':
        values['target_depth'].fill(2)
        kwargs['consistency_threshold'] = 4
    elif which == 'invalid_rgb':
        values['target_depth'].fill(0)
        values['source_depth'][0, 2].fill(2)
    elif which == 'conflict':
        values['source_depth'][0, 2].fill(1.2)
        values['source_rgb'][0, :2].fill(0)
    elif which == 'zero_confidence':
        values['source_confidence'] = np.zeros_like(values['source_depth'])
    else:
        kwargs['min_support'] = 4
    compare_all(temporal_rgbd_reference(**values, **kwargs),
                frozen().temporal_rgbd_reference(**{k: torch.from_numpy(v) for k, v in values.items()}, **kwargs))


def test_singleton_threshold_boundary_and_alignment_confidence():
    torch = pytest.importorskip('torch')
    values = fields(n=3)
    values['source_depth'][0, :, 0, 0, 0] = [.75, .5, 1.5]
    values['source_confidence'] = np.full((1, 3, 1, 1, 1), .5, np.float32)
    values['target_rgb'] = np.zeros((1, 3, 1, 1), np.float32)
    out = temporal_reproject_rgb(**values, consistency_threshold=.25)
    np.testing.assert_array_equal(out['aligned_valid'].reshape(-1), [True, False, True])
    np.testing.assert_allclose(out['alignment_confidence'].reshape(-1), [.5 * np.exp(-1), 0, .5 * np.exp(-2)], rtol=1e-6)
    compare_all(out, frozen().temporal_reproject_rgb(**{k: torch.from_numpy(v) for k, v in values.items()}, consistency_threshold=.25))


@pytest.mark.parametrize('mutation', ['rgb', 'depth', 'pose', 'confidence_shape', 'confidence_nan', 'confidence_range', 'threshold', 'min_support', 'neighbors'])
def test_profile_validation(mutation):
    values = fields()
    kwargs = {}
    if mutation == 'rgb': values['source_rgb'] = values['source_rgb'][:, :, :2]
    if mutation == 'depth': values['source_depth'] = values['source_depth'][:, 0]
    if mutation == 'pose': values['source_pose_c2w'] = values['source_pose_c2w'][..., :3, :]
    if mutation == 'confidence_shape': values['source_confidence'] = np.ones((1,), np.float32)
    if mutation == 'confidence_nan': values['source_confidence'] = np.full_like(values['source_depth'], np.nan)
    if mutation == 'confidence_range': values['source_confidence'] = np.full_like(values['source_depth'], 1.1)
    if mutation == 'threshold': kwargs['consistency_threshold'] = 0
    if mutation == 'min_support': kwargs['min_support'] = True
    if mutation == 'neighbors':
        for key in ('source_rgb', 'source_depth', 'source_intrinsics', 'source_pose_c2w'):
            values[key] = values[key][:, :0]
    with pytest.raises(ValueError):
        temporal_rgbd_reference(**values, **kwargs)
