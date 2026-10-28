# SPDX-License-Identifier: Apache-2.0
"""Independent checks for the two deliberately different pyramid profiles."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sintergpu.references.depth_pyramid_profiles import vkl_level, slam_level

F = np.float32
EPS = F(np.finfo(np.float32).eps)
DONOR = Path(__file__).resolve().parents[1] / 'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py'


@pytest.mark.parametrize('mask', range(16))
def test_all_masks_and_profiles(mask):
    depth = np.array([1, 1, 1, 1], dtype=np.float32).reshape(1, 2, 2)
    valid = np.array([7 if mask & (1 << i) else 0 for i in range(4)], np.uint32).reshape(depth.shape)
    confidence = np.array([.1, .2, .3, .4], np.float32).reshape(depth.shape)
    d, v, c = vkl_level(depth, valid, confidence, EPS)
    expected_index = next((i for i in range(4) if mask & (1 << i)), None)
    assert v.item() == (mask != 0)
    assert d.item() == (1 if mask else 0)
    expected = F(0) if mask == 0 else F(.25) if mask == 15 else confidence.flat[expected_index]
    assert c.item() == expected
    slam_input = np.where(valid[0] != 0, depth[0], F(0))
    assert slam_level(slam_input, F(0)).item() == (1 if mask else 0)


def test_profiles_do_not_conflate_minimum_with_average_or_rejection():
    d = np.array([1, 2, 0, 0], np.float32).reshape(1, 2, 2)
    mask = np.ones_like(d, dtype=np.uint32)
    conf = np.array([.3, .8, .7, .6], np.float32).reshape(d.shape)
    out, valid, chosen = vkl_level(d, mask, conf, EPS)
    assert (out.item(), valid.item(), chosen.item()) == (1, 1, conf.flat[0])
    assert slam_level(d[0], F(1)).item() == F(1.5)
    assert slam_level(d[0], np.nextafter(F(1), F(0))).item() == 0


def test_threshold_neighbors_and_first_tie():
    threshold = F(EPS * F(32))
    for jump, mean in [(np.nextafter(threshold, F(0)), True), (threshold, True), (np.nextafter(threshold, F(np.inf)), False)]:
        d = np.array([1, 1, 1, F(1) + jump], np.float32).reshape(1, 2, 2)
        c = np.array([.125, .25, .5, 1], np.float32).reshape(d.shape)
        out, _, conf = vkl_level(d, np.ones_like(d, np.uint32), c, EPS)
        # Values are rounded on addition; compare the actual representable gap.
        actual_gap = F(d.flat[-1] - d.flat[0])
        assert (conf.item() != c.flat[0]) == (actual_gap <= threshold)
        assert out.item() >= 1
    d = np.array([[1, 2], [0, -1]], np.float32)
    for jump, expected in [(np.nextafter(F(1), F(0)), 0), (F(1), 1.5), (np.nextafter(F(1), F(np.inf)), 1.5)]:
        assert slam_level(d, jump).item() == expected


def test_nonfinite_large_and_signed_zero():
    for values, jump, expected in [([np.nan, -np.inf, -0., -1], 1, 0), ([np.inf, 0, 0, 0], np.inf, np.inf), ([1e21, 0, 0, 0], 0, 0), ([1e21, 0, 0, 0], 1e21, F(1e21))]:
        assert slam_level(np.array(values, np.float32).reshape(2,2), F(jump)).item() == expected
    d = np.array([np.nan, np.inf, -0., -np.inf, 1e21, 2, 2, 2], np.float32).reshape(2,2,2)
    mask = np.ones(d.shape, np.uint32)
    c = np.arange(8, dtype=np.float32).reshape(d.shape) / 8
    out, valid, conf = vkl_level(d, mask, c, EPS)
    np.testing.assert_array_equal(out.ravel(), [0, 2])
    np.testing.assert_array_equal(valid.ravel(), [0, 1])
    assert conf.ravel()[1] == c.ravel()[5]


@pytest.mark.parametrize('shape', [(2,1,1), (2,1,7), (2,7,1), (3,5,7), (2,34,38)])
def test_shape_repeated_and_inputs_unchanged(shape):
    rng = np.random.default_rng(811)
    depth = rng.uniform(.1, 4, shape).astype(np.float32)
    valid = rng.integers(0, 3, shape, dtype=np.uint32)
    confidence = rng.random(shape, dtype=np.float32)
    before = [x.copy() for x in (depth, valid, confidence)]
    first = vkl_level(depth, valid, confidence, EPS)
    second = vkl_level(depth, valid, confidence, EPS)
    assert first[0].shape == (shape[0], (shape[1]+1)//2, (shape[2]+1)//2)
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
    for a,b in zip((depth,valid,confidence),before): np.testing.assert_array_equal(a,b)


def test_slam_even_shape_contract_and_repeat():
    depth = np.arange(1, 69*70+1, dtype=np.float32).reshape(70,69)[:, :68]
    before = depth.copy()
    a = slam_level(depth, F(100))
    assert a.shape == (35,34)
    np.testing.assert_array_equal(a, slam_level(depth, F(100)))
    np.testing.assert_array_equal(depth, before)
    with pytest.raises(ValueError): slam_level(np.ones((3,2), np.float32), F(1))


def test_frozen_vkl_donor_multilevel():
    torch = pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('frozen_vkl_depth_pyramid', DONOR)
    donor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(donor)
    rng = np.random.default_rng(922)
    for shape in [(2,1,7), (2,5,7), (3,8,10), (1,1,1)]:
        d = rng.uniform(.2, 4, shape).astype(np.float32)
        v = rng.integers(0, 3, shape, dtype=np.uint32)
        c = rng.random(shape, dtype=np.float32)
        d.flat[0] = np.nan
        original = donor.depth_pyramid(depth=torch.from_numpy(d[:,None]), valid=torch.from_numpy((v!=0)[:,None]), confidence=torch.from_numpy(c[:,None]))
        usable = (v != 0) & np.isfinite(d) & (d > 0)
        current = (np.where(usable,d,F(0)), usable.astype(np.uint32), np.where(usable,c,F(0)))
        for key in ('/1','/2','/4','/8'):
            if key not in original: break
            for actual, name in zip(current, ('depth','valid','confidence')):
                expected = original[key][name].numpy()[:,0]
                if name == 'valid': np.testing.assert_array_equal(actual, expected)
                else: np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            if current[0].shape[-2:] == (1,1): break
            current = vkl_level(*current, EPS)


def test_frozen_source_identities():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    expected = {
        'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py': 'a1e7fdd46785607a9b52f98819469e5ebe7192558df2c574a583c553b33b6ed1',
        'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/shaders/tracker.slang': 'd6575a6d6f73df2dbc98e23c3b935ce0eb9e697112c64dd85f096fad649b690a',
    }
    for path, digest in expected.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def test_poisoned_input_guards_and_sequential_accumulation():
    # Noncontiguous views leave poisoned surrounding storage untouched.
    storage = np.full((2, 6, 8), np.nan, np.float32)
    depth = storage[:, 1:5, 1:7]
    depth[:] = 1
    mask_storage = np.full((2, 6, 8), 0xffffffff, np.uint32)
    mask = mask_storage[:, 1:5, 1:7]
    mask[:] = 1
    conf_storage = np.full((2, 6, 8), np.nan, np.float32)
    conf = conf_storage[:, 1:5, 1:7]
    conf[:] = .5
    before = [x.copy() for x in (storage, mask_storage, conf_storage)]
    for _ in range(2):
        out, valid, certainty = vkl_level(depth, mask, conf, EPS)
        np.testing.assert_array_equal(out, np.ones((2,2,3), np.float32))
        np.testing.assert_array_equal(valid, np.ones((2,2,3), np.uint32))
        np.testing.assert_array_equal(certainty, np.full((2,2,3), .5, np.float32))
    for actual, expected in zip((storage, mask_storage, conf_storage), before):
        np.testing.assert_array_equal(actual, expected)
    # Large-first accumulation rounds each addition, not an ideal real sum.
    block = np.array([[16777216, 1], [1, 1]], np.float32)
    assert slam_level(block, F(16777216)).item() == F(4194304)
