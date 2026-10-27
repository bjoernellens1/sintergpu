"""Independent CPU contracts and opt-in borrowed-resource Slang parity."""
import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from sintergpu.references.vkl.numpy import (depth_convert, depth_pyramid_level, mask_u32,
    photometric_isp, project_depth_evidence)

ROOT = Path(__file__).resolve().parents[1]


def projection_case():
    # Irregular N crosses 256-thread boundary; two independent camera batches.
    xyz = np.zeros((2, 259, 3), np.float32)
    xyz[..., 2] = 1
    xyz[:, :9, 0] = [0.5, 1.5, 2.5, -0.5, -1.5, 1e30, np.nan, np.inf, 0]
    xyz[:, 9:13, 2] = [0, -1, np.finfo(np.float32).eps, np.inf]
    masks = np.ones((2, 259), bool)
    visible = masks.copy(); visible[:, 15] = False
    provisional = masks.copy(); provisional[:, 16] = False
    candidate = masks.copy(); candidate[:, 16:18] = False
    pose = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)); pose[1, 1, 3] = 1
    k = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    sensor = np.ones((2, 1, 3, 4), np.float32)
    sensor[:, 0, 0, 2] = 2
    sensor[1, 0, 1, 2] = np.nan
    sensor[:,0,2,1:] = [0,-1,np.inf]
    xyz[:,18:21,:] = [[1,2,1],[2,2,1],[3,2,1]]
    return xyz, visible, provisional, candidate, pose, k, sensor


def test_projection_ties_invalid_and_batch():
    args = projection_case()
    out = project_depth_evidence(*args, support_threshold=0.5, conflict_threshold=0.5)
    assert out['projected_valid'][0, :7].tolist() == [True, True, True, True, False, False, False]
    assert out['depth_residual'][0, :4].tolist() == [0, 1, 1, 0]
    assert out['conflict'][0, 1] and not out['sensor_valid'][1, 1]
    assert not out['eligible'][0, 6:8].any()
    assert not out['projected_valid'][:, 9:13].any()
    assert out['eligible'][0, 16] == False
    assert out['depth_residual'][~out['sensor_valid']].tolist() == [0] * np.count_nonzero(~out['sensor_valid'])


def test_strict_thresholds_and_explicit_mask_conversion():
    xyz, visible, provisional, candidate, pose, k, sensor = projection_case()
    sensor[:] = 1.5
    out = project_depth_evidence(xyz, visible, provisional, candidate, pose, k, sensor, support_threshold=0.5, conflict_threshold=0.5)
    assert not out['support'].any() and not out['conflict'].any()
    converted = mask_u32(np.array([False, True], bool))
    assert converted.dtype == np.uint32 and converted.tolist() == [0, 1]
    with pytest.raises(ValueError):
        mask_u32(np.array([1], np.uint8))


def test_photometric_black_singleton_and_order():
    rgb = np.array([[[[0, .25]], [[.5, 1]], [[.75, .1]]]], np.float32)
    color = np.array([[[0, 1, 0], [0, 0, 1], [1, 0, 0]]], np.float32)
    result = photometric_isp(rgb, exposure_ev=np.array([1]), rgb_gain=np.array([[1, .5, 2]]), color_matrix=color)
    np.testing.assert_array_equal(result, np.minimum(1, np.stack((rgb[:, 1], rgb[:, 2]*4, rgb[:, 0]*2), axis=1)))
    black = photometric_isp(np.zeros((1, 3, 1, 1), np.float32), response_gamma=np.array([.5]))
    assert not black.any()
    # torch.linspace(-1,1,1) is -1, so singleton radius squared is 2.
    attenuated = photometric_isp(np.ones((1, 3, 1, 1), np.float32), vignette=np.ones((1, 3, 3), np.float32))
    np.testing.assert_allclose(attenuated, np.exp(-14), rtol=1e-6)


@pytest.mark.parametrize('representation', [0, 1, 2])
def test_depth_convert_invalid(representation):
    values = np.array([0, -1, 2, np.nan, np.inf], np.float32)
    out = depth_convert(values, np.ones(5, bool), np.ones(5, np.float32), representation=representation, scale=2, focal_baseline=4)
    assert out['valid'].tolist() == [False, False, True, False, False]
    assert out['depth'][2] == [4, .25, 2][representation]
    assert not out['confidence'][~out['valid']].any()


def test_pyramid_padding_and_first_minimum_confidence():
    d = np.ones((1, 1, 3, 3), np.float32)
    c = np.arange(9, dtype=np.float32).reshape(d.shape) / 9
    valid = np.ones_like(d, bool); valid[0, 0, 0, 0] = False
    result = depth_pyramid_level(d, valid, c)
    assert result['depth'].shape == (1, 1, 2, 2)
    assert result['confidence'][0, 0, 0, 0] == c[0, 0, 0, 1]
    assert result['confidence'][0, 0, 1, 1] == c[0, 0, 2, 2]


def donor_reference(name):
    # Isolated namespace bypasses the frozen application's __init__ and imports
    # only the retained reference file and its relative geometry helper.
    package = '_sintergpu_frozen_vkl_reference'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / 'references/donors/vkl/src/vkl/kernels/reference')]
        sys.modules[package] = module
    return getattr(importlib.import_module(package + '.' + name), name)


def test_independent_projection_against_retained_torch():
    torch = pytest.importorskip('torch')
    args = projection_case()
    expected = donor_reference('project_depth_evidence')(*(torch.from_numpy(x) for x in args), support_threshold=.5, conflict_threshold=.5)
    actual = project_depth_evidence(*args, support_threshold=.5, conflict_threshold=.5)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key].numpy())


def test_independent_photometric_against_retained_torch():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(8)
    rgb = rng.random((2, 3, 3, 5), dtype=np.float32)
    kwargs = dict(exposure_ev=np.array([-.5, .3], np.float32).reshape(2,1,1,1), rgb_gain=rng.random((2,3,1,1), dtype=np.float32), color_matrix=rng.random((2,3,3), dtype=np.float32), vignette=rng.random((2,3,3), dtype=np.float32), response_gamma=np.array([.5, 1.2], np.float32).reshape(2,1,1,1))
    expected = donor_reference('photometric_isp')(torch.from_numpy(rgb), **{k:torch.from_numpy(v) for k,v in kwargs.items()}, evidence=False)['rgb_observed'].numpy()
    np.testing.assert_allclose(photometric_isp(rgb, **kwargs), expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('representation', [0,1,2])
def test_independent_depth_convert_against_retained_torch(representation):
    torch = pytest.importorskip('torch')
    values = np.array([0,-1,2,np.nan,np.inf,1e-38,1e38], np.float32)
    valid = np.ones(values.shape, bool)
    confidence = np.ones_like(values)
    source = donor_reference('depth_convert')
    kwargs = dict(representation=['DEPTH','INVERSE_DEPTH','DISPARITY'][representation])
    if representation == 2:
        kwargs.update(focal_length=2,baseline=2)
    else:
        kwargs['scale'] = 2
    expected = source(values=torch.from_numpy(values),valid=torch.from_numpy(valid),confidence=torch.from_numpy(confidence),**kwargs)
    actual = depth_convert(values,valid,confidence,representation=representation,scale=2,focal_baseline=4)
    for key in actual:
        np.testing.assert_array_equal(actual[key],expected[key].numpy())


def test_independent_pyramid_against_retained_torch():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(72)
    depth = rng.uniform(.1,2,(2,1,7,9)).astype(np.float32)
    valid = rng.random(depth.shape) > .3
    confidence = rng.random(depth.shape,dtype=np.float32)
    depth[0,0,:2,:2] = 1
    valid[0,0,:2,:2] = True
    expected = donor_reference('depth_pyramid')(depth=torch.from_numpy(depth),valid=torch.from_numpy(valid),confidence=torch.from_numpy(confidence))['/2']
    actual = depth_pyramid_level(depth,valid,confidence)
    for key in actual:
        np.testing.assert_array_equal(actual[key],expected[key].numpy())


@pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS') != '1', reason='GPU execution explicitly requested only')
class TestVklGPU:
    @pytest.fixture(scope='class')
    @classmethod
    def device(cls):
        import slangpy as spy
        return spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})

    def dispatch(self, device, filename, entry, count, params, inputs, outputs):
        import slangpy as spy
        module = device.load_module(str(ROOT/'src/sintergpu/kernels/vkl'/filename))
        program = device.link_program([module], [module.entry_point(entry)])
        kernel = device.create_compute_kernel(program)
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        variables = {'params': params}
        buffers = {}
        for name, array in {**inputs, **outputs}.items():
            data = np.ascontiguousarray(array).reshape(-1)
            buffers[name] = device.create_buffer(element_count=data.size, struct_size=4, usage=usage, data=data)
            variables[name] = buffers[name]
        kernel.dispatch(thread_count=[count, 1, 1], vars=variables)
        return {name: buffers[name].to_numpy().view(array.dtype).reshape(array.shape) for name, array in outputs.items()}

    @pytest.mark.parametrize('randomized', [False, True])
    @pytest.mark.parametrize('threshold', [.5, 1.])
    def test_projection(self, device, randomized, threshold):
        args = list(projection_case())
        if randomized:
            rng = np.random.default_rng(31)
            args[0] = rng.normal(size=args[0].shape).astype(np.float32)
            args[0][...,2] = rng.uniform(.1, 3, args[0].shape[:2])
            args[1:4] = [rng.random(args[1].shape) > .3 for _ in range(3)]
            angle = np.float32(.17)
            args[4][0,:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
            args[4][0,0,3] = .2
            args[5][:,0,2] = 1.5
            args[5][:,1,2] = 1
            args[6] = rng.uniform(.1, 3, args[6].shape).astype(np.float32)
        expected = project_depth_evidence(*args, support_threshold=threshold, conflict_threshold=threshold)
        names = dict(eligible='eligibleOut', projected_valid='projectedOut', sensor_valid='sensorOut', support='supportOut', conflict='conflictOut', depth_residual='residualOut')
        count = args[0].shape[0]*args[0].shape[1]
        outputs = {gpu:np.full(count+7, 93, np.float32 if cpu=='depth_residual' else np.uint32) for cpu,gpu in names.items()}
        inputs = dict(zip(('xyz','visible','provisional','depthCandidate','worldToCamera','intrinsics','sensorDepth'), [args[0], *(mask_u32(m) for m in args[1:4]), *args[4:]]))
        actual = self.dispatch(device, 'project_depth_evidence.slang', 'projectDepthEvidence', count,
            dict(batches=2,points=259,height=3,width=4,supportThreshold=threshold,conflictThreshold=threshold,epsilon=float(np.finfo(np.float32).eps)), inputs, outputs)
        for cpu,gpu in names.items():
            assert np.all(actual[gpu][count:] == 93), 'padding must stay untouched'
            if cpu == 'depth_residual':
                np.testing.assert_allclose(actual[gpu][:count], expected[cpu].reshape(-1), atol=3e-7, rtol=2e-6)
            else:
                np.testing.assert_array_equal(actual[gpu][:count], expected[cpu].reshape(-1).astype(np.uint32))

    @pytest.mark.parametrize('shape', [(2,3,3,47), (1,3,1,1)])
    @pytest.mark.parametrize('enabled', [False, True])
    def test_photometric(self, device, shape, enabled):
        rng = np.random.default_rng(55)
        b,_,h,w = shape
        rgb = rng.random(shape, dtype=np.float32)
        rgb.reshape(-1)[0] = 0
        kwargs = dict(exposure_ev=rng.uniform(-1,1,b).astype(np.float32), rgb_gain=rng.uniform(.2,2,(b,3)).astype(np.float32), vignette=rng.uniform(0,.1,(b,3,3)).astype(np.float32), color_matrix=rng.uniform(-.5,1,(b,3,3)).astype(np.float32), response_gamma=rng.uniform(.3,1.5,b).astype(np.float32))
        expected = photometric_isp(rgb, **(kwargs if enabled else {}))
        inputs = dict(rgbLinear=rgb, exposureEV=kwargs['exposure_ev'], rgbGain=kwargs['rgb_gain'], vignette=kwargs['vignette'], colorMatrix=kwargs['color_matrix'], responseGamma=kwargs['response_gamma'])
        outputs = {'rgbObserved': np.full(rgb.size+7, -9, np.float32)}
        params = dict(batches=b,height=h,width=w,epsilon=float(np.finfo(np.float32).eps), **{name:int(enabled) for name in ('useExposure','useGain','useVignette','useColor','useGamma')})
        actual = self.dispatch(device, 'photometric_isp.slang', 'photometricISP', b*h*w, params, inputs, outputs)['rgbObserved']
        assert np.all(actual[rgb.size:] == -9)
        np.testing.assert_allclose(actual[:rgb.size].reshape(shape), expected, atol=3e-7, rtol=2e-5)

    @pytest.mark.parametrize('representation', [0,1,2])
    @pytest.mark.parametrize('extreme', [False, pytest.param(True, marks=pytest.mark.xfail(strict=True, reason='SlangPy 0.43.1 default denormal mode flushes extreme float32 conversions; qualification pending'))])
    def test_depth_convert(self, device, representation, extreme):
        small, large = (1e-38, 1e38) if extreme else (1e-20, 1e20)
        values = np.tile(np.array([0,-1,2,np.nan,np.inf,small,large], np.float32), 39)
        valid = np.arange(values.size)%3 != 0
        confidence = np.random.default_rng(9).random(values.shape, dtype=np.float32)
        expected = depth_convert(values, valid, confidence, representation=representation, scale=2, focal_baseline=4)
        names = dict(depth='depthOut',inverse_depth='inverseOut',valid='validOut',confidence='confidenceOut')
        outputs = {gpu:np.full(values.size+7, 73, np.uint32 if cpu=='valid' else np.float32) for cpu,gpu in names.items()}
        actual = self.dispatch(device, 'depth_convert.slang', 'depthConvert', values.size,
            dict(count=values.size,representation=representation,scale=2.,focalBaseline=4.),
            dict(values=values,valid=mask_u32(valid),confidence=confidence), outputs)
        for cpu,gpu in names.items():
            assert np.all(actual[gpu][values.size:] == 73)
            np.testing.assert_allclose(actual[gpu][:values.size], expected[cpu], atol=0, rtol=2e-6)

    def test_pyramid(self, device):
        rng = np.random.default_rng(7)
        depth = rng.uniform(.1,2,(2,1,17,31)).astype(np.float32)
        depth[0,0,:2,:2] = 1
        depth[0,0,3,3] = np.nan
        valid = rng.random(depth.shape) > .2
        valid[0,0,:2,:2] = True
        confidence = rng.random(depth.shape, dtype=np.float32)
        expected = depth_pyramid_level(depth,valid,confidence)
        count = expected['depth'].size
        names = dict(depth='depthOut',valid='validOut',confidence='confidenceOut')
        outputs = {gpu:np.full(count+7, 73, np.uint32 if cpu=='valid' else np.float32) for cpu,gpu in names.items()}
        actual = self.dispatch(device,'depth_pyramid.slang','depthPyramidLevel',count,
            dict(batches=2,height=17,width=31,epsilon=float(np.finfo(np.float32).eps)),
            dict(depth=depth,valid=mask_u32(valid),confidence=confidence),outputs)
        for cpu,gpu in names.items():
            assert np.all(actual[gpu][count:] == 73)
            np.testing.assert_allclose(actual[gpu][:count], expected[cpu].reshape(-1), atol=1e-7, rtol=2e-6)
