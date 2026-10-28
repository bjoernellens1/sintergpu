"""Opt-in full-output D4/T0 GPU parity. Never creates a device at collection."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS') != '1', reason='explicit GPU qualification only')


@pytest.fixture(scope='module')
def device():
    import slangpy as spy
    return spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})


def cameras(b):
    return np.tile(np.eye(3, dtype=np.float32), (b, 1, 1)), np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))


def frozen_module(name):
    import importlib
    from pathlib import Path
    import sys
    import types
    package = '_sintergpu_reprojection_gpu_frozen'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parents[1] / 'references/donors/vkl/src/vkl/kernels/reference')]
        sys.modules[package] = module
    return importlib.import_module(package + '.' + name)


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for key, array in expected.items():
        assert actual[key].shape == array.shape
        if array.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual[key], array, err_msg=key)
        else:
            # Same staged float32 math; native GPU sqrt/exp and bounded tree
            # reductions can differ by a few ULPs from CPU scalar/reduction math.
            np.testing.assert_allclose(actual[key], array, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('b,h,w,th,tw', [(1, 3, 5, 5, 7), (8, 17, 19, 13, 23), (1, 1, 1, 1, 1), (1, 1, 7, 1, 5), (1, 7, 1, 5, 1)])
@pytest.mark.parametrize('with_target,raster', [(False, False), (True, False), (False, True), (True, True)])
def test_d4_full_outputs_and_reused_scratch(device, b, h, w, th, tw, with_target, raster):
    from benchmarks.reprojection_support import PreparedGraph
    from sintergpu.reprojection import reprojection_plan
    from sintergpu.references.reprojection_numeric import reproject_rgbd, rasterize_reprojection
    rng = np.random.default_rng(51)
    k, p = cameras(b); tk = k.copy(); tk[:, 0, 0] = .37
    sp = p.copy(); sp[:, :3, 3] = [.025, .015, .12]
    depth = rng.uniform(.5, 2, (b, 1, h, w)).astype(np.float32)
    depth.reshape(-1)[:min(4, depth.size)] = [0, np.nan, np.inf, -1][:min(4, depth.size)]
    confidence = rng.random(depth.shape, dtype=np.float32)
    td = rng.uniform(.5, 2, (b, 1, th, tw)).astype(np.float32)
    inputs = dict(source_depth=depth, source_intrinsics=k, source_pose_c2w=sp, target_intrinsics=tk, target_pose_c2w=p)
    if with_target: inputs['target_depth'] = td
    if raster: inputs['source_confidence'] = confidence
    graph = reprojection_plan(b, h, w, th, tw, max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]), target_depth=with_target, raster=raster)
    prepared = PreparedGraph(device, graph, inputs)
    for iteration in range(2):
        if iteration:
            depth.fill(0)  # clearRaster must remove every previous winner/support.
            depth.reshape(-1)[-1] = 1
            prepared.upload('source_depth', depth)
        expected = reproject_rgbd(depth, k, sp, tk, p, td if with_target else None, target_hw=(th, tw))
        if raster:
            expected = rasterize_reprojection(depth, confidence, k, sp, tk, p, target_hw=(th, tw))
        encoder = device.create_command_encoder()
        prepared.record(encoder)
        device.submit_command_buffer(encoder.finish())
        device.wait_for_idle()
        actual = prepared.outputs(expected)
        compare(actual, expected)
        torch = pytest.importorskip('torch')
        args = [torch.from_numpy(a) for a in [depth, k, sp, tk, p]]
        if raster:
            donor = frozen_module('reproject').rasterize_reprojection(args[0], torch.from_numpy(confidence), *args[1:], target_hw=(th, tw))
        else:
            donor = frozen_module('reproject').reproject_rgbd(*args, torch.from_numpy(td) if with_target else None, target_hw=(th, tw))
        compare(actual, {key: value.numpy() for key, value in donor.items()})
        prepared.assert_poison_tails()


@pytest.mark.parametrize('b,n,h,w', [(1, 1, 3, 5), (8, 3, 17, 19), (1, 2, 1, 1), (1, 2, 1, 7), (1, 2, 7, 1)])
@pytest.mark.parametrize('depth_option,confidence_option,reference', [(False, False, False), (True, False, False), (False, True, False), (True, True, False), (True, True, True), (False, False, True)])
def test_t0_full_outputs_and_source_major_packing(device, b, n, h, w, depth_option, confidence_option, reference):
    from benchmarks.reprojection_support import PreparedGraph
    from sintergpu.reprojection import temporal_reprojection_plan
    from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb, temporal_rgbd_reference
    rng = np.random.default_rng(8)
    k, p = cameras(b)
    sk = np.repeat(k[:, None], n, axis=1)
    sp = np.repeat(p[:, None], n, axis=1)
    sp[..., :3, 3] = [.012, -.007, .021]
    angle = np.float32(.017)
    sp[..., :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    k[:, 0, 0] = 4.3; k[:, 1, 1] = 3.7
    k[:, 0, 2] = .31; k[:, 1, 2] = .17
    sk[..., 0, 0] = 4.1; sk[..., 1, 1] = 3.9
    sk[..., 0, 2] = .27; sk[..., 1, 2] = .21
    td = rng.uniform(.9, 1.1, (b, 1, h, w)).astype(np.float32)
    sd = rng.uniform(.9, 1.1, (b, n, 1, h, w)).astype(np.float32)
    rgb = rng.random((b, n, 3, h, w), dtype=np.float32)
    conf = rng.random(sd.shape, dtype=np.float32)
    td.reshape(-1)[0] = 0
    inputs = dict(target_depth=td, source_rgb=rgb, target_intrinsics=k, target_pose_c2w=p, source_intrinsics=sk, source_pose_c2w=sp)
    if depth_option or reference: inputs['source_depth'] = sd
    if confidence_option: inputs['source_confidence'] = conf
    graph = temporal_reprojection_plan(b, n, h, w, max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]), source_depth=depth_option, source_confidence=confidence_option, reference=reference)
    prepared = PreparedGraph(device, graph, inputs)
    for iteration in range(2):
        if iteration:
            td.fill(0); prepared.upload('target_depth', td)
            if depth_option or reference:
                sd.fill(0); sd.reshape(-1)[-1] = 1; prepared.upload('source_depth', sd)
        if reference:
            expected = temporal_rgbd_reference(**inputs)
        else:
            expected = temporal_reproject_rgb(target_rgb=np.zeros((b, 3, h, w), np.float32), **inputs)
        encoder = device.create_command_encoder(); prepared.record(encoder)
        device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
        actual = prepared.outputs(expected)
        compare(actual, expected)
        torch = pytest.importorskip('torch')
        donor_inputs = {key: torch.from_numpy(value) for key, value in inputs.items()}
        if reference:
            donor = frozen_module('temporal_reproject').temporal_rgbd_reference(**donor_inputs)
        else:
            donor = frozen_module('temporal_reproject').temporal_reproject_rgb(target_rgb=torch.zeros((b, 3, h, w)), **donor_inputs)
        compare(actual, {key: value.numpy() for key, value in donor.items()})
        prepared.assert_poison_tails()


@pytest.mark.parametrize('case', ['normalized_half_tie', 'epsilon_support', 'epsilon_projection', 'invalid_translated', 'huge_finite_grid', 'huge_grid', 'nan_grid'])
def test_literal_sampler_raster_and_raw_diagnostics(device, case):
    from benchmarks.reprojection_support import PreparedGraph
    from sintergpu.reprojection import reprojection_plan
    from sintergpu.references.reprojection_numeric import reproject_rgbd, rasterize_reprojection
    k, p = cameras(1); sk = k.copy(); sp = p.copy(); tk = k.copy()
    eps = np.finfo(np.float32).eps
    depth = np.ones((1, 1, 1, 1), np.float32)
    th, tw, raster, target = 1, 4, False, None
    if case == 'normalized_half_tie':
        sp[:, 0, 3] = .5
        target = np.array([[[[11, 22, 33, 44]]]], np.float32)
    elif case == 'epsilon_support':
        depth = np.array([[[[1, 1 + eps, 1 + 2 * eps]]]], np.float32)
        tk[:, 0, 0] = .01
        raster = True
    elif case == 'epsilon_projection':
        sp[:, 0, 3] = eps
        sp[:, 2, 3] = np.float32(eps / 2) - np.float32(1)
    elif case == 'huge_finite_grid':
        sp[:, 0, 3] = 1e30
    elif case == 'huge_grid':
        sp[:, 0, 3] = 1e38; sp[:, 2, 3] = -1
    elif case == 'nan_grid':
        depth = np.full((1, 1, 1, 3), 3e38, np.float32)
        sk[:, 0, 0] = .1
    else:
        depth = np.array([[[[0, np.nan, np.inf, -1]]]], np.float32)
        sp[:, :3, 3] = [2, 3, 4]
    if case in ('huge_finite_grid', 'huge_grid', 'nan_grid'):
        target = np.ones((1, 1, th, tw), np.float32)  # Execute nearestDepth guards too.
    h, w = depth.shape[-2:]
    inputs = dict(source_depth=depth, source_intrinsics=sk, source_pose_c2w=sp, target_intrinsics=tk, target_pose_c2w=p)
    if target is not None:
        inputs['target_depth'] = target
    if raster:
        confidence = np.array([[[[-0.0, .8, 1]]]], np.float32)
        inputs['source_confidence'] = confidence
        expected = rasterize_reprojection(depth, confidence, sk, sp, tk, p, target_hw=(th, tw))
    else:
        expected = reproject_rgbd(depth, sk, sp, tk, p, target, target_hw=(th, tw))
    graph = reprojection_plan(1, h, w, th, tw, max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]), raster=raster, target_depth=target is not None)
    prepared = PreparedGraph(device, graph, inputs)
    encoder = device.create_command_encoder(); prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    compare(prepared.outputs(expected), expected)
    prepared.assert_poison_tails()


@pytest.mark.parametrize('case', ['outside_depth_cluster_rgb', 'final_invalid_diagnostics', 'zero_confidence'])
def test_temporal_literal_cluster_rgb_and_final_mask(device, case):
    from benchmarks.reprojection_support import PreparedGraph
    from sintergpu.reprojection import temporal_reprojection_plan
    from sintergpu.references.temporal_reprojection_numeric import temporal_rgbd_reference
    k, p = cameras(1)
    inputs = dict(target_depth=np.ones((1, 1, 1, 1), np.float32),
                  source_rgb=np.ones((1, 3, 3, 1, 1), np.float32),
                  source_depth=np.array([1, 1, 1.2], np.float32).reshape(1, 3, 1, 1, 1),
                  source_confidence=np.ones((1, 3, 1, 1, 1), np.float32),
                  target_intrinsics=k, target_pose_c2w=p,
                  source_intrinsics=np.repeat(k[:, None], 3, axis=1),
                  source_pose_c2w=np.repeat(p[:, None], 3, axis=1))
    inputs['source_rgb'][0, :2] = 0
    if case == 'final_invalid_diagnostics': inputs['target_depth'].fill(0)
    if case == 'zero_confidence': inputs['source_confidence'].fill(0)
    expected = temporal_rgbd_reference(**inputs, consistency_threshold=.1)
    graph = temporal_reprojection_plan(1, 3, 1, 1, max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]), reference=True, source_confidence=True, consistency_threshold=.1)
    prepared = PreparedGraph(device, graph, inputs)
    encoder = device.create_command_encoder(); prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    actual = prepared.outputs(expected)
    compare(actual, expected)
    torch = pytest.importorskip('torch')
    donor = frozen_module('temporal_reproject').temporal_rgbd_reference(**{key: torch.from_numpy(value) for key, value in inputs.items()}, consistency_threshold=.1)
    compare(actual, {key: value.numpy() for key, value in donor.items()})
    prepared.assert_poison_tails()


@pytest.mark.parametrize('case', ['huge_finite_grid', 'huge_grid', 'nan_grid'])
def test_temporal_extreme_grid_reaches_bilinear_and_nearest_samplers(device, case):
    from benchmarks.reprojection_support import PreparedGraph
    from sintergpu.reprojection import temporal_reprojection_plan
    from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb
    k, p = cameras(1); target_pose = p.copy()
    depth = np.ones((1, 1, 1, 3), np.float32)
    if case == 'huge_finite_grid': target_pose[:, 0, 3] = 1e30
    elif case == 'huge_grid':
        target_pose[:, 0, 3] = 1e38; target_pose[:, 2, 3] = -1
    else:
        depth.fill(3e38); k[:, 0, 0] = .1
    inputs = dict(target_depth=depth, source_rgb=np.arange(9, dtype=np.float32).reshape(1, 1, 3, 1, 3),
                  source_depth=np.ones((1, 1, 1, 1, 3), np.float32),
                  source_confidence=np.ones((1, 1, 1, 1, 3), np.float32),
                  target_intrinsics=k, target_pose_c2w=target_pose,
                  source_intrinsics=np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
                  source_pose_c2w=p[:, None])
    target_rgb = np.zeros((1, 3, 1, 3), np.float32)
    expected = temporal_reproject_rgb(target_rgb=target_rgb, **inputs)
    graph = temporal_reprojection_plan(1, 1, 1, 3, max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]), source_depth=True, source_confidence=True)
    prepared = PreparedGraph(device, graph, inputs)
    encoder = device.create_command_encoder(); prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    actual = prepared.outputs(expected)
    compare(actual, expected)
    torch = pytest.importorskip('torch')
    donor = frozen_module('temporal_reproject').temporal_reproject_rgb(target_rgb=torch.from_numpy(target_rgb), **{key: torch.from_numpy(value) for key, value in inputs.items()})
    compare(actual, {key: value.numpy() for key, value in donor.items()})
    prepared.assert_poison_tails()
