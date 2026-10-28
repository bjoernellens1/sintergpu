# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only payloads for the complete D4/T0 GPU parity corpus.

Every yielded record owns snapshots of its uploaded arrays and frozen-oracle
keyword arguments. Importing or iterating this module needs NumPy only, creates
no device, and never imports Torch, SlangPy, or the frozen numerical donors.
The graph is the existing caller-owned setup plan, not a serialized substitute.
"""
from itertools import product

import numpy as np

from sintergpu.reprojection import reprojection_plan, temporal_reprojection_plan


def cameras(b):
    return np.tile(np.eye(3, dtype=np.float32), (b, 1, 1)), np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))


def _snapshot(function, flags, iteration, family, graph, inputs, module, oracle, **kwargs):
    def copy(value):
        return value.copy() if isinstance(value, np.ndarray) else value

    # Oracle-only arguments (target_rgb/target_hw/threshold) are frozen as well.
    return dict(case_id=function + '[' + ','.join(f'{key}={value}' for key, value in flags.items()) + ']',
                iteration=iteration, family=family, graph=graph,
                inputs={key: value.copy() for key, value in inputs.items()},
                oracle_module=module, oracle_function=oracle,
                oracle_kwargs={key: copy(value) for key, value in kwargs.items()},
                case_flags=flags.copy())


def _d4_cases(b, h, w, th, tw, with_target, raster, max_workgroups_x):
    flags = dict(b=b, h=h, w=w, th=th, tw=tw, with_target=with_target, raster=raster)
    rng = np.random.default_rng(51)
    k, p = cameras(b)
    tk = k.copy()
    tk[:, 0, 0] = 0.37
    sp = p.copy()
    sp[:, :3, 3] = [0.025, 0.015, 0.12]
    depth = rng.uniform(0.5, 2, (b, 1, h, w)).astype(np.float32)
    depth.reshape(-1)[:min(4, depth.size)] = [0, np.nan, np.inf, -1][:min(4, depth.size)]
    confidence = rng.random(depth.shape, dtype=np.float32)
    td = rng.uniform(0.5, 2, (b, 1, th, tw)).astype(np.float32)
    inputs = dict(source_depth=depth, source_intrinsics=k, source_pose_c2w=sp, target_intrinsics=tk, target_pose_c2w=p)
    if with_target:
        inputs['target_depth'] = td
    if raster:
        inputs['source_confidence'] = confidence
    graph = reprojection_plan(b, h, w, th, tw, max_workgroups_x=max_workgroups_x, target_depth=with_target, raster=raster)
    for iteration in range(2):
        if iteration:
            depth.fill(0)
            depth.reshape(-1)[-1] = 1
        kwargs = dict(inputs, target_hw=(th, tw))
        if raster:
            kwargs.pop('target_depth', None)
        else:
            kwargs['target_depth'] = td if with_target else None
        yield _snapshot('test_d4_full_outputs_and_reused_scratch', flags, iteration, 'raster' if raster else 'projection', graph, inputs,
                        'reproject', 'rasterize_reprojection' if raster else 'reproject_rgbd', **kwargs)


def _temporal_cases(b, n, h, w, depth_option, confidence_option, reference, max_workgroups_x):
    flags = dict(b=b, n=n, h=h, w=w, depth_option=depth_option, confidence_option=confidence_option, reference=reference)
    rng = np.random.default_rng(8)
    k, p = cameras(b)
    sk = np.repeat(k[:, None], n, axis=1)
    sp = np.repeat(p[:, None], n, axis=1)
    sp[..., :3, 3] = [0.012, -0.007, 0.021]
    angle = np.float32(0.017)
    sp[..., :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    k[:, 0, 0] = 4.3
    k[:, 1, 1] = 3.7
    k[:, 0, 2] = 0.31
    k[:, 1, 2] = 0.17
    sk[..., 0, 0] = 4.1
    sk[..., 1, 1] = 3.9
    sk[..., 0, 2] = 0.27
    sk[..., 1, 2] = 0.21
    td = rng.uniform(0.9, 1.1, (b, 1, h, w)).astype(np.float32)
    sd = rng.uniform(0.9, 1.1, (b, n, 1, h, w)).astype(np.float32)
    rgb = rng.random((b, n, 3, h, w), dtype=np.float32)
    conf = rng.random(sd.shape, dtype=np.float32)
    td.reshape(-1)[0] = 0
    inputs = dict(target_depth=td, source_rgb=rgb, target_intrinsics=k, target_pose_c2w=p, source_intrinsics=sk, source_pose_c2w=sp)
    if depth_option or reference:
        inputs['source_depth'] = sd
    if confidence_option:
        inputs['source_confidence'] = conf
    graph = temporal_reprojection_plan(b, n, h, w, max_workgroups_x=max_workgroups_x, source_depth=depth_option, source_confidence=confidence_option, reference=reference)
    for iteration in range(2):
        if iteration:
            td.fill(0)
            if depth_option or reference:
                sd.fill(0)
                sd.reshape(-1)[-1] = 1
        kwargs = dict(inputs)
        if not reference:
            kwargs['target_rgb'] = np.zeros((b, 3, h, w), np.float32)
        yield _snapshot('test_t0_full_outputs_and_source_major_packing', flags, iteration, 'reference' if reference else 'temporal', graph, inputs,
                        'temporal_reproject', 'temporal_rgbd_reference' if reference else 'temporal_reproject_rgb', **kwargs)


def _literal_projection_case(case, max_workgroups_x):
    flags = dict(case=case)
    k, p = cameras(1)
    sk = k.copy()
    sp = p.copy()
    tk = k.copy()
    eps = np.finfo(np.float32).eps
    depth = np.ones((1, 1, 1, 1), np.float32)
    th, tw, raster, target = (1, 4, False, None)
    if case == 'normalized_half_tie':
        sp[:, 0, 3] = 0.5
        target = np.array([[[[11, 22, 33, 44]]]], np.float32)
    elif case == 'epsilon_support':
        depth = np.array([[[[1, 1 + eps, 1 + 2 * eps]]]], np.float32)
        tk[:, 0, 0] = 0.01
        raster = True
    elif case == 'epsilon_projection':
        sp[:, 0, 3] = eps
        sp[:, 2, 3] = np.float32(eps / 2) - np.float32(1)
    elif case == 'huge_finite_grid':
        sp[:, 0, 3] = 1e+30
    elif case == 'huge_grid':
        sp[:, 0, 3] = 1e+38
        sp[:, 2, 3] = -1
    elif case == 'nan_grid':
        depth = np.full((1, 1, 1, 3), 3e+38, np.float32)
        sk[:, 0, 0] = 0.1
    else:
        depth = np.array([[[[0, np.nan, np.inf, -1]]]], np.float32)
        sp[:, :3, 3] = [2, 3, 4]
    if case in ('huge_finite_grid', 'huge_grid', 'nan_grid'):
        target = np.ones((1, 1, th, tw), np.float32)
    h, w = depth.shape[-2:]
    inputs = dict(source_depth=depth, source_intrinsics=sk, source_pose_c2w=sp, target_intrinsics=tk, target_pose_c2w=p)
    if target is not None:
        inputs['target_depth'] = target
    if raster:
        confidence = np.array([[[[-0.0, 0.8, 1]]]], np.float32)
        inputs['source_confidence'] = confidence
    graph = reprojection_plan(1, h, w, th, tw, max_workgroups_x=max_workgroups_x, raster=raster, target_depth=target is not None)
    kwargs = dict(inputs, target_hw=(th, tw))
    if not raster:
        kwargs['target_depth'] = target
    yield _snapshot('test_literal_sampler_raster_and_raw_diagnostics', flags, 0, 'raster' if raster else 'projection', graph, inputs,
                    'reproject', 'rasterize_reprojection' if raster else 'reproject_rgbd', **kwargs)


def _literal_reference_case(case, max_workgroups_x):
    flags = dict(case=case)
    k, p = cameras(1)
    inputs = dict(target_depth=np.ones((1, 1, 1, 1), np.float32),
                  source_rgb=np.ones((1, 3, 3, 1, 1), np.float32),
                  source_depth=np.array([1, 1, 1.2], np.float32).reshape(1, 3, 1, 1, 1),
                  source_confidence=np.ones((1, 3, 1, 1, 1), np.float32),
                  target_intrinsics=k, target_pose_c2w=p,
                  source_intrinsics=np.repeat(k[:, None], 3, axis=1),
                  source_pose_c2w=np.repeat(p[:, None], 3, axis=1))
    inputs['source_rgb'][0, :2] = 0
    if case == 'final_invalid_diagnostics':
        inputs['target_depth'].fill(0)
    if case == 'zero_confidence':
        inputs['source_confidence'].fill(0)
    graph = temporal_reprojection_plan(1, 3, 1, 1, max_workgroups_x=max_workgroups_x, reference=True, source_confidence=True, consistency_threshold=0.1)
    yield _snapshot('test_temporal_literal_cluster_rgb_and_final_mask', flags, 0, 'reference', graph, inputs, 'temporal_reproject',
                    'temporal_rgbd_reference', **inputs, consistency_threshold=.1)


def _extreme_temporal_case(case, max_workgroups_x):
    flags = dict(case=case)
    k, p = cameras(1)
    target_pose = p.copy()
    depth = np.ones((1, 1, 1, 3), np.float32)
    if case == 'huge_finite_grid':
        target_pose[:, 0, 3] = 1e+30
    elif case == 'huge_grid':
        target_pose[:, 0, 3] = 1e+38
        target_pose[:, 2, 3] = -1
    else:
        depth.fill(3e+38)
        k[:, 0, 0] = 0.1
    inputs = dict(target_depth=depth,
                  source_rgb=np.arange(9, dtype=np.float32).reshape(1, 1, 3, 1, 3),
                  source_depth=np.ones((1, 1, 1, 1, 3), np.float32),
                  source_confidence=np.ones((1, 1, 1, 1, 3), np.float32),
                  target_intrinsics=k, target_pose_c2w=target_pose,
                  source_intrinsics=np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
                  source_pose_c2w=p[:, None])
    target_rgb = np.zeros((1, 3, 1, 3), np.float32)
    graph = temporal_reprojection_plan(1, 1, 1, 3, max_workgroups_x=max_workgroups_x, source_depth=True, source_confidence=True)
    yield _snapshot('test_temporal_extreme_grid_reaches_bilinear_and_nearest_samplers', flags, 0, 'temporal', graph, inputs, 'temporal_reproject',
                    'temporal_reproject_rgb', **inputs, target_rgb=target_rgb)


def payloads(max_workgroups_x=65535):
    """Yield all 113 iteration snapshots belonging to the original 63 cases.

    ``case_flags`` contains exactly the original pytest parameter names/values.
    Case IDs combine the original test function and those ordered parameters.
    Arrays are copied at each yield, including the second reused-scratch state.
    """
    for options, shape in product(
            [(False, False), (True, False), (False, True), (True, True)],
            [(1, 3, 5, 5, 7), (8, 17, 19, 13, 23), (1, 1, 1, 1, 1),
             (1, 1, 7, 1, 5), (1, 7, 1, 5, 1)]):
        yield from _d4_cases(*shape, *options, max_workgroups_x)
    for options, shape in product(
            [(False, False, False), (True, False, False), (False, True, False),
             (True, True, False), (True, True, True), (False, False, True)],
            [(1, 1, 3, 5), (8, 3, 17, 19), (1, 2, 1, 1), (1, 2, 1, 7), (1, 2, 7, 1)]):
        yield from _temporal_cases(*shape, *options, max_workgroups_x)
    for case in ('normalized_half_tie', 'epsilon_support', 'epsilon_projection',
                 'invalid_translated', 'huge_finite_grid', 'huge_grid', 'nan_grid'):
        yield from _literal_projection_case(case, max_workgroups_x)
    for case in ('outside_depth_cluster_rgb', 'final_invalid_diagnostics', 'zero_confidence'):
        yield from _literal_reference_case(case, max_workgroups_x)
    for case in ('huge_finite_grid', 'huge_grid', 'nan_grid'):
        yield from _extreme_temporal_case(case, max_workgroups_x)
