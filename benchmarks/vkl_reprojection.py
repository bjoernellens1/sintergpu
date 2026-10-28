# SPDX-License-Identifier: Apache-2.0
"""Opt-in D4/T0 complete-family measurements with CPU-only case enumeration.

SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. python benchmarks/vkl_reprojection.py \
    --family reference --case collision --neighbors 3 --max-cases 1 \
    --iterations 30 --output build/vkl-reprojection-timings.json

Defaults are small fixtures. Full-sized dimensions require explicit --height /
--width. CPU timing measures independent NumPy public-function oracles; no
per-entry CPU implementation or per-entry CPU timing counterpart is claimed.
"""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.reprojection_support import PreparedGraph, measure_graph, validate_outputs

VARIANTS = ('ordinary', 'collision', 'sparse', 'all_invalid', 'odd_tail')
FAMILIES = ('projection', 'raster', 'temporal', 'reference')


def workload_identity(case, *, arguments=None):
    """Bind exact dtype/shape/C-order bytes, graph, algorithm and run arguments."""
    arrays = {}
    for name, array in sorted(case['inputs'].items()):
        array = np.asarray(array)
        arrays[name] = dict(dtype=array.dtype.str, shape=list(array.shape),
                            sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
    function = case['reference'].func
    payload = dict(family=case['family'], variant=case['variant'], inputs=arrays,
                   test_allocation=dict(padding_scalar32=13, poison=73),
                   graph=case['graph'], algorithm=function.__module__ + '.' + function.__name__,
                   case_arguments=case['arguments'], run_arguments=arguments or {})
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), description=payload)


def cases(*, family='all', batches=1, neighbors=None, height=None, width=None,
          target_height=None, target_width=None, source_depth=False,
          source_confidence=False, target_depth=False, variants=VARIANTS,
          max_cases=None, max_workgroups_x=65535):
    """Yield CPU-only {name,family,variant,graph,inputs,reference,workload_identity}.

    Unspecified neighbor count exercises N1 and N3 for temporal/reference;
    projection/raster use N1. Consumers may enumerate operation IDs without
    importing any GPU library or invoking an oracle.
    """
    from sintergpu.reprojection import reprojection_plan, temporal_reprojection_plan
    from sintergpu.references.reprojection_numeric import reproject_rgbd, rasterize_reprojection
    from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb, temporal_rgbd_reference
    if family not in (*FAMILIES, 'all') or not variants or any(v not in VARIANTS for v in variants):
        raise ValueError('unknown numerical family/distribution')
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('max_cases must be a positive integer')
    if neighbors is not None and (type(neighbors) is not int or neighbors < 1):
        raise ValueError('neighbors must be a positive integer')
    for name, value in (('height', height), ('width', width), ('target_height', target_height), ('target_width', target_width)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(name + ' must be a positive integer')
    counts = (1, 3) if neighbors is None else (neighbors,)
    families = FAMILIES if family == 'all' else (family,)
    emitted = 0
    for selected_family in families:
        for variant in variants:
            for n in counts if selected_family in ('temporal', 'reference') else (counts[0],):
                h = (17 if height is None else height | 1) if variant == 'odd_tail' else (3 if height is None else height)
                w = (19 if width is None else width | 1) if variant == 'odd_tail' else (5 if width is None else width)
                th = h if target_height is None else target_height
                tw = w if target_width is None else target_width
                if selected_family in ('temporal', 'reference'):
                    th, tw = h, w
                # Graph validation precedes NumPy allocation and rejects counts.
                if selected_family in ('projection', 'raster'):
                    graph = reprojection_plan(batches, h, w, th, tw, max_workgroups_x=max_workgroups_x,
                                              target_depth=target_depth, raster=selected_family == 'raster')
                else:
                    graph = temporal_reprojection_plan(batches, n, h, w, max_workgroups_x=max_workgroups_x,
                                                       source_depth=source_depth, source_confidence=source_confidence,
                                                       reference=selected_family == 'reference')
                rng = np.random.default_rng(20261028 + VARIANTS.index(variant) * 17 + n)
                sk = np.tile(np.eye(3, dtype=np.float32), (batches, 1, 1))
                tk = sk.copy()
                pose = np.tile(np.eye(4, dtype=np.float32), (batches, 1, 1))
                d = rng.uniform(.99, 1.01, (batches, 1, h, w)).astype(np.float32)
                priors = rng.uniform(.99, 1.01, (batches, n, 1, h, w)).astype(np.float32)
                confidence = rng.random(d.shape, dtype=np.float32)
                prior_confidence = rng.random(priors.shape, dtype=np.float32)
                rgb = rng.random((batches, n, 3, h, w), dtype=np.float32)
                target = rng.uniform(.99, 1.01, (batches, 1, th, tw)).astype(np.float32)
                if variant == 'collision':
                    d.fill(1); priors.fill(1)
                    if d.size > 1: d.reshape(-1)[1] += np.finfo(np.float32).eps
                    if priors.size > 1: priors.reshape(-1)[1] += np.finfo(np.float32).eps
                    tk[:, 0, 0] = .01; tk[:, 1, 1] = .01
                elif variant == 'all_invalid':
                    d.fill(0); priors.fill(0)
                elif variant == 'sparse':
                    for values in (d, priors):
                        values.reshape(-1)[np.arange(values.size) % 7 != 0] = 0
                        if values.size > 1: values.reshape(-1)[1] = np.nan
                if selected_family in ('projection', 'raster'):
                    inputs = dict(source_depth=d, source_intrinsics=sk, source_pose_c2w=pose,
                                  target_intrinsics=tk, target_pose_c2w=pose)
                    if target_depth: inputs['target_depth'] = target
                    if selected_family == 'raster':
                        inputs['source_confidence'] = confidence
                        oracle = functools.partial(rasterize_reprojection, d, confidence, sk, pose, tk, pose, target_hw=(th, tw))
                    else:
                        oracle = functools.partial(reproject_rgbd, d, sk, pose, tk, pose, target if target_depth else None, target_hw=(th, tw))
                else:
                    inputs = dict(target_depth=d, source_rgb=rgb, target_intrinsics=tk, target_pose_c2w=pose,
                                  source_intrinsics=np.repeat(sk[:, None], n, axis=1),
                                  source_pose_c2w=np.repeat(pose[:, None], n, axis=1))
                    if source_depth or selected_family == 'reference': inputs['source_depth'] = priors
                    if source_confidence: inputs['source_confidence'] = prior_confidence
                    if selected_family == 'reference':
                        oracle = functools.partial(temporal_rgbd_reference, **inputs)
                    else:
                        oracle = functools.partial(temporal_reproject_rgb, target_rgb=np.zeros((batches, 3, h, w), np.float32), **inputs)
                arguments = dict(family=family, batches=batches, neighbors=neighbors, height=height, width=width,
                                 target_height=target_height, target_width=target_width, source_depth=source_depth,
                                 source_confidence=source_confidence, target_depth=target_depth, variants=list(variants),
                                 max_cases=max_cases, max_workgroups_x=max_workgroups_x)
                case = dict(name=f'{selected_family}/{variant}/b{batches}-n{n}-{h}x{w}-to{th}x{tw}',
                            family=selected_family, variant=variant, neighbors=n,
                            graph=graph, inputs=inputs, reference=oracle, arguments=arguments)
                case['workload_identity'] = workload_identity(case)
                yield case
                emitted += 1
                if max_cases is not None and emitted >= max_cases:
                    return


build_cases = cases


def run(*, family='all', batches=1, neighbors=None, height=None, width=None,
        target_height=None, target_width=None, source_depth=False,
        source_confidence=False, target_depth=False, variants=VARIANTS,
        max_cases=None, iterations=30, warmups=5, cli_arguments=None):
    """Own exactly one Device, close it even on parity/measurement failure."""
    import os
    if type(iterations) is not int or iterations < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('positive iteration/warmup counts required')
    if os.getenv('SINTERGPU_GPU_TESTS') != '1':
        raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    arguments = dict(family=family, batches=batches, neighbors=neighbors, height=height, width=width,
                     target_height=target_height, target_width=target_width, source_depth=source_depth,
                     source_confidence=source_confidence, target_depth=target_depth, variants=list(variants),
                     max_cases=max_cases, iterations=iterations, warmups=warmups, cli_arguments=cli_arguments)
    import slangpy as spy
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    try:
        limit = int(device.info.limits.max_compute_dispatch_thread_groups[0])
        records = {}
        for case in cases(family=family, batches=batches, neighbors=neighbors, height=height, width=width,
                          target_height=target_height, target_width=target_width, source_depth=source_depth,
                          source_confidence=source_confidence, target_depth=target_depth, variants=variants,
                          max_cases=max_cases, max_workgroups_x=limit):
            prepared = PreparedGraph(device, case['graph'], case['inputs'])
            expected = case['reference']()
            validate_outputs(prepared, expected)
            record = measure_graph(prepared, iterations=iterations, warmups=warmups, reference=case['reference'])
            record['workload_identity'] = workload_identity(case, arguments=arguments)
            record['family'] = case['family']; record['variant'] = case['variant']
            records[case['name']] = record
        return dict(status='measurement_only', adoption='unqualified', arguments=arguments, measurements=records)
    finally:
        device.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['all', *FAMILIES], default='all')
    for name, default in [('batches', 1), ('neighbors', None), ('height', None), ('width', None), ('iterations', 30), ('warmups', 5), ('max-cases', None)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name in ('target-height', 'target-width'):
        parser.add_argument('--' + name, type=int)
    for name in ('source-depth', 'source-confidence', 'target-depth'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--case', choices=VARIANTS, action='append', dest='variants')
    parser.add_argument('--output', type=Path, required=True)
    arguments = vars(parser.parse_args())
    cli_arguments = {name: str(value) if isinstance(value, Path) else value for name, value in arguments.items()}
    output = arguments.pop('output')
    if arguments['variants'] is None: arguments['variants'] = VARIANTS
    result = run(**arguments, cli_arguments=cli_arguments)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], adoption=result['adoption'], measurements=list(result['measurements']))))
