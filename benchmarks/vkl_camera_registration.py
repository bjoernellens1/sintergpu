# SPDX-License-Identifier: Apache-2.0
"""Opt-in paired camera dispatch timings on canonical donor fixtures.

Pass a source-bound archive and its trusted manifest SHA256. All fixtures are
validated before creating a GPU device. Defaults: 50 samples and five warmups.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.camera_registration_fixtures import payloads
from benchmarks.camera_registration_gpu_fixture import fixture_graph, FixtureResources
from benchmarks.camera_registration_timing_support import measure_graph
from benchmarks.camera_registration_fixture_archive import payload_identity, load_all


def cases(*, case_ids=None, max_workgroups_x=65535):
    from sintergpu.references import camera_registration_numeric as reference
    selected = None if case_ids is None else set(case_ids)
    if selected is not None and (not selected or len(selected) != len(case_ids)):
        raise ValueError('case IDs must be nonempty and unique')
    seen = set()
    for payload in payloads():
        if selected is not None and payload['case_id'] not in selected: continue
        graph, inputs = fixture_graph(payload, max_workgroups_x=max_workgroups_x)
        function = getattr(reference, payload['function'])
        def call(function=function, kwargs=payload['kwargs']):
            result = function(**kwargs)
            return result if isinstance(result, dict) else {'result': result}
        case = dict(name=payload['case_id'], payload=payload, graph=graph, inputs=inputs,
                    reference=call, workload_identity=payload_identity(payload))
        seen.add(payload['case_id'])
        yield case
    if selected is not None and selected != seen:
        raise ValueError('unknown case IDs: ' + ', '.join(sorted(selected - seen)))


def canonical_records(directory, expected_sha256):
    path = Path(directory) / 'manifest.json'
    if path.is_symlink() or len(expected_sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('canonical fixture manifest digest mismatch')
    return {payload['case_id']: expected for payload, expected in load_all(directory)}


def _compare_reference(actual, donor):
    if actual.keys() != donor.keys(): raise ValueError('canonical output set mismatch')
    for key, expected in donor.items():
        value = actual[key]
        assert value.shape == expected.shape and value.dtype == expected.dtype, key
        if value.dtype.kind in 'biu': np.testing.assert_array_equal(value, expected, err_msg=key)
        else: np.testing.assert_allclose(value, expected, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)


def run(*, archive, manifest_sha256, case_ids=None, iterations=50, warmups=5):
    import os
    if os.getenv('SINTERGPU_GPU_TESTS') != '1': raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if any(type(n) is not int or n < 1 for n in (iterations, warmups)):
        raise ValueError('positive integer iteration/warmup counts required')
    # Verify complete source-bound archive, case IDs, and independent CPU oracle
    # before device initialization; no fallback to live donor execution.
    donor = canonical_records(archive, manifest_sha256)
    preflight = list(cases(case_ids=case_ids))
    for case in preflight: _compare_reference(case['reference'](), donor[case['name']])
    import slangpy as spy
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    try:
        limit = int(device.info.limits.max_compute_dispatch_thread_groups[0])
        records = {}; rows = []
        for case in cases(case_ids=case_ids, max_workgroups_x=limit):
            if case['graph']['features']: raise RuntimeError('unverified logical-device features')
            fixture = FixtureResources(device, case['graph'], case['inputs'], enabled_features=())
            record = measure_graph(fixture, case, iterations=iterations, warmups=warmups)
            rows.extend(record.pop('paired_measurements'))
            record['workload_identity'] = case['workload_identity']
            records[case['name']] = record
        return dict(schema_version=1, status='measurement_only', adoption='unqualified',
                    canonical_manifest_sha256=manifest_sha256, iterations=iterations, warmups=warmups,
                    measurements=rows, pipeline_measurements=records)
    finally:
        device.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--case', dest='case_ids', action='append')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    result = run(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], paired_stage_rows=len(result['measurements']), pipelines=list(result['pipeline_measurements']))))
