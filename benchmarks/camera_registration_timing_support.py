# SPDX-License-Identifier: Apache-2.0
"""Independent per-dispatch camera qualification and timing support.

Prefixes and resets are outside the timed intervals; readback and validation are
explicit qualification boundaries. This helper does not change the production API.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.camera_registration_gpu_fixture import DTYPES
from benchmarks.reprojection_support import runtime_environment, array_state_identity

ROOT = Path(__file__).resolve().parents[1]


def source_identities():
    paths = {ROOT / p for p in (
        'src/sintergpu/kernels/vkl/camera_registration.slang',
        'src/sintergpu/references/camera_registration_numeric.py',
        'src/sintergpu/references/camera_registration_stage_numeric.py',
        'src/sintergpu/camera_registration.py', 'src/sintergpu/vkl.py',
        'benchmarks/camera_registration_support.py', 'benchmarks/camera_registration_gpu_fixture.py',
        'benchmarks/camera_registration_fixtures.py', 'benchmarks/camera_registration_timing_support.py',
        'benchmarks/vkl_camera_registration.py', 'benchmarks/device_snapshot.py',
        'benchmarks/reprojection_support.py', 'benchmarks/vkl_geometry_depth.py',
        'benchmarks/locus_timings.py', 'src/sintergpu/references/reprojection_stage_numeric.py',
        'src/sintergpu/timings.py', 'src/sintergpu/catalog.py',
        'benchmarks/camera_registration_fixture_archive.py')}
    paths.update((ROOT / 'src/sintergpu/kernels/common').glob('*.slang'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def stage_cases(graphcase):
    """Evaluate untimed prefix and snapshot each distinct stage boundary."""
    from sintergpu.references.camera_registration_stage_numeric import run_stage
    graph = graphcase['graph']
    resources = {name: np.full(spec['elements'] + 13, 73, DTYPES[spec['dtype']])
                 for name, spec in graph['resources'].items()}
    for name, value in graphcase['inputs'].items():
        resources[name][:value.size] = value.reshape(-1)
    for index, stage in enumerate(graph['stages']):
        bindings = {name: resources[resource] for name, resource in stage['bindings'].items()}
        initial = {resource: resources[resource].copy() for resource in set(stage['bindings'].values())}
        result = run_stage(stage['entrypoint'], stage['params'], bindings)
        written = tuple(result)
        written_resources = {stage['bindings'][name] for name in written}
        readonly = tuple(name for name, resource in stage['bindings'].items() if resource not in written_resources)
        arrays = {name: initial[resource] for name, resource in stage['bindings'].items()}
        case = dict(operation=stage['operation'], entrypoint=stage['entrypoint'], index=index,
                    params=stage['params'].copy(), role=stage['bindings'].copy(), bindings=arrays,
                    written=written, readonly=readonly,
                    initial={name: arrays[name].copy() for name in written},
                    readonly_initial={name: arrays[name].copy() for name in readonly},
                    expected={name: value.copy() for name, value in result.items()},
                    live_counts={name: graph['resources'][resource]['elements'] for name, resource in stage['bindings'].items()})
        identity = dict(operation=case['operation'], index=index, role=case['role'], params=case['params'],
                        thread_count=stage['thread_count'], graph_workload=graphcase['workload_identity'],
                        arrays={name: dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name, a in sorted(arrays.items())})
        encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        case['workload'] = dict(id=f"vkl-camera-registration-stage/{graphcase['name']}/{index}-{stage['entrypoint']}",
                                sha256=hashlib.sha256(encoded).hexdigest(), details=identity,
                                description='Prepared independent per-dispatch inputs, parameters, roles, and persistent outputs.')
        yield case


def reset_cpu_stage(case):
    seen = set()
    for name in case['written']:
        array = case['bindings'][name]
        if id(array) not in seen:
            np.copyto(array, case['initial'][name]); seen.add(id(array))


def _compare(name, actual, expected, count, *, readonly=False):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
    if readonly:
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8), err_msg=name + ' readonly')
        return
    np.testing.assert_array_equal(actual[count:].view(np.uint8), expected[count:].view(np.uint8), err_msg=name + ' tail')
    if name == 'minDepthBits':
        np.testing.assert_allclose(actual[:count].view(np.float32), expected[:count].view(np.float32), rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name)
    elif actual.dtype.kind in 'iu':
        np.testing.assert_array_equal(actual[:count], expected[:count], err_msg=name)
    else:
        np.testing.assert_allclose(actual[:count], expected[:count], rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name)


def check_stage_parity(case, actual, *, readonly_before=None):
    for name in case['written']:
        if case['entrypoint'] == 'clearRegistration' and name == 'minDepthBits':
            np.testing.assert_array_equal(actual[name], case['expected'][name], err_msg=name)
        else:
            _compare(name, actual[name], case['expected'][name], case['live_counts'][name])
    readonly = case['readonly_initial'] if readonly_before is None else readonly_before
    for name in case['readonly']:
        _compare(name, actual[name], readonly[name], case['live_counts'][name], readonly=True)


def check_prefix(case, actual):
    for name in case['readonly']:
        _compare(name, actual[name], case['readonly_initial'][name], case['live_counts'][name])


def time_cpu_stage(case, *, iterations=50, warmups=5, clock_ns=None):
    from sintergpu.references.camera_registration_stage_numeric import run_stage
    from sintergpu.timings import summarize
    if type(iterations) is not int or iterations < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('positive integer iteration/warmup counts required')
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    for _ in range(warmups):
        reset_cpu_stage(case); run_stage(case['entrypoint'], case['params'], case['bindings'])
        check_stage_parity(case, case['bindings'])
    samples = []
    for _ in range(iterations):
        reset_cpu_stage(case); start = clock()
        run_stage(case['entrypoint'], case['params'], case['bindings'])
        stop = clock(); samples.append((stop-start)/1e6)
        check_stage_parity(case, case['bindings'])
    summarize(samples)
    return samples


def _replay(prepared, encoder, stop):
    for kernel, threads, bindings, barrier in prepared._dispatches[:stop]:
        if barrier: encoder.global_barrier()
        kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)


def _read_bindings(fixture, case, names):
    return {name: fixture.buffers[case['role'][name]].to_numpy().view(case['bindings'][name].dtype).reshape(-1) for name in names}


def _atomic_min_expected(case, initial):
    """Exact atomic result from actual captured GPU prefix, without timing it."""
    from sintergpu.references.camera_registration_stage_numeric import run_stage
    bindings = {name: value.copy() for name, value in initial.items()}
    return run_stage(case['entrypoint'], case['params'], bindings)['minDepthBits']


def validate_measured_sample(fixture, case, before, actual):
    """Validate this sample before any subsequent prefix/reset or dispatch."""
    check_prefix(case, before)
    if case['entrypoint'] == 'registrationMin':
        # The GPU's float prefix is independently compared to the CPU prefix
        # above; atomics then operate on exact captured GPU bits, not a tolerance.
        np.testing.assert_array_equal(before['minDepthBits'], case['initial']['minDepthBits'], err_msg='atomic initial minimum')
        expected = _atomic_min_expected(case, before)
        np.testing.assert_array_equal(actual['minDepthBits'], expected, err_msg='atomic minimum bits')
        remaining = dict(case, written=tuple(n for n in case['written'] if n != 'minDepthBits'))
        check_stage_parity(remaining, actual, readonly_before={n: before[n] for n in case['readonly']})
    else:
        check_stage_parity(case, actual, readonly_before={n: before[n] for n in case['readonly']})
    states = fixture.read()
    for name, spec in fixture.graph['resources'].items():
        np.testing.assert_array_equal(states[name][spec['elements']:], 73, err_msg=name + ' tail')


def measure_stage(fixture, case, *, iterations, warmups):
    """Validate every isolated GPU sample before resetting its prefix."""
    import slangpy as spy
    prepared = fixture.prepared; device = prepared.device; index = case['index']
    samples = []; hashes = []
    for sample in range(warmups + iterations):
        encoder = device.create_command_encoder(); _replay(prepared, encoder, index)
        device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
        before = _read_bindings(fixture, case, case['bindings'])
        pool = device.create_query_pool(spy.QueryType.timestamp, 2)
        encoder = device.create_command_encoder()
        kernel, threads, bindings, _ = prepared._dispatches[index]
        encoder.global_barrier()
        encoder.write_timestamp(pool, 0)
        kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
        encoder.write_timestamp(pool, 1)
        device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
        actual = _read_bindings(fixture, case, case['bindings'])
        validate_measured_sample(fixture, case, before, actual)
        ticks = pool.get_results(0, 2)
        value = (int(ticks[1])-int(ticks[0])) * 1000 / int(device.info.timestamp_frequency)
        if sample >= warmups:
            samples.append(value)
            hashes.append(array_state_identity({name: before[name] for name in case['readonly']}))
    from sintergpu.timings import summarize
    summarize(samples)
    return samples, hashes


def validate_outputs(fixture, expected):
    device = fixture.prepared.device
    encoder = device.create_command_encoder(); fixture.prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    actual = fixture.outputs_and_guards(expected)
    for name, value in expected.items():
        if value.dtype.kind in 'biu': np.testing.assert_array_equal(actual[name], value, err_msg=name)
        else: np.testing.assert_allclose(actual[name], value, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name)


def stage_timing_row(case, cpu_samples, gpu_samples, environment, warmups):
    from sintergpu.catalog import get_operation
    from sintergpu.timings import dependency_identities, summarize
    if len(cpu_samples) != len(gpu_samples): raise ValueError('paired sample counts must match')
    summarize(cpu_samples); summarize(gpu_samples)
    closure = source_identities()
    op = get_operation(case['operation'])
    return dict(**dependency_identities(), operation=case['operation'], variant=op['variant'],
        source_sha256=closure['src/sintergpu/kernels/vkl/camera_registration.slang'],
        reference_sha256=closure['src/sintergpu/references/camera_registration_stage_numeric.py'],
        runner_sha256=closure['benchmarks/vkl_camera_registration.py'],
        runner_helper_sha256=closure['benchmarks/camera_registration_timing_support.py'],
        reference_source_closure=closure,
        reference_source_closure_sha256=hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest(),
        workload=case['workload'], environment=dict(environment), iterations=len(cpu_samples), warmups=warmups,
        stage_index=case['index'], stage_role=case['role'],
        metrics=dict(cpu_reference_wall_ms=list(cpu_samples), gpu_kernel_ms=list(gpu_samples)),
        correctness='Every sample written bindings, readonly bindings and padded tails checked before next replay/reset; atomic minimum is exact against captured GPU prefix bits.',
        scope='Independent NumPy stage on precomputed inputs; CPU reset and GPU prefix/barriers outside clocks. Complete graph and transfers measured separately. Measurement only; no speedup or adoption claim.')


def measure_graph(fixture, graphcase, *, iterations=50, warmups=5, reference_iterations=3):
    import os
    from benchmarks.vkl_geometry_depth import measure
    if os.getenv('SINTERGPU_GPU_TESTS') != '1': raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if any(type(n) is not int or n < 1 for n in (iterations, warmups, reference_iterations)):
        raise ValueError('positive integer iteration/warmup counts required')
    reference = graphcase['reference']; expected = reference()
    validate_outputs(fixture, expected)
    cpu_cases = list(stage_cases(graphcase))
    device = fixture.prepared.device; environment = runtime_environment(device)
    uploads = [(fixture.buffers[name], data) for name, data in fixture.initial.items()]
    readbacks = [fixture.buffers[name] for name in dict.fromkeys(fixture.graph['outputs'].values())]
    complete = measure(device, fixture.prepared.record, uploads, readbacks, iterations, warmups)
    rows = []
    for case in cpu_cases:
        cpu = time_cpu_stage(case, iterations=iterations, warmups=warmups)
        gpu, hashes = measure_stage(fixture, case, iterations=iterations, warmups=warmups)
        row = stage_timing_row(case, cpu, gpu, environment, warmups)
        row['gpu_readonly_initial_sha256_per_sample'] = hashes
        row['cpu_readonly_initial_sha256'] = array_state_identity(case['readonly_initial'])
        rows.append(row)
        validate_outputs(fixture, expected)
    cpu_complete = []
    for _ in range(reference_iterations):
        start = time.perf_counter_ns(); reference(); cpu_complete.append((time.perf_counter_ns()-start)/1e6)
    return dict(status='measurement_only', qualification='unqualified', environment=environment, graph=fixture.graph,
                complete=complete, cpu_reference_samples_ms=cpu_complete, paired_measurements=rows,
                source_sha256=source_identities(), iterations=iterations, warmups=warmups,
                scope='Per-stage paired independent CPU/GPU timings; complete CPU oracle, resident graph and transfer-inclusive timings are separate. Shared host; no performance selection claim.')
