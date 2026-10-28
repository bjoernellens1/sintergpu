# SPDX-License-Identifier: Apache-2.0
"""Test/measurement helper for graphs recorded on a caller's existing device.

All device resources, kernel compilation, binding dictionaries and uploads are
prepared before recording. record() only inserts declared storage barriers and
records dispatches on one borrowed encoder. Readback is explicit and separate.
This helper owns test buffers; it is not part of the production graph API.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {'float32': np.float32, 'uint32': np.uint32, 'int32': np.int32}


class PreparedGraph:
    def __init__(self, device, graph, inputs, *, padding=13):
        import slangpy as spy
        self.device = device
        self.graph = graph
        self.padding = padding
        self.buffers = {}
        self.uploads = {}
        self.dispatches = []
        self.source_paths = {}
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        for name, resource in graph['resources'].items():
            dtype = DTYPES[resource['dtype']]
            data = np.full(resource['elements'] + padding, 73, dtype=dtype)
            if resource['role'] == 'input':
                if name not in inputs:
                    raise ValueError('missing graph input: ' + name)
                value = np.asarray(inputs[name])
                if value.dtype != dtype or value.size != resource['elements']:
                    raise ValueError('graph input dtype/capacity mismatch: ' + name)
                data[:value.size] = value.reshape(-1)
                self.uploads[name] = data
            self.buffers[name] = device.create_buffer(element_count=data.size, struct_size=4, usage=usage, data=data)
        kernels = {}
        modules = {}
        for stage in graph['stages']:
            operation = stage['operation']
            if operation not in kernels:
                module_name = operation.split('.')[1]
                path = ROOT / 'src/sintergpu/kernels/vkl' / (module_name + '.slang')
                self.source_paths[operation] = path
                if module_name not in modules:
                    modules[module_name] = device.load_module(str(path))
                module = modules[module_name]
                kernels[operation] = device.create_compute_kernel(device.link_program([module], [module.entry_point(stage['entrypoint'])]))
            bindings = dict(params=stage['params'], **{binding: self.buffers[resource] for binding, resource in stage['bindings'].items()})
            self.dispatches.append((kernels[operation], list(stage['thread_count']), bindings, bool(stage['barriers'])))

    def upload(self, name, array):
        resource = self.graph['resources'][name]
        value = np.asarray(array)
        if name not in self.uploads or value.dtype != DTYPES[resource['dtype']] or value.size != resource['elements']:
            raise ValueError('graph input dtype/capacity mismatch: ' + name)
        data = self.uploads[name]
        data[:value.size] = value.reshape(-1)
        self.buffers[name].copy_from_numpy(data)

    def record(self, encoder):
        if encoder is None:
            raise ValueError('borrowed command encoder required')
        for kernel, threads, bindings, barrier in self.dispatches:
            if barrier:
                encoder.global_barrier()
            kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)

    def outputs(self, expected):
        if expected.keys() != self.graph['outputs'].keys():
            raise ValueError('expected shapes must match the complete frozen output set')
        result = {}
        for key, reference in expected.items():
            name = self.graph['outputs'][key]
            resource = self.graph['resources'][name]
            values = self.buffers[name].to_numpy().view(DTYPES[resource['dtype']])
            if reference.size != resource['elements']:
                raise ValueError('expected output shape does not match graph capacity')
            result[key] = values[:reference.size].reshape(reference.shape)
        return result

    def assert_poison_tails(self):
        for name, resource in self.graph['resources'].items():
            values = self.buffers[name].to_numpy().view(DTYPES[resource['dtype']])
            np.testing.assert_array_equal(values[resource['elements']:], 73, err_msg=name + ' poison tail overwritten')


def measure_graph(prepared, *, iterations=30, warmups=5, reference=None, reference_iterations=3):
    """Retain per-stage, complete resident and transfer-inclusive raw samples.

    Explicit opt-in only; call after parity checks and device ownership/review.
    Query/submit/wait belongs to this benchmark boundary, never product code.
    """
    import hashlib
    import importlib.metadata
    import os
    import platform
    import time
    if os.getenv('SINTERGPU_GPU_TESTS') != '1':
        raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if type(iterations) is not int or iterations < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('positive integer iteration and warmup counts required')
    if not callable(reference) or type(reference_iterations) is not int or reference_iterations < 1:
        raise ValueError('independent NumPy reference callable and positive reference_iterations required')
    from benchmarks.vkl_geometry_depth import measure
    device = prepared.device
    uploads = [(prepared.buffers[name], value) for name, value in prepared.uploads.items()]
    readbacks = [prepared.buffers[name] for name in dict.fromkeys(prepared.graph['outputs'].values())]
    expected = reference()
    validate_outputs(prepared, expected)
    # Capture one-pass atomic scratch before isolated repetitions. The support
    # postcondition is checked BEFORE any complete replay can reset corruption.
    isolated_expected = {}
    for stage in prepared.graph['stages']:
        if stage['entrypoint'] == 'rasterSupport':
            name = stage['bindings']['supportBits']
            size = prepared.graph['resources'][name]['elements']
            isolated_expected[name] = prepared.buffers[name].to_numpy().view(np.uint32)[:size].copy()
    complete = measure(device, prepared.record, uploads, readbacks, iterations, warmups)
    # Each isolated sample replays its prefix outside the timestamp interval.
    stages = []
    for index, specification in enumerate(prepared.graph['stages']):
        stages.append(dict(index=index, role=specification['bindings'], operation=specification['operation'],
                           measurements=measure_stage(prepared, index, iterations=iterations, warmups=warmups)))
        validate_isolated_stage(prepared, specification, isolated_expected)
        if expected is not None:
            validate_outputs(prepared, expected)
    reference_samples = []
    if reference is not None:
        if not callable(reference) or type(reference_iterations) is not int or reference_iterations < 1:
            raise ValueError('reference must be callable with positive reference_iterations')
        for _ in range(reference_iterations):
            start = time.perf_counter_ns(); reference()
            reference_samples.append((time.perf_counter_ns() - start) / 1e6)
    import slangpy as spy
    compiler_libraries = sorted(Path(spy.__file__).parent.glob('libslang-compiler*'))
    return dict(status='measurement_only', qualification='unqualified', device=device.info.adapter_name,
                api=device.info.api_name, python=platform.python_version(), slangpy=importlib.metadata.version('slangpy'),
                floating_point_mode='precise (caller device configuration required)', numpy=np.__version__,
                compiler_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in compiler_libraries},
                cpu_reference_samples_ms=reference_samples, cpu_reference_scope='complete independent NumPy public-function oracle; no frozen Torch or per-entry CPU timing claimed',
                post_sample_validation='support scratch checked immediately after each isolated support sample batch against fresh one-pass snapshot, before reset; complete replay/all outputs/poison after each stage batch',
                graph=prepared.graph, complete=complete, stages=stages,
                source_sha256=source_identities(prepared.source_paths.values()),
                scope='Per-stage samples replay all prerequisites outside each timestamp interval; complete samples include the full graph with explicit barriers. Test buffer padding is included in explicit transfer bytes. No comparative speedup/adoption claim.')


def measure_stage(prepared, index, *, iterations, warmups):
    """Reset/replay prerequisites before EVERY stateful stage measurement."""
    import slangpy as spy
    device = prepared.device
    def replay(encoder, stop):
        for kernel, threads, bindings, barrier in prepared.dispatches[:stop]:
            if barrier:
                encoder.global_barrier()
            kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
    for _ in range(warmups):
        encoder = device.create_command_encoder(); replay(encoder, index + 1)
        device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    pool = device.create_query_pool(spy.QueryType.timestamp, 2 * iterations)
    encoder = device.create_command_encoder()
    kernel, threads, bindings, barrier = prepared.dispatches[index]
    for sample in range(iterations):
        replay(encoder, index)
        # Make the preceding writers visible before the timed stage. A barrier
        # after reset is necessary even when this sample follows another one.
        encoder.global_barrier()
        encoder.write_timestamp(pool, 2 * sample)
        if barrier:
            encoder.global_barrier()
        kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
        encoder.write_timestamp(pool, 2 * sample + 1)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    ticks = pool.get_results(0, 2 * iterations)
    samples = [(int(ticks[2 * n + 1]) - int(ticks[2 * n])) * 1000 / int(device.info.timestamp_frequency) for n in range(iterations)]
    if not all(value > 0 for value in samples):
        raise RuntimeError('non-increasing GPU timestamps')
    return dict(resident=dict(samples_ms=samples, median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95))),
                initialization='all preceding graph stages replayed before each sample, outside timestamp interval')



def validate_outputs(prepared, expected):
    """Full replay/parity outside any timing window, including atomic resets."""
    device = prepared.device
    encoder = device.create_command_encoder(); prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    actual = prepared.outputs(expected)
    for key, value in expected.items():
        if value.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], value, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)
    prepared.assert_poison_tails()


def validate_isolated_stage(prepared, specification, expected):
    """Read the measured state BEFORE replay/reset, checking support accumulation."""
    if specification['entrypoint'] != 'rasterSupport':
        return
    name = specification['bindings']['supportBits']
    size = prepared.graph['resources'][name]['elements']
    sampled = prepared.buffers[name].to_numpy().view(np.uint32)[:size]
    np.testing.assert_array_equal(sampled, expected[name], err_msg='isolated rasterSupport must equal fresh one-pass support before any reset')


def source_identities(shader_paths):
    """Bind shaders/common closure and every host/oracle/recipe implementation."""
    import hashlib
    paths = set(shader_paths) | set((ROOT / 'src/sintergpu/kernels/common').glob('*.slang'))
    paths |= {ROOT / name for name in (
        'src/sintergpu/references/reprojection_numeric.py',
        'src/sintergpu/references/temporal_reprojection_numeric.py',
        'src/sintergpu/references/geometry_depth_numeric.py',
        'src/sintergpu/reprojection.py', 'src/sintergpu/vkl.py',
        'benchmarks/reprojection_support.py', 'benchmarks/vkl_reprojection.py')}
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
