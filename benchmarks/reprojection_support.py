# SPDX-License-Identifier: Apache-2.0
"""Test/measurement helper for graphs recorded on a caller's existing device.

All device resources, kernel compilation, binding dictionaries and uploads are
prepared before recording. record() only inserts declared storage barriers and
records dispatches on one borrowed encoder. Readback is explicit and separate.
This helper owns test buffers; it is not part of the production graph API.
"""
from pathlib import Path

import numpy as np

from sintergpu.references.reprojection_stage_numeric import run_stage

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


def measure_graph(prepared, *, graph_case, iterations=50, warmups=5, reference=None, reference_iterations=3):
    """True paired new stages plus separately scoped complete/transfer costs."""
    import os
    import time
    if os.getenv('SINTERGPU_GPU_TESTS') != '1':
        raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if type(iterations) is not int or iterations < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('positive integer iteration/warmup counts required')
    if not callable(reference) or type(reference_iterations) is not int or reference_iterations < 1:
        raise ValueError('independent NumPy reference callable and positive reference_iterations required')
    from benchmarks.vkl_geometry_depth import measure
    device = prepared.device
    uploads = [(prepared.buffers[name], value) for name, value in prepared.uploads.items()]
    readbacks = [prepared.buffers[name] for name in dict.fromkeys(prepared.graph['outputs'].values())]
    expected = reference()
    validate_outputs(prepared, expected)
    cpu_cases = {case['index']: case for case in stage_cases(graph_case)}
    environment = runtime_environment(device)
    complete = measure(device, prepared.record, uploads, readbacks, iterations, warmups)
    stages, paired = [], []
    for index, specification in enumerate(prepared.graph['stages']):
        case = cpu_cases.get(index)
        if case is not None:
            cpu_samples = time_cpu_stage(case, iterations=iterations, warmups=warmups)
            # Snapshot actual GPU readonly inputs after an untimed prefix, before
            # the isolated batch. Prefix float rounding need not equal CPU bits.
            prepare_gpu_prefix(prepared, index)
            readonly_before = read_stage_bindings(prepared, case, case['readonly'])
            check_prefix_inputs(case, readonly_before)
        gpu = measure_stage(prepared, index, iterations=iterations, warmups=warmups)
        if case is not None:
            # No replay/reset may occur before this all-write/readonly/tail check.
            validate_isolated_stage(prepared, specification, stage_case=case, readonly_before=readonly_before)
            row = stage_timing_row(case, cpu_samples, gpu['resident']['samples_ms'], environment=environment, warmups=warmups)
            row['gpu_readonly_initial_sha256'] = array_state_identity(readonly_before)
            row['cpu_readonly_initial_sha256'] = array_state_identity(case['readonly_initial'])
            paired.append(row)
        stages.append(dict(index=index, role=specification['bindings'], operation=specification['operation'],
                           cpu_counterpart='independent per-dispatch NumPy' if case is not None else 'not timed by this recipe', measurements=gpu))
        validate_outputs(prepared, expected)
    reference_samples = []
    for _ in range(reference_iterations):
        start = time.perf_counter_ns(); reference()
        reference_samples.append((time.perf_counter_ns() - start) / 1e6)
    return dict(status='measurement_only', qualification='unqualified', environment=environment,
                cpu_reference_samples_ms=reference_samples, cpu_reference_iterations=reference_iterations, pipeline_iterations=iterations, warmups=warmups,
                cpu_reference_scope='complete independent NumPy public-function oracle; separate from paired per-dispatch CPU samples',
                post_sample_validation='all new stage writes, actual GPU readonly inputs and poison tails checked immediately after each isolated batch, before any replay/reset; complete parity replay follows',
                graph=prepared.graph, complete=complete, stages=stages, paired_measurements=paired,
                source_sha256=source_identities(prepared.source_paths.values()),
                scope='New-stage CPU/GPU samples represent matching dispatch boundaries. Prefixes and resets are outside both clocks. Full pipeline/oracle and transfer costs remain separate; padding bytes are explicit. No optimized CPU, speedup or adoption claim.')


def runtime_environment(device):
    """Capture actual CPU/runtime/compiler/device identity outside timing."""
    from datetime import datetime, timezone
    import hashlib
    import importlib.metadata
    import os
    import platform
    import slangpy as spy
    from benchmarks.locus_timings import cpu_identity
    from sintergpu import __version__
    compilers = sorted(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(compilers) != 1:
        raise RuntimeError('identify exactly one actual bundled Slang compiler')
    return dict(cpu=cpu_identity(), platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                hostname=platform.node(), logical_cpus=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                captured_at_utc=datetime.now(timezone.utc).isoformat(), library_version=__version__,
                gpu=device.info.adapter_name, adapter_luid=str(getattr(device.info, 'adapter_luid', 'unavailable')), api=device.info.api_name, timestamp_frequency=int(device.info.timestamp_frequency),
                slangpy=importlib.metadata.version('slangpy'), compiler_library=compilers[0].name,
                compiler_sha256=hashlib.sha256(compilers[0].read_bytes()).hexdigest(),
                floating_point_mode='precise (configured by recipe owner)', vulkan_icd_filenames=os.getenv('VK_ICD_FILENAMES'),
                vulkan_driver_files=os.getenv('VK_DRIVER_FILES'), scalar32_padding=13)


def prepare_gpu_prefix(prepared, stop):
    """Untimed prefix only; used to capture the actual readonly starting state."""
    encoder = prepared.device.create_command_encoder()
    for kernel, threads, bindings, barrier in prepared.dispatches[:stop]:
        if barrier:
            encoder.global_barrier()
        kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
    prepared.device.submit_command_buffer(encoder.finish()); prepared.device.wait_for_idle()


def read_stage_bindings(prepared, case, names):
    return {name: prepared.buffers[case['role'][name]].to_numpy().view(case['bindings'][name].dtype).reshape(-1)
            for name in names}


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


def validate_isolated_stage(prepared, specification, expected=None, *, stage_case=None, readonly_before=None):
    """Read measured state BEFORE replay/reset: every new-stage write/input/tail."""
    if stage_case is not None:
        actual = read_stage_bindings(prepared, stage_case, stage_case['bindings'])
        check_stage_parity(stage_case, actual, readonly_before=readonly_before)
        prepared.assert_poison_tails()
        return
    # Backward-compatible narrow diagnostic entry point; real measurements use
    # the complete independent stage case above, never only this support check.
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
        'src/sintergpu/references/reprojection_stage_numeric.py',
        'src/sintergpu/reprojection.py', 'src/sintergpu/vkl.py',
        'benchmarks/reprojection_support.py', 'benchmarks/vkl_reprojection.py')}
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _prefix_stage(stage, bindings):
    """Evaluate reused geometry/D9 stages only during untimed prefix setup."""
    from sintergpu.references.geometry_depth_numeric import depth_geometry, temporal_depth_fuse
    p = stage['params']
    if stage['entrypoint'] == 'depthGeometry':
        b, h, w = p['batches'], p['height'], p['width']; count = b * h * w
        result = depth_geometry(bindings['depth'][:count].reshape(b, 1, h, w),
                                bindings['intrinsics'][:9*b].reshape(b, 3, 3),
                                bindings['pose'][:16*b].reshape(b, 4, 4) if p['useWorld'] else None,
                                bindings['confidence'][:count].reshape(b, 1, h, w) if p['useConfidence'] else None)
    elif stage['entrypoint'] == 'temporalDepthFuse':
        count, sources = p['count'], p['sources']; shape = (sources, 1, 1, 1, count)
        fields = [bindings[name][:sources*count].reshape(shape) for name in ('depths', 'valids', 'confidences', 'visibility', 'consistency', 'trusted')]
        result = temporal_depth_fuse(*fields, consistency_threshold=p['threshold'], min_support=p['minSupport'], min_temporal_support=p['minTemporalSupport'])
    else:
        raise ValueError('no independent prefix oracle for ' + stage['entrypoint'])
    for name, value in result.items():
        bindings[name + 'Out'][:value.size] = value.reshape(-1)


def stage_cases(graphcase):
    """CPU-only preparation of each NEW dispatch's true input/inout boundary.

    Prefixes and persistent snapshots are prepared here, never in a CPU timing
    interval. Repeated stages retain their graph index/role and distinct state.
    """
    import hashlib
    import json
    graph = graphcase['graph']
    resources = {name: np.full(spec['elements'] + 13, 73, DTYPES[spec['dtype']]) for name, spec in graph['resources'].items()}
    for name, value in graphcase['inputs'].items():
        resources[name][:value.size] = value.reshape(-1)
    for index, stage in enumerate(graph['stages']):
        bindings = {name: resources[resource] for name, resource in stage['bindings'].items()}
        if not stage['operation'].startswith('vkl.reprojection.'):
            _prefix_stage(stage, bindings)
            continue
        initial_resources = {resource: resources[resource].copy() for resource in set(stage['bindings'].values())}
        result = run_stage(stage['entrypoint'], stage['params'], bindings)
        arrays = {name: initial_resources[resource] for name, resource in stage['bindings'].items()}
        written = tuple(result)
        readonly = tuple(name for name in bindings if name not in written)
        case = dict(operation=stage['operation'], entrypoint=stage['entrypoint'], index=index,
                    graph_name=graphcase['name'], params=stage['params'].copy(), thread_count=stage['thread_count'],
                    role=stage['bindings'].copy(), bindings=arrays, written=written, readonly=readonly,
                    initial={name: arrays[name].copy() for name in written},
                    readonly_initial={name: arrays[name].copy() for name in readonly},
                    expected={name: value.copy() for name, value in result.items()},
                    live_counts={name: graph['resources'][resource]['elements'] for name, resource in stage['bindings'].items()},
                    graph_workload=graphcase['workload_identity'],
                    shader_paths=tuple(ROOT / 'src/sintergpu/kernels/vkl' / (op['operation'].split('.')[1] + '.slang') for op in graph['stages']))
        identity = dict(operation=case['operation'], index=index, role=case['role'], params=case['params'],
                        thread_count=case['thread_count'], written=written, graph_workload=case['graph_workload'],
                        arrays={name: dict(dtype=array.dtype.str, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest()) for name, array in sorted(arrays.items())})
        case['workload'] = dict(id=f"vkl-reprojection-stage/{graphcase['name']}/{index}-{stage['entrypoint']}",
                                description='Independent prepared NumPy dispatch boundary; exact prefix arrays/inout seeds and aliases, graph role and arguments are identity-bound.',
                                sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), details=identity)
        yield case


def reset_cpu_stage(case):
    """Reset only persistent written/inout storage, outside the CPU clock."""
    copied = set()
    for name in case['written']:
        array = case['bindings'][name]
        if id(array) not in copied:
            np.copyto(array, case['initial'][name]); copied.add(id(array))


def check_stage_parity(case, actual, *, readonly_before=None):
    """Every live write, readonly input and capacity tail; no replay/reset here."""
    readonly = case['readonly_initial'] if readonly_before is None else readonly_before
    for name in case['written']:
        observed, expected = actual[name], case['expected'][name]; count = case['live_counts'][name]
        assert observed.dtype == expected.dtype and observed.shape == expected.shape, name
        np.testing.assert_array_equal(observed[count:].view(np.uint32), expected[count:].view(np.uint32), err_msg=name + ' tail')
        if name in ('minDepthBits', 'maxConfidenceBits'):
            np.testing.assert_allclose(observed[:count].view(np.float32), expected[:count].view(np.float32), rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name)
        elif observed.dtype.kind in 'iu':
            np.testing.assert_array_equal(observed[:count], expected[:count], err_msg=name)
        else:
            np.testing.assert_allclose(observed[:count], expected[:count], rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name)
    for name in case['readonly']:
        np.testing.assert_array_equal(actual[name].view(np.uint32), readonly[name].view(np.uint32), err_msg=name + ' readonly')


def time_cpu_stage(case, *, iterations=50, warmups=5, clock_ns=None):
    """Time ONLY run_stage; allocation, prefix, resets and validation are outside."""
    import time
    if type(iterations) is not int or iterations < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('positive iteration/warmup counts required')
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    for _ in range(warmups):
        reset_cpu_stage(case); run_stage(case['entrypoint'], case['params'], case['bindings'])
        check_stage_parity(case, case['bindings'])
    samples = []
    for _ in range(iterations):
        reset_cpu_stage(case)
        start = clock()
        run_stage(case['entrypoint'], case['params'], case['bindings'])
        stop = clock()
        samples.append((stop - start) / 1e6)
        check_stage_parity(case, case['bindings'])
    from sintergpu.timings import summarize
    summarize(samples)
    return samples


def stage_timing_row(case, cpu_samples, gpu_samples, *, environment, warmups):
    """Registry-compatible row for one real paired new dispatch workload."""
    import hashlib
    from sintergpu.catalog import get_operation
    from sintergpu.timings import dependency_identities, summarize
    if len(cpu_samples) != len(gpu_samples):
        raise ValueError('paired CPU/GPU sample counts must match')
    summarize(cpu_samples); summarize(gpu_samples)
    operation = get_operation(case['operation'])
    shader = ROOT / 'src/sintergpu' / operation['path']
    oracle = ROOT / 'src/sintergpu/references/reprojection_stage_numeric.py'
    closure = source_identities(case['shader_paths'])
    import json
    closure_hash = hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()
    return dict(**dependency_identities(), operation=case['operation'], variant=operation['variant'],
                source_sha256=hashlib.sha256(shader.read_bytes()).hexdigest(), reference_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256((ROOT / 'benchmarks/vkl_reprojection.py').read_bytes()).hexdigest(),
                runner_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_source_closure=closure, reference_source_closure_sha256=closure_hash, workload=case['workload'], environment=dict(environment),
                metrics=dict(cpu_reference_wall_ms=list(cpu_samples), gpu_kernel_ms=list(gpu_samples)),
                iterations=len(cpu_samples), warmups=warmups, stage_index=case['index'], stage_role=case['role'],
                correctness='Every isolated stage live write, readonly input and poison tail validated before any replay/reset; integer logical/counts exact, encoded positive floats compared numerically.',
                scope='Independent NumPy per-dispatch CPU reference on precomputed flat inputs and preallocated outputs; CPU clock includes only selected run_stage work and its scalar/sampler temporaries. Prefix evaluation, persistent resets, validation and all copies are outside CPU clock. GPU timestamps include only selected kernel dispatch; prefixes, declared visibility barriers and resets are outside the interval. Complete pipeline/oracle and transfer-inclusive costs are separate. Not an optimized CPU alternative, speedup or adoption qualification.')


def array_state_identity(arrays):
    """Exact actual/expected input-state hash, separate from numerical parity."""
    import hashlib
    import json
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        h.update(json.dumps([name, array.dtype.str, array.shape]).encode() + b'\0' + array.tobytes() + b'\0')
    return h.hexdigest()


def check_prefix_inputs(case, actual):
    """Independent CPU and GPU prefixes agree before pairing their stage costs."""
    for name in case['readonly']:
        observed, expected = actual[name], case['readonly_initial'][name]; count = case['live_counts'][name]
        assert observed.dtype == expected.dtype and observed.shape == expected.shape, name
        np.testing.assert_array_equal(observed[count:].view(np.uint32), expected[count:].view(np.uint32), err_msg=name + ' prefix tail')
        if name in ('minDepthBits', 'maxConfidenceBits'):
            np.testing.assert_allclose(observed[:count].view(np.float32), expected[:count].view(np.float32), rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name + ' prefix')
        elif observed.dtype.kind in 'iu':
            np.testing.assert_array_equal(observed[:count], expected[:count], err_msg=name + ' prefix')
        else:
            np.testing.assert_allclose(observed[:count], expected[:count], rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=name + ' prefix')
