"""Standalone fixture harness. Uploads/readbacks are intentional test boundaries."""
import importlib.metadata
import math
import platform
import struct
import time


def _f32(rows):
    return [list(struct.unpack('<4f', struct.pack('<4f', *row))) for row in rows]


def _compare(case, actual, expected, actual_state, expected_state):
    for index, (value, golden) in enumerate(zip(actual, [v for row in expected for v in row], strict=True)):
        if not (math.isnan(value) and math.isnan(golden)) and not math.isclose(value, golden, rel_tol=2e-4, abs_tol=2e-4):
            raise AssertionError(f'{case["label"]}: float[{index}]={value}, expected {golden}')
    if list(actual_state) != expected_state:
        raise AssertionError(f'{case["label"]}: state mismatch: {actual_state} != {expected_state}')


def _summary(values):
    ordered = sorted(values)
    return {f'p{p}_us': ordered[min(len(ordered)-1, math.ceil(len(ordered)*p/100)-1)] for p in (50,95,99)}


def run(operations, *, backend, device_index=0, iterations=1):
    from .fixtures.locus.cases import cases
    from .fixtures.locus.reference import execute
    from . import __version__
    if not 1 <= iterations <= 100000:
        raise ValueError('iterations must be in [1,100000]')
    requested = {op['fixture']: op for op in operations if op.get('fixture')}
    records = []
    unsupported = [op['id'] for op in operations if not op.get('fixture')]
    device, adapter = None, None
    if backend == 'vulkan':
        import numpy as np
        import slangpy as spy
        from .adapters.slangpy import SlangPyAdapter
        adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if not 0 <= device_index < len(adapters):
            raise ValueError('device index unavailable')
        device = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=adapters[device_index].luid)
        adapter = SlangPyAdapter(device, allow_experimental=True)
    for case in cases():
        if case['name'] not in requested:
            continue
        params = list(struct.unpack('<32f', struct.pack('<32f', *case['params'])))
        inputs, initial = [_f32(buf) for buf in case['inputs']], _f32(case['output'])
        expected, expected_state = execute(case['name'], params, inputs, initial, case['state'])
        if backend == 'reference':
            records.append({'operation': requested[case['name']]['id'], 'case': case['label'], 'status': 'reference-executed'})
            continue
        start = time.perf_counter_ns()
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        arrays = [np.asarray(buf, dtype=np.float32) for buf in inputs] + [np.asarray(initial, dtype=np.float32), np.asarray(case['state'], dtype=np.uint32)]
        buffers = [device.create_buffer(element_count=len(a), struct_size=16 if a.ndim == 2 else 4, usage=usage, data=a) for a in arrays]
        bindings = dict(zip(('input0','input1','input2','output0','state'), buffers))
        bindings['Parameters'] = {'p': [spy.math.float4(*params[i:i+4]) for i in range(0,32,4)]}
        kernel = adapter.kernel(requested[case['name']]['id'])
        # Warm pipeline and driver before timestamps; compilation is setup.
        kernel.dispatch(thread_count=[case['width'],case['height'],1], vars=bindings)
        device.wait()
        times, host = [], []
        queries = device.create_query_pool(spy.QueryType.timestamp, 2)
        for _ in range(iterations):
            # State resets are outside the measured resident dispatch region.
            buffers[3].copy_from_numpy(arrays[3])
            buffers[4].copy_from_numpy(arrays[4])
            device.wait()
            queries.reset()
            host_start = time.perf_counter_ns()
            kernel.dispatch(thread_count=[case['width'],case['height'],1], vars=bindings,
                            query_pool=queries, query_index_before=0, query_index_after=1)
            device.wait()
            host.append((time.perf_counter_ns()-host_start)/1000)
            values = queries.get_timestamp_results(0,2)
            times.append((values[1]-values[0])*1e6)
        actual = buffers[3].to_numpy().view(np.float32).reshape(-1)
        actual_state = buffers[4].to_numpy().view(np.uint32).reshape(-1)
        _compare(case, actual, expected, actual_state, expected_state)
        records.append({'operation': requested[case['name']]['id'], 'case': case['label'], 'status': 'passed',
                        'iterations': iterations, 'gpu_kernel': _summary(times),
                        'host_submit_complete': _summary(host),
                        'fixture_setup_dispatch_readback_us': (time.perf_counter_ns()-start)/1000,
                        'uploaded_bytes': sum(a.nbytes for a in arrays),
                        'readback_bytes': actual.nbytes+actual_state.nbytes})
    environment = {'python': platform.python_version(), 'version': __version__}
    if device is not None:
        environment.update({'device': device.info.adapter_name, 'api': device.info.api_name,
                            'vendor_id': adapters[device_index].vendor_id, 'device_id': adapters[device_index].device_id,
                            'slangpy': importlib.metadata.version('slangpy'),
                            'compiler': 'SlangPy bundled compiler; compiler binary digest recorded separately by qualification tooling'})
    return {'status': 'pending' if unsupported else 'passed', 'backend': backend,
            'scope': 'scalar reference execution only' if backend == 'reference' else 'GPU fixture parity, not full hardware/application qualification',
            'environment': environment, 'cases': records, 'unavailable_operations': unsupported,
            'tolerance': {'absolute': 2e-4, 'relative': 2e-4},
            'nvidia_qualification': 'pending'}
