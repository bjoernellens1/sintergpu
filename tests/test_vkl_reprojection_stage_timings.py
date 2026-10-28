"""CPU-only proof of per-entry timing scope, prefix state and parity gates."""
import numpy as np


def test_runtime_rows_bind_actual_driver_snapshot_and_require_matching_host(tmp_path, monkeypatch):
    import hashlib
    import importlib.metadata
    import json
    import platform
    import sys
    from types import SimpleNamespace
    import pytest
    from benchmarks.device_snapshot import parse_snapshot
    from benchmarks.reprojection_support import runtime_environment
    package = tmp_path / 'spy'; package.mkdir()
    (package / 'libslang-compiler.so').write_bytes(b'fixture compiler identity')
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(__file__=str(package / '__init__.py')))
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'fixture-version')
    raw = 'GPU0:\n' + '\n'.join(name + ' = ' + value for name, value in {
        'apiVersion': '1.4.354', 'driverVersion': '26.2.3', 'vendorID': '0x1002',
        'deviceID': '0x1586', 'deviceName': 'fixture GPU', 'driverName': 'radv',
        'driverInfo': 'Mesa 26.2.3', 'deviceUUID': '00000000-c400-0000-0000-000000000000',
        'driverUUID': '414d442d-4d45-5341-2d44-525600000000'}.items())
    snapshot = parse_snapshot(raw, hostname=platform.node(), captured_at_utc='2026-10-28T00:00:00+00:00')
    path = tmp_path / 'device.json'; path.write_text(json.dumps(snapshot))
    device = SimpleNamespace(info=SimpleNamespace(adapter_name='fixture GPU', api_name='Vulkan', timestamp_frequency=1000))
    monkeypatch.setenv('SINTERGPU_DEVICE_SNAPSHOT', str(path))
    environment = runtime_environment(device)
    assert environment['driver'] == {'name': 'radv', 'info': 'Mesa 26.2.3', 'version': '26.2.3',
                                     'uuid': '414d442d-4d45-5341-2d44-525600000000'}
    assert environment['physical_device_uuid'] == snapshot['physical_device_uuid']
    assert environment['device_snapshot']['snapshot_file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot['hostname'] = 'different-host'; path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='hostname'): runtime_environment(device)
    monkeypatch.delenv('SINTERGPU_DEVICE_SNAPSHOT')
    with pytest.raises(RuntimeError, match='SINTERGPU_DEVICE_SNAPSHOT'): runtime_environment(device)
import pytest

from benchmarks.vkl_reprojection import cases, stage_cases
from benchmarks.reprojection_support import check_stage_parity, time_cpu_stage, stage_timing_row


def representative():
    return next(cases(family='reference', variants=('collision',), neighbors=3, source_confidence=True))


def test_prepared_stage_cases_cover_exact_nine_entries_and_real_write_sets():
    prepared = list(stage_cases(representative()))
    assert {case['entrypoint'] for case in prepared} == {'projectPoints', 'reduceProjection', 'clearRaster', 'rasterMin', 'rasterSupport', 'finalizeRaster', 'sampleTemporal', 'packTemporal', 'finalizeReference'}
    assert len([case for case in prepared if case['entrypoint'] == 'projectPoints']) == 2
    from sintergpu.references.reprojection_stage_numeric import run_stage
    for case in prepared:
        actual = run_stage(case['entrypoint'], case['params'], case['bindings'])
        assert set(actual) == set(case['written']) == set(case['expected'])
        assert all(actual[name] is case['bindings'][name] for name in actual)
        check_stage_parity(case, case['bindings'])


def test_all_stage_writes_readonly_and_tail_corruption_are_detected():
    from sintergpu.references.reprojection_stage_numeric import run_stage
    for case in stage_cases(representative()):
        run_stage(case['entrypoint'], case['params'], case['bindings'])
        for name in case['written']:
            array = case['bindings'][name]
            if case['live_counts'][name]:
                original = array[0].copy()
                view = array.view(np.float32) if name in ('minDepthBits', 'maxConfidenceBits') else array
                view[0] = 42 if view[0] != 42 else 41
                with pytest.raises(AssertionError): check_stage_parity(case, case['bindings'])
                array[0] = original
            array[-1] += 1
            with pytest.raises(AssertionError): check_stage_parity(case, case['bindings'])
            array[-1] -= 1
        for name in case['readonly']:
            array = case['bindings'][name]
            original = array[0].copy(); array[0] = 42 if array[0] != 42 else 41
            with pytest.raises(AssertionError): check_stage_parity(case, case['bindings'])
            array[0] = original


def test_cpu_stage_resets_outside_every_clock_and_times_only_selected_entry(monkeypatch):
    import benchmarks.reprojection_support as support
    case = next(case for case in stage_cases(representative()) if case['entrypoint'] == 'rasterSupport')
    events = []; ticks = iter([10, 30, 40, 70])
    original = support.run_stage
    def execute(entrypoint, params, bindings):
        events.append(('run', entrypoint)); return original(entrypoint, params, bindings)
    def clock(): events.append(('clock',)); return next(ticks)
    copy = np.copyto
    def reset(*args, **kwargs): events.append(('reset',)); return copy(*args, **kwargs)
    monkeypatch.setattr(support, 'run_stage', execute)
    monkeypatch.setattr(support.np, 'copyto', reset)
    samples = time_cpu_stage(case, iterations=2, warmups=1, clock_ns=clock)
    assert samples == [20 / 1e6, 30 / 1e6]
    assert [item[1] for item in events if item[0] == 'run'] == ['rasterSupport'] * 3
    clock_indices = [i for i, event in enumerate(events) if event[0] == 'clock']
    for start, stop in zip(clock_indices[::2], clock_indices[1::2]):
        assert events[start + 1:stop] == [('run', 'rasterSupport')]
    check_stage_parity(case, case['bindings'])


def test_registry_ingests_nine_genuine_paired_stage_operations_without_pipeline_copy(tmp_path):
    from sintergpu.catalog import list_operations
    from sintergpu.timings import build_registry
    selected = list(stage_cases(representative()))
    environment = dict(cpu='unit-test CPU', gpu='explicit synthetic unit-test GPU identity', compiler_sha256='0' * 64)
    rows = [stage_timing_row(case, [1., 2.], [.1, .2], environment=environment, warmups=1) for case in selected]
    catalog = [dict(op) for op in list_operations() if op['id'].startswith('vkl.reprojection.')]
    # In-memory catalog represents the exact source under test; no registry file is modified.
    for op in catalog: op['sha256'] = next(row['source_sha256'] for row in rows if row['operation'] == op['id'])
    import json
    report = tmp_path / 'paired-stage-report.json'
    report.write_text(json.dumps(dict(schema_version=1, measurements=rows)))
    registry = build_registry(catalog, json.loads(report.read_text())['measurements'])
    assert registry['coverage']['both_measured'] == 9
    assert all(row['metrics'] == {'cpu_reference_wall_ms': [1., 2.], 'gpu_kernel_ms': [.1, .2]} for row in rows)
    assert len({row['workload']['id'] for row in rows}) == len(rows)
    assert all('gpu_pipeline_ms' not in row['metrics'] for row in rows)


def test_prefix_final_writes_match_each_complete_public_oracle():
    mappings = {
        'projection': {'projectPoints': ('projected_xy', 'projected_depth', 'visible', 'displacement', 'depth_consistency', 'depth_consistency_valid', 'occluded', 'occlusion_valid'), 'reduceProjection': ('overlap', 'parallax')},
        'raster': {'finalizeRaster': ('depth', 'valid', 'confidence', 'support_count')},
        'temporal': {'sampleTemporal': ('aligned_rgb', 'aligned_valid', 'alignment_confidence')},
        'reference': {'finalizeReference': ('rgb', 'depth', 'valid', 'confidence')},
    }
    for graphcase in cases(variants=('ordinary', 'all_invalid'), neighbors=3, source_depth=True, source_confidence=True, target_depth=True):
        full = graphcase['reference']()
        for case in stage_cases(graphcase):
            for output in mappings[graphcase['family']].get(case['entrypoint'], ()):
                expected = full[output].reshape(-1)
                actual = case['expected'][output + 'Out'][:expected.size]
                if expected.dtype.kind in 'biu': np.testing.assert_array_equal(actual, expected)
                else: np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6, equal_nan=True)


def test_no_complete_pipeline_callback_is_used_by_any_cpu_stage_timer(monkeypatch):
    import benchmarks.reprojection_support as support
    graphcase = representative()
    def forbidden(): raise AssertionError('whole pipeline callback must not run inside stage preparation/timing')
    graphcase['reference'] = forbidden
    calls = []; original = support.run_stage
    def run(entrypoint, params, bindings):
        calls.append(entrypoint); return original(entrypoint, params, bindings)
    monkeypatch.setattr(support, 'run_stage', run)
    for case in stage_cases(graphcase):
        calls.clear()
        support.time_cpu_stage(case, iterations=2, warmups=1)
        assert calls == [case['entrypoint']] * 3


@pytest.mark.parametrize('corrupt', [False, True])
def test_isolated_gpu_validation_precedes_replay_that_would_hide_counter_corruption(monkeypatch, corrupt):
    import benchmarks.reprojection_support as support
    import benchmarks.vkl_geometry_depth as complete_recipe
    graphcase = representative()
    selected = next(case for case in stage_cases(graphcase) if case['entrypoint'] == 'rasterSupport')
    selected['index'] = 0
    stage = dict(operation=selected['operation'], entrypoint=selected['entrypoint'], params=selected['params'], bindings=selected['role'])
    data = {resource: selected['bindings'][name].copy() for name, resource in selected['role'].items()}
    initial = {name: value.copy() for name, value in data.items()}
    events = []
    class Buffer:
        def __init__(self, name): self.name = name
        def to_numpy(self): return data[self.name].copy()
    class Prepared:
        graph = dict(stages=[stage], outputs={})
        device = object(); uploads = {}; source_paths = {}
        buffers = {name: Buffer(name) for name in data}
        def record(self, encoder): raise AssertionError('mock collector must not execute GPU recording')
        def assert_poison_tails(self):
            events.append('tail-check')
            for array in data.values(): np.testing.assert_array_equal(array[-13:], 73)
    def restore():
        for name in data: np.copyto(data[name], initial[name])
    def replay(*args): events.append('replay'); restore()
    def prefix(*args): events.append('prefix'); restore()
    def gpu(*args, **kwargs):
        events.append('gpu-batch')
        bindings = {name: data[resource] for name, resource in selected['role'].items()}
        support.run_stage('rasterSupport', selected['params'], bindings)
        if corrupt: bindings['supportBits'][0] += 1
        return dict(resident=dict(samples_ms=[.1, .2]))
    monkeypatch.setenv('SINTERGPU_GPU_TESTS', '1')
    monkeypatch.setattr(support, 'stage_cases', lambda case: iter([selected]))
    monkeypatch.setattr(support, 'runtime_environment', lambda device: dict(cpu='unit-test CPU', gpu='mock unit-test GPU', compiler_sha256='0'*64))
    monkeypatch.setattr(support, 'validate_outputs', replay)
    monkeypatch.setattr(support, 'prepare_gpu_prefix', prefix)
    monkeypatch.setattr(support, 'measure_stage', gpu)
    monkeypatch.setattr(complete_recipe, 'measure', lambda *args: {})
    if corrupt:
        with pytest.raises(AssertionError):
            support.measure_graph(Prepared(), graph_case=graphcase, iterations=2, warmups=1, reference=lambda: {})
        assert 'replay' not in events[events.index('gpu-batch') + 1:]
    else:
        report = support.measure_graph(Prepared(), graph_case=graphcase, iterations=2, warmups=1, reference=lambda: {})
        assert len(report['paired_measurements']) == 1
        assert events.index('tail-check') < len(events) - 1


def test_prefix_inputs_and_exact_gpu_readonly_state_are_both_validated():
    from benchmarks.reprojection_support import check_prefix_inputs, array_state_identity
    from sintergpu.references.reprojection_stage_numeric import run_stage
    case = next(case for case in stage_cases(representative()) if case['entrypoint'] == 'reduceProjection')
    before = {name: value.copy() for name, value in case['readonly_initial'].items()}
    check_prefix_inputs(case, before)
    before['partialCount'][0] += 1
    with pytest.raises(AssertionError): check_prefix_inputs(case, before)
    assert array_state_identity(before) != array_state_identity(case['readonly_initial'])
    before['partialCount'][0] -= 1
    run_stage(case['entrypoint'], case['params'], case['bindings'])
    # A snapshot from the actual GPU prefix is the exact readonly comparator.
    check_stage_parity(case, case['bindings'], readonly_before=before)
    before['partialParallax'][0] = 123
    with pytest.raises(AssertionError): check_stage_parity(case, case['bindings'], readonly_before=before)


def test_paired_row_rejects_mismatched_counts_and_cpu_timer_rejects_invalid_counts():
    case = next(stage_cases(representative()))
    environment = dict(cpu='test CPU', gpu='mock test GPU', compiler_sha256='0'*64)
    with pytest.raises(ValueError, match='sample counts'):
        stage_timing_row(case, [1, 2], [1], environment=environment, warmups=1)
    for iterations, warmups in [(0, 1), (1, 0), (True, 1), (1, True)]:
        with pytest.raises(ValueError): time_cpu_stage(case, iterations=iterations, warmups=warmups)
