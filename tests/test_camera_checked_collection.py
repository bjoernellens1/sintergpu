"""CPU-only checks for borrowed-device and per-sample camera collection."""
from types import SimpleNamespace
import numpy as np
import pytest

from benchmarks.camera_registration_timing_support import measure_checked_pipeline
from benchmarks.vkl_camera_registration import run_checked


def test_borrowed_device_rejected_before_archive_or_dispatch(monkeypatch):
    from benchmarks import camera_checked_collection as collector
    monkeypatch.setenv('VK_DRIVER_FILES', collector.ICD)
    monkeypatch.delenv('VK_ICD_FILENAMES', raising=False)
    device = SimpleNamespace(info=SimpleNamespace(api_name='vulkan', adapter_name='NVIDIA'))
    snapshot = {'physical_device_uuid': collector.DEVICE_UUID, 'adapter_name': 'NVIDIA'}
    with pytest.raises(RuntimeError, match='UUID unavailable'):
        run_checked(device=device, snapshot=snapshot, archive='missing', manifest_sha256='0'*64)


def test_pipeline_checks_every_sample_before_next_upload_or_dispatch():
    events = []
    class Buffer:
        size = 4
        def copy_from_numpy(self, data): events.append('upload')
        def to_numpy(self): events.append('readback'); return np.array([1], np.float32)
    class Encoder:
        def write_timestamp(self, pool, index): events.append(('timestamp', index))
        def finish(self): return self
    class Device:
        info = SimpleNamespace(timestamp_frequency=1000000)
        def create_query_pool(self, *args): return SimpleNamespace(get_results=lambda *args: [10, 20])
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self, encoder): events.append('submit')
        def wait_for_idle(self): events.append('wait')
    class Prepared:
        device = Device()
        def record(self, encoder): events.append('record')
    class Fixture:
        prepared = Prepared()
        buffers = {'input': Buffer(), 'output': Buffer()}
        initial = {'input': np.array([1], np.float32)}
        graph = {'outputs': {'result': 'output'}, 'resources': {'input': {'dtype': 'float32'}, 'output': {'dtype': 'float32'}}}
        def outputs_and_guards(self, expected):
            events.append('validate')
            if events.count('validate') == 2: raise AssertionError('corrupted input')
            return expected
    expected = {'result': np.array([1], np.float32)}
    with pytest.raises(AssertionError, match='corrupted input'):
        measure_checked_pipeline(Fixture(), expected, expected, iterations=2, warmups=1,
                                 clock_ns=lambda: 1, timestamp_type='timestamp')
    assert events.count('record') == 2
    assert events.count('validate') == 2
    assert events.count('upload') == 2  # writable reset before each sample


def test_device_identity_requires_actual_selected_uuid(monkeypatch):
    from benchmarks.camera_checked_collection import verify_selected_device, DEVICE_UUID, ICD
    monkeypatch.setenv('VK_DRIVER_FILES', ICD)
    monkeypatch.delenv('VK_ICD_FILENAMES', raising=False)
    snapshot = {'physical_device_uuid': DEVICE_UUID, 'adapter_name': 'NVIDIA'}
    device = SimpleNamespace(info=SimpleNamespace(api_name='vulkan', adapter_name='NVIDIA'))
    with pytest.raises(RuntimeError, match='UUID unavailable'):
        verify_selected_device(device, snapshot)
    device.info.physical_device_uuid = DEVICE_UUID
    with pytest.raises(RuntimeError, match='bridge is not verified'):
        verify_selected_device(device, snapshot)
    device.info.physical_device_uuid = '00000000-0000-0000-0000-000000000000'
    with pytest.raises(RuntimeError, match='UUID unavailable'):
        verify_selected_device(device, snapshot)


def test_collector_rejects_existing_output_without_touching_lock(tmp_path, monkeypatch):
    from benchmarks import camera_checked_collection as collector
    monkeypatch.setenv('SINTERGPU_GPU_TESTS', '1')
    monkeypatch.setenv('VK_DRIVER_FILES', collector.ICD)
    monkeypatch.delenv('VK_ICD_FILENAMES', raising=False)
    monkeypatch.setattr(collector, 'LOCK', str(tmp_path / 'absent-lock'))
    output = tmp_path / 'receipt.json'
    output.write_bytes(b'preserve')
    with pytest.raises(FileExistsError): collector.collect(output=output, archive=tmp_path)
    assert output.read_bytes() == b'preserve'
    assert not (tmp_path / 'absent-lock').exists()


def test_atomic_receipt_never_overwrites_and_has_no_partial_output(tmp_path, monkeypatch):
    from benchmarks import camera_checked_collection as collector
    output = tmp_path / 'receipt.json'
    collector.publish_receipt(output, {'status': 'failed', 'error': 'device close'})
    assert __import__('json').loads(output.read_text())['error'] == 'device close'
    with pytest.raises(FileExistsError): collector.publish_receipt(output, {'status': 'success'})
    assert __import__('json').loads(output.read_text())['status'] == 'failed'
    second = tmp_path / 'second.json'
    monkeypatch.setattr(collector.os, 'link', lambda *a, **k: (_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError, match='disk error'): collector.publish_receipt(second, {'status': 'failed'})
    assert not second.exists()
    assert not list(tmp_path.glob('.second.json.*'))


def test_missing_lock_publishes_valid_failure_receipt(tmp_path, monkeypatch):
    import json
    from benchmarks import camera_checked_collection as collector
    monkeypatch.setenv('SINTERGPU_GPU_TESTS', '1')
    monkeypatch.setenv('VK_DRIVER_FILES', collector.ICD)
    monkeypatch.delenv('VK_ICD_FILENAMES', raising=False)
    monkeypatch.setattr(collector, 'LOCK', str(tmp_path / 'missing-lock'))
    output = tmp_path / 'failure.json'
    with pytest.raises(FileNotFoundError): collector.collect(output=output, archive=tmp_path)
    assert json.loads(output.read_text())['error_type'] == 'FileNotFoundError'
    assert not (tmp_path / 'failure.json.reservation').exists()


def test_preflight_checks_reference_and_resource_bounds(monkeypatch):
    from benchmarks import vkl_camera_registration as recipe
    expected = {'result': np.array([1], np.float32)}
    case = {'name': 'bounded', 'reference': lambda: expected,
            'graph': {'resources': {'x': {'elements': 2, 'element_size': 4}}}}
    monkeypatch.setattr(recipe, 'canonical_records', lambda *args: {'bounded': expected})
    monkeypatch.setattr(recipe, 'cases', lambda **kwargs: iter([case]))
    assert recipe.checked_preflight(archive='x', manifest_sha256='x')['bounded'] is expected
    case['graph']['resources']['x']['elements'] = 64 * 1024 * 1024
    with pytest.raises(ValueError, match='resource exceeds'):
        recipe.checked_preflight(archive='x', manifest_sha256='x')


def test_checked_pipeline_retains_raw_ticks_and_validates_each_phase():
    events = []
    class Buffer:
        size = 4
        def copy_from_numpy(self, data): events.append('reset/upload')
        def to_numpy(self): events.append('timed-readback'); return np.ones(1, np.float32)
    class Encoder:
        def write_timestamp(self, pool, index): events.append(('tick', index))
        def finish(self): return self
    class Device:
        info = SimpleNamespace(timestamp_frequency=1000000)
        def create_query_pool(self, *args): return SimpleNamespace(get_results=lambda *args: [21, 26])
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self, _): events.append('submit')
        def wait_for_idle(self): events.append('wait')
    class Prepared:
        device = Device()
        def record(self, _): events.append('record')
    class Fixture:
        prepared = Prepared()
        buffers = {'input': Buffer(), 'output': Buffer()}
        initial = {'input': np.ones(1, np.float32)}
        graph = {'outputs': {'result': 'output'}, 'resources': {'input': {'dtype': 'float32'}, 'output': {'dtype': 'float32'}}}
        def outputs_and_guards(self, expected): events.append('validate'); return expected
    expected = {'result': np.ones(1, np.float32)}
    clock = iter([0, 1000000, 2000000, 3000000])
    result = measure_checked_pipeline(Fixture(), expected, expected, iterations=1, warmups=1,
                                      clock_ns=lambda: next(clock), timestamp_type='timestamp')
    assert result['resident_tick_pairs'] == [[21, 26]]
    assert result['resident_samples_ms'] == [0.005]
    assert result['transfer_inclusive_samples_ms'] == [1.0]
    assert events.count('validate') == events.count('record') == 4
    for index, event in enumerate(events):
        if event == 'record':
            assert 'validate' in events[index + 1:next((j for j in range(index + 1, len(events)) if events[j] == 'record'), len(events))]


def test_stage_reset_catches_second_sample_skipped_store(monkeypatch):
    import sys
    from benchmarks.camera_registration_timing_support import measure_stage
    state = np.array([5, 73], dtype=np.uint32)  # stale prior complete graph output
    events = []
    class Buffer:
        size = state.nbytes
        def copy_from_numpy(self, data): state[:] = data; events.append('reset')
        def to_numpy(self): return state.copy()
    class Encoder:
        def global_barrier(self): pass
        def write_timestamp(self, *args): pass
        def finish(self): return self
    class Device:
        info = SimpleNamespace(timestamp_frequency=1_000_000)
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self, _): pass
        def wait_for_idle(self): pass
        def create_query_pool(self, *args): return SimpleNamespace(get_results=lambda *args: [1, 2])
    class Kernel:
        def dispatch(self, **kwargs):
            events.append('dispatch')
            if events.count('dispatch') == 1: state[0] = 5
            # The second sample intentionally skips the output store.
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(QueryType=SimpleNamespace(timestamp='timestamp')))
    device = Device()
    prepared = SimpleNamespace(device=device, _dispatches=[(Kernel(), [1,1,1], {}, False)])
    fixture = SimpleNamespace(prepared=prepared, initial={}, buffers={'out': Buffer()},
                              graph={'resources': {'out': {'dtype': 'uint32', 'elements': 1}}},
                              read=lambda: {'out': state.copy()})
    initial = np.array([73,73], dtype=np.uint32)
    case = {'index': 0, 'entrypoint': 'test', 'bindings': {'output': initial.copy()},
            'role': {'output': 'out'}, 'written': ('output',), 'readonly': (),
            'initial': {'output': initial}, 'readonly_initial': {},
            'expected': {'output': np.array([5,73], dtype=np.uint32)}, 'live_counts': {'output': 1}}
    with pytest.raises(AssertionError):
        measure_stage(fixture, case, iterations=1, warmups=1, retain_ticks=True)
    assert events == ['reset', 'dispatch', 'reset', 'dispatch']
