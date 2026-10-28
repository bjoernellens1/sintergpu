"""CPU-only guards for camera paired timing boundaries and state validation."""
import numpy as np
import pytest

from sintergpu.references.camera_registration_stage_numeric import run_stage
from benchmarks.vkl_camera_registration import cases
from benchmarks.camera_registration_timing_support import (stage_cases, reset_cpu_stage,
    check_stage_parity, check_prefix, time_cpu_stage)


def test_all_fixture_stage_boundaries_prepare_without_gpu():
    names = set(); count = 0
    for graphcase in cases():
        stages = list(stage_cases(graphcase))
        assert len(stages) == len(graphcase['graph']['stages'])
        for stage in stages:
            names.add(stage['entrypoint']); count += 1
            assert len(stage['workload']['sha256']) == 64
            run_stage(stage['entrypoint'], stage['params'], stage['bindings'])
            check_stage_parity(stage, stage['bindings'])
            check_prefix(stage, stage['readonly_initial'])
    assert count > 100
    assert {'rectifyGrid', 'rectifyFromGrid', 'depthCameraRays', 'clearRegistration',
            'projectRegistration', 'registrationMin', 'registrationWinnerCount',
            'finalizeRegistration', 'finalizeRegistrationDebug'} <= names


def test_atomic_reset_is_outside_clock_and_does_not_accumulate():
    graphcase = next(cases(case_ids=['collisions-tie']))
    stage = next(c for c in stage_cases(graphcase) if c['entrypoint'] == 'registrationWinnerCount')
    calls = []
    def clock():
        calls.append([stage['bindings'][name].copy() for name in stage['written']])
        return len(calls) * 1_000_000
    samples = time_cpu_stage(stage, iterations=3, warmups=2, clock_ns=clock)
    assert samples == [1., 1., 1.]
    for start in calls[::2]:
        for observed, name in zip(start, stage['written']):
            np.testing.assert_array_equal(observed, stage['initial'][name])
    for stop in calls[1::2]:
        for observed, name in zip(stop, stage['written']):
            np.testing.assert_array_equal(observed, stage['expected'][name])


def test_validation_rejects_each_write_readonly_and_tail_corruption():
    case = next(c for c in stage_cases(next(cases(case_ids=['collisions-tie'])))
                if c['entrypoint'] == 'registrationWinnerCount')
    run_stage(case['entrypoint'], case['params'], case['bindings'])
    actual = {k: v.copy() for k, v in case['bindings'].items()}
    for name in case['written']:
        mutated = {k: v.copy() for k, v in actual.items()}
        mutated[name].view(np.uint8)[0] ^= 127
        with pytest.raises(AssertionError): check_stage_parity(case, mutated)
        mutated = {k: v.copy() for k, v in actual.items()}
        mutated[name][-1] += 1
        with pytest.raises(AssertionError): check_stage_parity(case, mutated)
    for name in case['readonly']:
        mutated = {k: v.copy() for k, v in actual.items()}
        mutated[name].view(np.uint8)[0] ^= 127
        with pytest.raises(AssertionError): check_stage_parity(case, mutated)
        with pytest.raises(AssertionError): check_prefix(case, mutated)


def test_invalid_case_and_bad_sample_counts_rejected():
    with pytest.raises(ValueError, match='unknown case'): list(cases(case_ids=['not-a-case']))
    with pytest.raises(ValueError, match='unique'): list(cases(case_ids=['collisions-tie'] * 2))
    case = next(stage_cases(next(cases(case_ids=['collisions-tie']))))
    for iterations, warmups in ((0, 1), (1, 0), (True, 1)):
        with pytest.raises(ValueError): time_cpu_stage(case, iterations=iterations, warmups=warmups)


def test_gpu_timestamps_and_early_corruption_rejected_before_next_prefix(monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks.camera_registration_timing_support import measure_stage
    import benchmarks.camera_registration_timing_support as timing_support
    monkeypatch.setattr(timing_support, 'reset_writable', lambda fixture: None)
    events = []
    case = next(c for c in stage_cases(next(cases(case_ids=['collisions-tie'])))
                if c['entrypoint'] == 'registrationWinnerCount')
    # First sample produces corrupt actual state. A hypothetical later good
    # sample must never execute before that corruption is reported.
    states = {name: value.copy() for name, value in case['bindings'].items()}
    resources = {resource: states[name] for name, resource in case['role'].items()}
    graph = {'resources': {resource: {'elements': case['live_counts'][name]} for name, resource in case['role'].items()}}
    class Buffer:
        def __init__(self, array): self.array = array
        def to_numpy(self): return self.array.copy()
    class Encoder:
        def global_barrier(self): events.append('barrier')
        def write_timestamp(self, pool, index): events.append(('timestamp', index))
        def finish(self): return self
    class Kernel:
        def __init__(self, name): self.name = name
        def dispatch(self, **kwargs):
            events.append(self.name)
            if self.name == 'selected':
                for name in case['written']: states[name][:] = case['expected'][name]
                if events.count('selected') == 2:
                    states[case['written'][0]][0] += 1
    class Pool:
        def get_results(self, start, count): return [1, 2]
    class Device:
        info = SimpleNamespace(timestamp_frequency=1_000_000)
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self, encoder): events.append('submit')
        def wait_for_idle(self): events.append('wait')
        def create_query_pool(self, kind, count): return Pool()
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(QueryType=SimpleNamespace(timestamp='timestamp')))
    prepared = SimpleNamespace(device=Device(), _dispatches=[
        (Kernel('clear'), [1,1,1], {}, False),
        (Kernel('prefix'), [1,1,1], {}, True),
        (Kernel('selected'), [1,1,1], {}, True)])
    case = dict(case, index=2)
    fixture = SimpleNamespace(prepared=prepared, graph=graph, initial={},
        buffers={name: Buffer(array) for name, array in resources.items()},
        read=lambda: {name: array.copy() for name, array in resources.items()})
    with pytest.raises(AssertionError): measure_stage(fixture, case, iterations=2, warmups=1)
    assert events.count('clear') == 2 and events.count('selected') == 2
    start = events.index(('timestamp', 0)); stop = events.index(('timestamp', 1))
    assert events[start-1] == 'barrier'
    assert events[start+1:stop] == ['selected']


def test_atomic_min_rejects_close_but_wrong_bits():
    from benchmarks.camera_registration_timing_support import validate_measured_sample
    from types import SimpleNamespace
    case = next(c for c in stage_cases(next(cases(case_ids=['collisions-tie'])))
                if c['entrypoint'] == 'registrationMin')
    before = {name: value.copy() for name, value in case['bindings'].items()}
    run_stage(case['entrypoint'], case['params'], case['bindings'])
    actual = {name: value.copy() for name, value in case['bindings'].items()}
    states = {resource: actual[name] for name, resource in case['role'].items()}
    graph = {'resources': {resource: {'elements': case['live_counts'][name]} for name, resource in case['role'].items()}}
    fixture = SimpleNamespace(graph=graph, initial={}, read=lambda: states)
    validate_measured_sample(fixture, case, before, actual)
    actual['minDepthBits'][0] = np.float32(1.000001).view(np.uint32)
    with pytest.raises(AssertionError, match='atomic minimum bits'):
        validate_measured_sample(fixture, case, before, actual)


def test_bad_archive_fails_before_device_initialization(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks.vkl_camera_registration import run
    monkeypatch.setenv('SINTERGPU_GPU_TESTS', '1')
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(Device=lambda **kw: pytest.fail('device must not be created')))
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest digest'):
        run(archive=tmp_path, manifest_sha256='0' * 64, case_ids=['collisions-tie'])


def test_recipe_loads_historical_canonical_archive_after_version_change():
    from benchmarks.vkl_camera_registration import canonical_records
    from benchmarks.camera_registration_fixtures import payloads
    from benchmarks.camera_registration_fixture_history import MANIFEST_SHA256, ROOT
    directory = ROOT / 'qualification/evidence/vkl-camera-registration-frozen-0.2.6-closure-v2'
    records = canonical_records(directory, MANIFEST_SHA256)
    assert len(records) == 80
    assert set(records) == {p['case_id'] for p in payloads()}
    with pytest.raises(ValueError, match='manifest digest'):
        canonical_records(directory, '0' * 64)


def test_timing_source_identity_binds_historical_fixture_inputs():
    import hashlib
    from benchmarks.camera_registration_timing_support import source_identities, ROOT
    identities = source_identities()
    for relative in ('benchmarks/camera_registration_fixture_history.py',
                     'qualification/evidence/vkl-camera-registration-frozen-version-0.2.6.txt'):
        assert identities[relative] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_source_manifest_includes_historical_fixture_version():
    from pathlib import Path
    from setuptools._distutils.filelist import FileList
    root = Path(__file__).resolve().parents[1]
    relative = 'qualification/evidence/vkl-camera-registration-frozen-version-0.2.6.txt'
    files = FileList()
    files.allfiles = [relative]
    for line in (root / 'MANIFEST.in').read_text().splitlines():
        files.process_template_line(line)
    assert relative in files.files
