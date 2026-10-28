"""Canonical donor differential tests; explicit opt-in and external GPU lock.

Each archive record is validated before device creation. Supported cases are
whatever the source-bound corpus contains: five operations, B1/B8, pinhole and
Brown, direct/cached, debug, sampling, collisions and empty states. GPU tests
provide no trajectory, gradient, no-regression or consumer-adoption evidence.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.camera_registration_fixtures import payloads, output_schema
from benchmarks.camera_registration_gpu_fixture import fixture_graph, FixtureResources

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'qualification/evidence/vkl-camera-registration-frozen-0.2.6-closure-v2'
MANIFEST_SHA256 = '00dde261bee03952bdc134b10f85c309f02869f00810ad967b8fd3ad4d862c54'
CASES = list(payloads())
CASE_IDS = [p['case_id'] for p in CASES]
if len(set(CASE_IDS)) != len(CASE_IDS):
    raise ValueError('duplicate camera fixture case id')


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].shape == value.shape and actual[key].dtype == value.dtype, key
        if value.dtype.kind in 'biu': np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else: np.testing.assert_allclose(actual[key], value, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)


def test_cpu_mapping_covers_all_current_fixtures():
    assert {p['function'] for p in CASES} == {'camera_rectify_grid','camera_rectify_from_grid','camera_rectify','depth_camera_rays','depth_register'}
    for payload in CASES:
        graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
        expected = output_schema(payload)
        mapped = {'result' if name == 'depth_rays' else name for name in graph['outputs']}
        assert mapped == set(expected), payload['case_id']
        assert all(name in graph['resources'] for name in inputs)


def validated_archive(directory):
    from benchmarks.camera_registration_fixture_archive import load_all
    path = Path(directory) / 'manifest.json'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('canonical camera fixture manifest digest mismatch')
    return load_all(directory)


@pytest.fixture(scope='module')
def canonical():
    records = validated_archive(Path(os.environ.get('SINTERGPU_CAMERA_FROZEN_ARCHIVE', ARCHIVE)))
    assert [p['case_id'] for p, _ in records] == CASE_IDS
    return {p['case_id']: (p, expected) for p, expected in records}


@pytest.fixture(scope='module')
def device(canonical):
    # The device fixture depends on complete, fail-closed corpus verification.
    import slangpy as spy
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    try: yield device
    finally: device.close()


def enabled_features(device, graph):
    # No feature is inferred from device support; unknown requirements fail closed.
    if graph['features']:
        raise RuntimeError('unverified logical-device features: ' + ', '.join(graph['features']))
    return ()


def execute(device, fixture, payload, donor):
    from sintergpu.references import camera_registration_numeric as reference
    actual_reference = getattr(reference, payload['function'])(**payload['kwargs'])
    if not isinstance(actual_reference, dict): actual_reference = {'result': actual_reference}
    compare(actual_reference, donor)
    encoder = device.create_command_encoder()
    fixture.prepared.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    actual = fixture.outputs_and_guards(actual_reference)
    compare(actual, donor); compare(actual, actual_reference)


optin = pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS') != '1', reason='explicit GPU qualification only')


@optin
@pytest.mark.parametrize('case_id', CASE_IDS)
def test_canonical_outputs_and_poison_tails(device, canonical, case_id):
    payload, donor = canonical[case_id]
    graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
    fixture = FixtureResources(device, graph, inputs, enabled_features=enabled_features(device,graph))
    execute(device, fixture, payload, donor)


@optin
def test_reuse_dense_empty_then_sparse(device, canonical):
    ids = ('collisions-tie', 'collisions-empty', 'collisions-ulp')
    assert all(case_id in canonical for case_id in ids), 'required reuse fixture states missing'
    graph, inputs = fixture_graph(canonical[ids[0]][0], max_workgroups_x=65535)
    fixture = FixtureResources(device, graph, inputs, enabled_features=enabled_features(device,graph))
    for index, case_id in enumerate(ids):
        payload, donor = canonical[case_id]
        next_graph, next_inputs = fixture_graph(payload, max_workgroups_x=65535)
        assert next_graph == graph
        if index: fixture.upload(next_inputs)
        execute(device, fixture, payload, donor)


def test_cpu_canonical_preflight_before_any_gpu_creation(monkeypatch):
    import sys
    before = set(sys.modules)
    records = validated_archive(ARCHIVE)
    assert [payload['case_id'] for payload, _ in records] == CASE_IDS
    assert 'torch' not in set(sys.modules) - before
    assert 'slangpy' not in set(sys.modules) - before
    with pytest.raises((FileNotFoundError, ValueError)):
        validated_archive(ARCHIVE / 'missing')


def test_modified_archive_and_rehashed_manifest_rejected_before_device(tmp_path):
    import json
    import shutil
    copied = tmp_path / 'archive'
    shutil.copytree(ARCHIVE, copied)
    manifest_path = copied / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = manifest['records'][0]
    archive_path = copied / record['file']
    with np.load(archive_path, allow_pickle=False) as source:
        outputs = {key: source[key].copy() for key in source.files}
    key = next(iter(outputs))
    if outputs[key].dtype.kind == 'b':
        outputs[key].flat[0] = not outputs[key].flat[0]
    else:
        outputs[key].flat[0] = 12345
    assert hashlib.sha256(outputs[key].tobytes(order='C')).hexdigest() != record['outputs'][key]['sha256']
    np.savez(archive_path, **outputs)
    record['sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    record['outputs'][key]['sha256'] = hashlib.sha256(outputs[key].tobytes(order='C')).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest digest'):
        validated_archive(copied)


def test_fake_device_allocation_record_and_output_guards(monkeypatch):
    import sys
    from types import SimpleNamespace
    from sintergpu.references import camera_registration_numeric as reference
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(BufferUsage=SimpleNamespace(shader_resource=1, unordered_access=2)))
    class Buffer:
        def __init__(self, data, short=False):
            self.array = data.copy() if not short else data[:-1].copy()
            self.size = self.array.nbytes
        def to_numpy(self): return self.array.copy()
        def copy_from_numpy(self, value): self.array[:] = value
    class Kernel:
        def dispatch(self, *, thread_count, vars, command_encoder): command_encoder.calls += 1
    class Device:
        def __init__(self, short=False): self.short=short
        def create_buffer(self, *, element_count, struct_size, usage, data):
            assert element_count == data.size and struct_size == data.dtype.itemsize and usage == 3
            return Buffer(data, self.short)
        def load_module(self, path): return SimpleNamespace(entry_point=lambda name: name)
        def link_program(self, modules, entries): return entries[0]
        def create_compute_kernel(self, program): return Kernel()
    payload = next(p for p in CASES if p['case_id'] == 'cached-nonfinite-nearest')
    graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
    with pytest.raises(ValueError, match='undersized'):
        FixtureResources(Device(short=True), graph, inputs, enabled_features=())
    fixture = FixtureResources(Device(), graph, inputs, enabled_features=())
    encoder = SimpleNamespace(calls=0, global_barrier=lambda: None)
    fixture.prepared.record(encoder)
    assert encoder.calls == len(graph['stages'])
    expected = reference.camera_rectify_from_grid(**payload['kwargs'])
    name = graph['outputs']['image']
    fixture.buffers[name].array[:expected['image'].size] = expected['image'].reshape(-1)
    compare(fixture.outputs_and_guards(expected), expected)
    fixture.buffers[name].array[-1] = 0
    with pytest.raises(AssertionError, match='tail'): fixture.outputs_and_guards(expected)
    fixture.buffers[name].array[-1] = 73
    fixture.buffers['input_image'].array[0] += 1
    with pytest.raises(AssertionError, match='input changed'): fixture.outputs_and_guards(expected)


def test_diagnostic_word_encoding_cpu_contract():
    values = np.array([0, 1, 0xfffffffe, -1], dtype='<i8')
    words = values.view('<u4').reshape(-1, 2)
    np.testing.assert_array_equal(words, [[0,0], [1,0], [0xfffffffe,0], [0xffffffff,0xffffffff]])
    assert values.dtype.itemsize == 8


@optin
def test_gpu_debug_diagnostics_match_exact_int64_words(device, canonical):
    # These are actual shader stores: empty sentinels and positive collision counts.
    for case_id in ('collisions-tie', 'collisions-empty', 'collisions-ulp'):
        payload, donor = canonical[case_id]
        graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
        fixture = FixtureResources(device, graph, inputs, enabled_features=enabled_features(device, graph))
        execute(device, fixture, payload, donor)
        state = fixture.read()
        for key in ('winner_source_index', 'collision_count'):
            assert donor[key].dtype == np.dtype('int64')
            expected_words = donor[key].astype('<i8', copy=False).view('<u4').reshape(-1, 2)
            actual_words = state[graph['outputs'][key]][:donor[key].size].view('<u4').reshape(-1, 2)
            np.testing.assert_array_equal(actual_words, expected_words, err_msg=key)


@optin
def test_gpu_finalizer_writes_max_word_and_negative_sentinel(device, canonical):
    # Isolated finalizer dispatch checks boundary encodings impractical to
    # produce with billions of source points; no claim about scene parity.
    payload = dict(canonical['collisions-tie'][0])
    payload['kwargs'] = dict(payload['kwargs'], output_hw=(1, 2))
    graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
    fixture = FixtureResources(device, graph, inputs, enabled_features=enabled_features(device, graph))
    for name, live in {'min_depth_bits': [0x3f800000, 0x7f800000],
                       'winner_index': [0xfffffffe, 5],
                       'collision_count_scratch': [0xfffffffe, 0]}.items():
        spec = graph['resources'][name]
        data = np.full(spec['elements'] + 13, 73, dtype=np.uint32)
        data[:spec['elements']] = live
        fixture.buffers[name].copy_from_numpy(data)
    kernel, threads, bindings, barrier = fixture.prepared._dispatches[-1]
    encoder = device.create_command_encoder()
    if barrier: encoder.global_barrier()
    kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    state = fixture.read()
    expected = {'winner_source_index': [0xfffffffe, -1], 'collision_count': [0xfffffffe, 0]}
    for key, values in expected.items():
        result = state[graph['outputs'][key]][:2]
        assert result.dtype == np.dtype('int64')
        np.testing.assert_array_equal(result, np.array(values, dtype=np.int64))
        np.testing.assert_array_equal(result.view('<u4').reshape(-1, 2),
            np.array(values, dtype='<i8').view('<u4').reshape(-1, 2))
        np.testing.assert_array_equal(state[graph['outputs'][key]][2:], 73)


def test_big_endian_host_is_rejected_before_device_allocation(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'byteorder', 'big')
    payload = next(p for p in CASES if p['case_id'] == 'collisions-tie')
    graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
    with pytest.raises(RuntimeError, match='little-endian'):
        FixtureResources(object(), graph, inputs, enabled_features=())
