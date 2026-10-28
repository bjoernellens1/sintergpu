"""Portable donor fixtures fail closed on content, lineage and runtime drift."""
import hashlib
import json

import numpy as np
import pytest

from benchmarks.reprojection_fixture_archive import load_fixture, payload_identity, source_identities


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'source'
    for name in ('references/donors/vkl/src/vkl/kernels/reference/reproject.py',
                 'benchmarks/reprojection_fixtures.py', 'benchmarks/reprojection_fixture_archive.py',
                 'tools/export_reprojection_fixtures.py', 'src/sintergpu/references/reprojection_numeric.py',
                 'src/sintergpu/references/temporal_reprojection_numeric.py',
                 'src/sintergpu/references/geometry_depth_numeric.py', 'src/sintergpu/reprojection.py',
                 'src/sintergpu/vkl.py'):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    payload = {'case_id': 'literal', 'iteration': 0, 'inputs': {'depth': np.array([0., -0., np.nan], np.float32)},
               'oracle_kwargs': {'flag': False, 'target_hw': (1, 3)}, 'graph': {}, 'case_flags': {}}
    identity = payload_identity(payload)
    directory = tmp_path / 'archive'; directory.mkdir()
    path = directory / (identity['sha256'] + '.npz')
    value = np.array([1, 0], np.uint32); np.savez_compressed(path, valid=value)
    manifest = {'schema_version': 1, 'profile': {
        'name': 'vkl-f32-torch-cpu-2.14.1-rocm7.2', 'frozen_revision': '3384023a311651ea4260cf12ddab1567972f89f3',
        'source_sha256': source_identities(root),
        'runtime': {'torch_version': '2.14.1+rocm7.2', 'execution_device': 'cpu',
                    'torch_config': 'recorded build configuration', 'python': '3.13.12', 'numpy_version': np.__version__,
                    'cpu': {'architecture': 'x86_64'}, 'threads': {'intraop': 1, 'interop': 1},
                    'binaries_sha256': {'torch/_C.so': 'a' * 64}}},
        'records': [{'identity': identity, 'file': path.name,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'outputs': {'valid': {'dtype': value.dtype.str, 'shape': list(value.shape),
                                           'sha256': hashlib.sha256(value.tobytes()).hexdigest()}}}]}
    def save(): (directory / 'manifest.json').write_text(json.dumps(manifest))
    save()
    return root, directory, payload, manifest, save, path


def test_complete_typed_payload_identity_and_valid_archive(archive):
    root, directory, payload, _, _, _ = archive
    np.testing.assert_array_equal(load_fixture(payload, directory, root=root)['valid'], [1, 0])
    identity = payload_identity(payload)
    for value in (True, False, 0., '0', None):
        assert payload_identity({**payload, 'iteration': value}) != identity
    changed = {**payload, 'inputs': {'depth': payload['inputs']['depth'].copy()}}
    changed['inputs']['depth'][1] = 0.
    assert payload_identity(changed) != identity  # Signed zero is input content.


def test_scalar_float_nan_payload_bits_are_part_of_identity():
    import struct
    first = struct.unpack('>d', bytes.fromhex('7ff8000000000001'))[0]
    second = struct.unpack('>d', bytes.fromhex('fff8000000000002'))[0]
    assert payload_identity({'scalar': first}) != payload_identity({'scalar': second})


def test_numpy_validation_and_graph_planner_closure_is_bound(archive):
    root, directory, payload, _, _, _ = archive
    required = {'src/sintergpu/references/reprojection_numeric.py',
                'src/sintergpu/references/temporal_reprojection_numeric.py',
                'src/sintergpu/references/geometry_depth_numeric.py',
                'src/sintergpu/reprojection.py', 'src/sintergpu/vkl.py'}
    assert required <= source_identities(root).keys()
    (root / 'src/sintergpu/references/reprojection_numeric.py').write_text('validator changed')
    with pytest.raises(ValueError, match='source closure'):
        load_fixture(payload, directory, root=root)


def test_numpy_runtime_version_is_required(archive):
    root, directory, payload, manifest, save, _ = archive
    del manifest['profile']['runtime']['numpy_version']; save()
    with pytest.raises(ValueError, match='runtime'):
        load_fixture(payload, directory, root=root)


@pytest.mark.parametrize('mutation', ['input', 'kwarg', 'donor', 'archive', 'dtype', 'shape', 'key', 'missing', 'duplicate', 'filename'])
def test_fixture_corruption_or_mismatched_call_fails(archive, mutation):
    root, directory, payload, manifest, save, path = archive
    record = manifest['records'][0]
    if mutation == 'input': payload['inputs']['depth'][0] = 1
    elif mutation == 'kwarg': payload['oracle_kwargs']['flag'] = 0
    elif mutation == 'donor': (root / 'references/donors/vkl/src/vkl/kernels/reference/reproject.py').write_text('changed')
    elif mutation == 'archive': path.write_bytes(path.read_bytes() + b'corrupt')
    elif mutation == 'dtype': record['outputs']['valid']['dtype'] = '<i4'
    elif mutation == 'shape': record['outputs']['valid']['shape'] = [1, 2]
    elif mutation == 'key': record['outputs']['other'] = record['outputs'].pop('valid')
    elif mutation == 'missing': manifest['records'] = []
    elif mutation == 'duplicate': manifest['records'].append(record.copy())
    elif mutation == 'filename': record['file'] = '../outside.npz'
    save()
    with pytest.raises(ValueError): load_fixture(payload, directory, root=root)


@pytest.mark.parametrize('mutation', ['name', 'revision', 'runtime', 'build', 'device', 'binary', 'threads', 'configuration'])
def test_unpinned_or_incomplete_reference_runtime_fails(archive, mutation):
    root, directory, payload, manifest, save, _ = archive
    profile = manifest['profile']; runtime = profile['runtime']
    if mutation == 'name': profile['name'] = 'arbitrary'
    elif mutation == 'revision': profile['frozen_revision'] = '0' * 40
    elif mutation == 'runtime': del profile['runtime']
    elif mutation == 'build': runtime['torch_version'] = '2.14.1+cpu'
    elif mutation == 'device': runtime['execution_device'] = 'cuda'
    elif mutation == 'binary': runtime['binaries_sha256'] = {}
    elif mutation == 'threads': del runtime['threads']
    elif mutation == 'configuration': del runtime['torch_config']
    save()
    with pytest.raises(ValueError, match='profile|runtime'): load_fixture(payload, directory, root=root)
