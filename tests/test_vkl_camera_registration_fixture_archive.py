import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import camera_registration_fixture_archive as archive
from benchmarks.camera_registration_fixtures import payloads

_REAL_SOURCE_IDENTITIES = archive.source_identities


def test_payloads_cover_five_functions_and_have_unique_typed_identities():
    cases = list(payloads())
    assert {p['function'] for p in cases} == {'camera_rectify_grid', 'camera_rectify_from_grid', 'camera_rectify', 'depth_camera_rays', 'depth_register'}
    ids = [archive.payload_identity(p)['sha256'] for p in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) >= 30
    assert any(p['kwargs'].get('debug') for p in cases)


def test_input_identity_distinguishes_types_shapes_and_nan_bits():
    identity = archive.payload_identity
    assert identity({'a': True}) != identity({'a': 1})
    assert identity({'a': (1,2)}) != identity({'a': [1,2]})
    assert identity({'a': np.array([1], dtype=np.float32)}) != identity({'a': np.array([[1]], dtype=np.float32)})
    assert identity({'a': float('nan')}) != identity({'a': -float('nan')})
    with pytest.raises(ValueError):
        identity({'a': np.array([object()], dtype=object)})


def test_loader_rejects_missing_and_invalid_manifest(tmp_path):
    with pytest.raises((ValueError, OSError)):
        archive.load_fixture(next(iter(payloads())), tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'schema_version': 999}))
    with pytest.raises(ValueError):
        archive.load_fixture(next(iter(payloads())), tmp_path)


@pytest.fixture
def synthetic_archive(tmp_path, monkeypatch):
    import hashlib
    from benchmarks.camera_registration_fixtures import output_schema
    p = next(iter(payloads()))
    monkeypatch.setattr(archive,'source_identities',lambda root: {'camera.py': 'a'*64})
    identity = archive.payload_identity(p)
    outputs = {key: np.zeros(shape,dtype=np.dtype(dtype)) for key,(dtype,shape) in output_schema(p).items()}
    filename = identity['sha256']+'.npz'
    np.savez_compressed(tmp_path/filename,**outputs)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    descriptors = {k:{'dtype':v.dtype.str,'shape':list(v.shape),'sha256':sha(v.tobytes())} for k,v in outputs.items()}
    record = {'identity':identity,'file':filename,'sha256':sha((tmp_path/filename).read_bytes()),'outputs':descriptors}
    runtime = {'torch_version':'2.14.1+rocm7.2','execution_device':'cpu','torch_config':'config','python':'version','numpy_version':'version','cpu':{'architecture':'x','proc_cpuinfo':'cpu flags'},'threads':{'intraop':1,'interop':1},'binaries_sha256':{name:'b'*64 for name in ('python_executable','torch/_C','torch/lib/libtorch_cpu.so','torch/lib/libtorch_python.so','torch/lib/libc10.so','torch_distribution/METADATA','torch_distribution/RECORD')}}
    manifest = {'schema_version':1,'profile':{'name':archive.PROFILE,'frozen_revision':archive.REVISION,'runtime':runtime,'source_sha256':{'camera.py':'a'*64}},'records':[record]}
    def save():
        (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    save()
    return p, manifest, save, tmp_path


def test_loader_accepts_matching_layout_and_rejects_modified_input(synthetic_archive):
    p, manifest, save, path = synthetic_archive
    result = archive.load_fixture(p,path)
    assert result.keys() == manifest['records'][0]['outputs'].keys()
    modified = dict(p,kwargs=dict(p['kwargs'], output_hw=(4,5)))
    with pytest.raises(ValueError):
        archive.load_fixture(modified,path)


@pytest.mark.parametrize('mutation', ['profile','revision','runtime','source','duplicate','filename','missing_output','dtype','shape','checksum'])
def test_loader_rejects_corrupt_manifests(synthetic_archive, mutation):
    p,m,save,path = synthetic_archive
    record=m['records'][0]
    key=next(iter(record['outputs']))
    if mutation=='profile': m['profile']['name']='wrong'
    elif mutation=='revision': m['profile']['frozen_revision']='wrong'
    elif mutation=='runtime': m['profile']['runtime']['torch_version']='wrong'
    elif mutation=='source': m['profile']['source_sha256']['camera.py']='wrong'
    elif mutation=='duplicate': m['records'].append(record.copy())
    elif mutation=='filename': record['file']='../escape.npz'
    elif mutation=='missing_output': del record['outputs'][key]
    elif mutation=='dtype': record['outputs'][key]['dtype']='<i4'
    elif mutation=='shape': record['outputs'][key]['shape']=[1]
    elif mutation=='checksum': record['sha256']='0'*64
    save()
    with pytest.raises((ValueError,KeyError)):
        archive.load_fixture(p,path)


def test_loader_rejects_modified_archive_bytes(synthetic_archive):
    p,m,save,path=synthetic_archive
    file=path/m['records'][0]['file']
    file.write_bytes(file.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        archive.load_fixture(p,path)


def test_export_refuses_existing_destination(tmp_path):
    from tools.export_camera_registration_fixtures import export_archive
    with pytest.raises(FileExistsError):
        export_archive(tmp_path)


def test_complete_loader_rejects_partial_archive(synthetic_archive):
    _, _, _, path = synthetic_archive
    with pytest.raises(ValueError, match='corpus'):
        archive.load_all(path)


def test_export_validation_rejects_numeric_drift_and_incomplete_outputs():
    from benchmarks.camera_registration_fixtures import output_schema
    from tools.export_camera_registration_fixtures import _validate
    p = next(iter(payloads()))
    outputs = {key: np.zeros(shape,dtype=np.dtype(dtype)) for key,(dtype,shape) in output_schema(p).items()}
    _validate(p,outputs,outputs)
    wrong = {k:v.copy() for k,v in outputs.items()}
    float_key = next(k for k,v in wrong.items() if v.dtype.kind == 'f')
    wrong[float_key].flat[0] = 1
    with pytest.raises(AssertionError):
        _validate(p,wrong,outputs)
    del wrong[float_key]
    with pytest.raises(ValueError):
        _validate(p,wrong,outputs)


@pytest.mark.parametrize('name', ['src/sintergpu/__init__.py', 'src/sintergpu/data/VERSION'])
def test_source_closure_binds_imported_package_and_version(tmp_path, name):
    import shutil
    root = archive.ROOT
    originals = archive.source_identities(root)
    for relative in set(originals) | {'src/sintergpu/__init__.py', 'src/sintergpu/data/VERSION'}:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    assert archive.source_identities(tmp_path) == originals
    with (tmp_path/name).open('ab') as stream:
        stream.write(b'\nchanged\n')
    assert archive.source_identities(tmp_path) != originals


@pytest.mark.parametrize('name', ['src/sintergpu/__init__.py', 'src/sintergpu/data/VERSION'])
def test_loader_rejects_mutated_package_inputs(synthetic_archive, tmp_path, monkeypatch, name):
    import shutil
    payload, manifest, save, directory = synthetic_archive
    monkeypatch.setattr(archive, 'source_identities', _REAL_SOURCE_IDENTITIES)
    source = archive.source_identities(archive.ROOT)
    root = tmp_path / 'source'
    for relative in source:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive.ROOT / relative, target)
    manifest['profile']['source_sha256'] = source
    save()
    archive.load_fixture(payload, directory, root=root)
    with (root/name).open('ab') as stream:
        stream.write(b'\nchanged\n')
    with pytest.raises(ValueError, match='source closure'):
        archive.load_fixture(payload, directory, root=root)


def test_historical_camera_loader_binds_old_version_and_all_current_code(tmp_path):
    import shutil
    from benchmarks.camera_registration_fixture_history import load_canonical, FROZEN_VERSION
    root = tmp_path / 'source'
    for relative in archive.source_identities():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive.ROOT / relative, target)
    # A changed current release version is allowed only by verifying a separate
    # exact historical VERSION, while all executable closure bytes remain bound.
    (root / 'src/sintergpu/data/VERSION').write_text('9.9.9\n')
    directory = archive.ROOT / 'qualification/evidence/vkl-camera-registration-frozen-0.2.6-closure-v2'
    assert len(load_canonical(directory, root=root)) == 80
    bad_version = tmp_path / 'bad-version'
    bad_version.write_bytes(FROZEN_VERSION.read_bytes() + b'x')
    with pytest.raises(ValueError, match='historical fixture version'):
        load_canonical(directory, root=root, frozen_version=bad_version)
    with (root / 'src/sintergpu/references/camera_registration_numeric.py').open('ab') as stream:
        stream.write(b'\n# modified\n')
    with pytest.raises(ValueError, match='executable source closure'):
        load_canonical(directory, root=root)
