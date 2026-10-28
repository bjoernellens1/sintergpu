"""Strict portable frozen-donor fixtures; never regenerates reference outputs."""
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'vkl-camera-eager-f32-torch-cpu-2.14.1-rocm7.2'
REVISION = '3384023a311651ea4260cf12ddab1567972f89f3'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _describe(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object arrays are not fixture inputs')
        return {'type': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'sha256': _sha(value.tobytes(order='C'))}
    if value is None:
        return {'type': 'none'}
    if type(value) in (bool, int, str):
        return {'type': type(value).__name__, 'value': value}
    if type(value) is float:
        return {'type': 'float64', 'bits': struct.pack('>d', value).hex()}
    if type(value) in (tuple, list):
        return {'type': type(value).__name__, 'items': [_describe(v) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {'type': 'dict', 'items': {k: _describe(v) for k, v in sorted(value.items())}}
    raise ValueError('unsupported fixture input type: ' + type(value).__name__)


def payload_identity(payload):
    description = _describe(payload)
    encoded = json.dumps(description, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'sha256': _sha(encoded), 'description': description}


def source_identities(root=ROOT):
    root = Path(root)
    paths = sorted((root / 'references/donors/vkl/src/vkl/kernels/reference').glob('*.py'))
    if not paths:
        raise ValueError('frozen donor source closure is missing')
    paths += [root / name for name in ('benchmarks/camera_registration_fixtures.py',
              'benchmarks/camera_registration_fixture_archive.py', 'tools/export_camera_registration_fixtures.py',
              'benchmarks/reprojection_fixture_archive.py', 'tools/export_reprojection_fixtures.py',
              'benchmarks/reprojection_fixtures.py', 'src/sintergpu/reprojection.py', 'src/sintergpu/vkl.py',
              'src/sintergpu/references/camera_registration_numeric.py',
              'src/sintergpu/__init__.py', 'src/sintergpu/data/VERSION')]
    result = {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in paths}
    donor = 'references/donors/vkl/src/vkl/kernels/reference/camera_registration.py'
    if result.get(donor) != '648628355f4b86cc708f0c22d04e3cb6bd21ad85a8cd51868b37b1e1fd05ad3c':
        raise ValueError('frozen camera donor hash mismatch')
    return result


def load_fixture(payload, directory, *, root=ROOT):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported frozen fixture schema')
    profile = manifest.get('profile', {})
    runtime = profile.get('runtime', {})
    binaries = runtime.get('binaries_sha256', {})
    threads = runtime.get('threads', {})
    if (profile.get('name') != PROFILE or profile.get('frozen_revision') != REVISION
            or runtime.get('torch_version') != '2.14.1+rocm7.2'
            or runtime.get('execution_device') != 'cpu'
            or not runtime.get('torch_config') or not runtime.get('python') or not runtime.get('cpu')
            or not runtime.get('numpy_version')
            or not isinstance(runtime.get('cpu'), dict)
            or not runtime['cpu'].get('architecture') or not runtime['cpu'].get('proc_cpuinfo')
            or threads != {'intraop': 1, 'interop': 1}
            or any(type(threads.get(k)) is not int for k in ('intraop', 'interop'))
            or not {'python_executable', 'torch/_C', 'torch/lib/libtorch_cpu.so', 'torch/lib/libtorch_python.so', 'torch/lib/libc10.so', 'torch_distribution/METADATA', 'torch_distribution/RECORD'} <= set(binaries)
            or not all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v) for v in binaries.values())):
        raise ValueError('unpinned or incomplete frozen reference runtime profile')
    if manifest.get('profile', {}).get('source_sha256') != source_identities(root):
        raise ValueError('frozen fixture source closure mismatch')
    identity = payload_identity(payload)
    records = manifest.get('records', [])
    ids = [r['identity']['sha256'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate frozen fixture identity')
    matches = [r for r in records if r['identity'] == identity]
    if len(matches) != 1:
        raise ValueError('missing or mismatched frozen fixture identity')
    record = matches[0]
    if record['file'] != identity['sha256'] + '.npz':
        raise ValueError('invalid frozen fixture filename')
    path = directory / record['file']
    if path.is_symlink() or path.resolve().parent != directory or _sha(path.read_bytes()) != record['sha256']:
        raise ValueError('frozen fixture archive checksum mismatch')
    with np.load(path, allow_pickle=False) as archive:
        outputs = record['outputs']
        if set(archive.files) != set(outputs):
            raise ValueError('frozen fixture output key mismatch')
        result = {}
        for key, descriptor in outputs.items():
            array = archive[key]
            actual = {'dtype': array.dtype.str, 'shape': list(array.shape),
                      'sha256': _sha(array.tobytes(order='C'))}
            if actual != descriptor:
                raise ValueError('frozen fixture output layout or checksum mismatch: ' + key)
            result[key] = array
    from benchmarks.camera_registration_fixtures import output_schema
    schema = output_schema(payload)
    if {k: (v.dtype.str, v.shape) for k, v in result.items()} != schema:
        raise ValueError('frozen fixture output schema mismatch')
    return result


def load_all(directory, *, root=ROOT):
    """Validate every expected state, rejecting incomplete or extra records."""
    from benchmarks.camera_registration_fixtures import payloads
    expected = list(payloads())
    manifest = json.loads((Path(directory) / 'manifest.json').read_text())
    records = manifest.get('records', [])
    identities = [payload_identity(p)['sha256'] for p in expected]
    observed = [r['identity']['sha256'] for r in records]
    if len(set(identities)) != len(identities) or len(observed) != len(identities) or set(observed) != set(identities):
        raise ValueError('incomplete or duplicate frozen fixture corpus')
    if manifest.get('counts') != {'cases': len(expected), 'states': len(expected)}:
        raise ValueError('frozen fixture count mismatch')
    return [(p, load_fixture(p, directory, root=root)) for p in expected]
