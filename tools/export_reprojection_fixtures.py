#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU-only export of unchanged frozen D4/T0 donor outputs.

Run in a fresh process with the pinned Torch build. This does not initialize a
GPU, import SlangPy, alter donors, or generate fixtures during GPU qualification.
Installed artifact hashes describe the actual runtime, not an upstream wheel.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import types

import numpy as np

from benchmarks.reprojection_fixture_archive import PROFILE, REVISION, ROOT, payload_identity, source_identities
from benchmarks.reprojection_fixtures import payloads

TORCH_VERSION = '2.14.1+rocm7.2'


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_profile(torch):
    if str(torch.__version__) != TORCH_VERSION:
        raise ValueError('frozen fixture export requires Torch ' + TORCH_VERSION)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threads = dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())
    if threads != dict(intraop=1, interop=1):
        raise ValueError('frozen fixture export requires one intraop and interop thread')
    package = Path(torch.__file__).resolve().parent
    binaries = {'python_executable': Path(sys.executable).resolve(),
                'torch/_C': Path(torch._C.__file__).resolve()}
    for name in ('libtorch_cpu.so', 'libtorch_python.so', 'libc10.so'):
        binaries['torch/lib/' + name] = package / 'lib' / name
    distribution = importlib.metadata.distribution('torch')
    for name in ('METADATA', 'RECORD'):
        matches = [p for p in distribution.files or () if str(p).endswith('.dist-info/' + name)]
        if len(matches) != 1:
            raise ValueError('missing or ambiguous installed Torch distribution ' + name)
        binaries['torch_distribution/' + name] = Path(distribution.locate_file(matches[0]))
    cpuinfo = Path('/proc/cpuinfo')
    cpu = dict(architecture=platform.machine(), processor=platform.processor(),
               proc_cpuinfo=cpuinfo.read_text() if cpuinfo.is_file() else '')
    if not cpu['proc_cpuinfo']:
        raise ValueError('CPU ISA provenance requires /proc/cpuinfo')
    return dict(torch_version=str(torch.__version__), execution_device='cpu',
                torch_config=torch.__config__.show(), python=sys.version, numpy_version=np.__version__,
                cpu=cpu, threads=threads,
                binaries_sha256={name: _file_sha(path) for name, path in binaries.items()})


def _frozen_module(name, root):
    path = Path(root) / 'references/donors/vkl/src/vkl/kernels/reference'
    # A private package preserves the donors' relative imports unchanged.
    package = '_sintergpu_reprojection_fixture_frozen'
    if package in sys.modules and sys.modules[package].__path__ != [str(path)]:
        raise ValueError('frozen donor package already loaded from a different root')
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    return importlib.import_module(package + '.' + name)


def numpy_outputs(payload):
    """Evaluate the independent NumPy reference using the identical call kwargs."""
    module = ('reprojection_numeric' if payload['oracle_module'] == 'reproject'
              else 'temporal_reprojection_numeric')
    oracle = getattr(importlib.import_module('sintergpu.references.' + module), payload['oracle_function'])
    return oracle(**payload['oracle_kwargs'])


def _validate_outputs(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError('frozen donor and NumPy output keys differ')
    for key, reference in expected.items():
        array = actual[key]
        if array.shape != reference.shape:
            raise ValueError('frozen donor and NumPy output shapes differ: ' + key)
        if reference.dtype.kind in 'biu':
            if array.dtype.kind not in 'biu':
                raise ValueError('integer/bool frozen output has noninteger dtype: ' + key)
            np.testing.assert_array_equal(array, reference, err_msg=key)
        else:
            np.testing.assert_allclose(array, reference, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)


def _donor_outputs(torch, payload, root):
    kwargs = {}
    for key, value in payload['oracle_kwargs'].items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value.copy())
            if value.device.type != 'cpu':
                raise ValueError('frozen donor inputs must reside on CPU')
        kwargs[key] = value
    oracle = getattr(_frozen_module(payload['oracle_module'], root), payload['oracle_function'])
    result = oracle(**kwargs)
    outputs = {}
    for key, tensor in result.items():
        if tensor.device.type != 'cpu':
            raise ValueError('frozen donor outputs must reside on CPU')
        array = tensor.detach().numpy().copy()
        if array.dtype.hasobject:
            raise ValueError('object arrays are not frozen outputs')
        outputs[key] = array
    return outputs


def export_archive(destination, *, torch_module=None, root=ROOT):
    """Export a fresh complete archive; reject existing paths and numeric drift."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('refusing to overwrite frozen fixture destination: ' + str(destination))
    if torch_module is None:
        torch_module = importlib.import_module('torch')
    runtime = _runtime_profile(torch_module)
    sources = source_identities(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    try:
        records = []
        identities = set()
        case_ids = set()
        for payload in payloads():
            try:
                outputs = _donor_outputs(torch_module, payload, root)
                _validate_outputs(outputs, numpy_outputs(payload))
            except Exception as error:
                raise RuntimeError('frozen CPU export failed for ' + payload['case_id']
                                   + ', iteration=' + str(payload['iteration'])) from error
            case_ids.add(payload['case_id'])
            identity = payload_identity(payload)
            if identity['sha256'] in identities:
                raise ValueError('duplicate payload identity')
            identities.add(identity['sha256'])
            path = staging / (identity['sha256'] + '.npz')
            np.savez_compressed(path, **outputs)
            descriptors = {key: dict(dtype=value.dtype.str, shape=list(value.shape),
                                    sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
                           for key, value in outputs.items()}
            records.append(dict(identity=identity, file=path.name, sha256=_file_sha(path), outputs=descriptors))
        if len(records) != 113 or len(case_ids) != 63:
            raise ValueError('expected all 113 frozen fixture states from 63 cases')
        if source_identities(root) != sources:
            raise ValueError('source closure changed during frozen fixture export')
        manifest = dict(schema_version=1, counts=dict(cases=len(case_ids), states=len(records)),
                        profile=dict(name=PROFILE, frozen_revision=REVISION,
                                     source_sha256=sources, runtime=runtime), records=records)
        (staging / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
        # Linux renameat2 RENAME_NOREPLACE provides atomic publication without
        # replacing a destination created concurrently after the initial check.
        _publish(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish(staging, destination):
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='fresh archive directory (must not already exist)')
    args = parser.parse_args(argv)
    manifest = export_archive(args.output)
    print(f"Exported {len(manifest['records'])} CPU frozen fixture states to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
