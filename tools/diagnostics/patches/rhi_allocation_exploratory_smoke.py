# SPDX-License-Identifier: Apache-2.0
"""REVIEW REQUIRED. Exploratory only; no timing or provenance qualification.

Explicit ICD is a loader selection setting, not proof of the loaded ICD. The
RHI instance-dispatch provider and complete native closure remain unverified.
Mapping pathname/inode observations and file hashes do not prove mapped page
contents, prevent races, or authenticate a caller-selected binary identity.
"""
import argparse
import fcntl
import json
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

from benchmarks.depth_pyramid_timestamp_probe import BorrowedLease, LOCK


@contextmanager
def owned_lease():
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = BorrowedLease(fd)
        lease.verify()
        yield lease
    finally:
        os.close(fd)


def write_once(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

HERE = Path(__file__).resolve().parent
CASES_HASH = 'b4b988cb947153d723e035c02cbfb0c6dee426418c6e7d1a292f18d984c03976'


def file_hash(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file(): raise ValueError('regular absolute file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapped_file(path):
    """Observe matching pathname and inode; overlay map and stat devices may differ.

    This does not authenticate the mapped pages or eliminate file mutation races.
    """
    path = Path(path)
    file_hash(path)  # Reject non-regular and final symlink paths first.
    st = path.stat()
    expected = path.resolve()
    matches = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(' (deleted)'):
            continue
        try:
            major, minor = (int(x, 16) for x in fields[3].split(':'))
            inode = int(fields[4])
        except ValueError:
            continue
        mapped = Path(fields[5])
        if mapped == expected:
            if inode != st.st_ino:
                raise ValueError('mapped pathname refers to a different inode')
            matches.append((fields[5], major, minor))
    if not matches or len(set(matches)) != 1:
        raise ValueError('expected binary mapping unavailable or ambiguous')
    # Overlayfs can expose the lower filesystem's device in /proc maps, whereas
    # stat reports the merged mount's device. Retain both; do not equate them.
    mapped_path, major, minor = matches[0]
    current = path.stat()
    if (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
        raise ValueError('binary path changed during inspection')
    return {'path': str(path), 'sha256': file_hash(path), 'mapped_path': mapped_path,
            'mapping_device': [major, minor], 'stat_device': [os.major(st.st_dev), os.minor(st.st_dev)],
            'inode': st.st_ino}


def load_verified_cases(path, expected):
    """Compile the exact hashed bytes; never consult an existing pyc."""
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('regular absolute case source required')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('case source differs from reviewed candidate')
    module = types.ModuleType('allocation_smoke_cases')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    if file_hash(path) != expected:
        raise ValueError('case source changed during execution')
    return module


def setup_runtime(args):
    icd = os.environ.get('VK_DRIVER_FILES')
    if not icd or os.environ.get('VK_ICD_FILENAMES') or ':' in icd: raise ValueError('one explicit VK_DRIVER_FILES required')
    icd_hash = file_hash(icd)
    expected = ((args.extension, args.extension_sha), (args.libsgl, args.libsgl_sha), (args.compiler, args.compiler_sha))
    if len({(Path(path).stat().st_dev, Path(path).stat().st_ino) for path, _ in expected}) != 3: raise ValueError('binary roles must be distinct')
    if any(len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) for _, sha in expected): raise ValueError('lowercase SHA-256 required')
    if any(file_hash(path) != sha for path, sha in expected): raise ValueError('requested binary digest mismatch')
    cases_path = HERE / 'rhi_allocation_vulkan_cases.py'
    if file_hash(cases_path) != CASES_HASH: raise ValueError('case source differs from reviewed candidate')
    if any(name == 'slangpy' or name.startswith('slangpy.') for name in sys.modules): raise ValueError('slangpy already imported')
    spy = importlib.import_module('slangpy')
    extension = sys.modules.get('slangpy.slangpy_ext')
    if extension is None or Path(extension.__file__).resolve() != Path(args.extension).resolve(): raise ValueError('wrong slangpy extension loaded')
    mappings = [mapped_file(path) for path, _ in expected]
    if any(record['sha256'] != sha for record, (_, sha) in zip(mappings, expected)): raise ValueError('mapped binary changed')
    cases = load_verified_cases(cases_path, CASES_HASH)
    return spy, cases, {'icd_configuration': {'path': icd, 'sha256': icd_hash}, 'mapped_binaries': mappings,
                        'scope': 'path/inode and file-hash observations only; mapped pages, loader, ICD binary, provider and complete closure unverified'}


def verify_runtime_unchanged(args, provenance):
    icd = provenance['icd_configuration']
    if os.environ.get('VK_DRIVER_FILES') != icd['path'] or os.environ.get('VK_ICD_FILENAMES') or file_hash(icd['path']) != icd['sha256']:
        raise ValueError('ICD configuration changed')
    for record in provenance['mapped_binaries']:
        if mapped_file(record['path']) != record:
            raise ValueError('binary or mapping changed')
    if file_hash(HERE / 'rhi_allocation_vulkan_cases.py') != CASES_HASH:
        raise ValueError('case source changed')


def execute(args, *, setup=None, lease_context=owned_lease, verify_runtime=verify_runtime_unchanged):
    # Production execution is blocked: provider, loader, ICD and source closure
    # are not independently verified. Only injected CPU mocks may exercise flow.
    if setup is None or setup is setup_runtime:
        raise RuntimeError('exploratory Vulkan execution blocked pending independent provenance review')
    if args.execute is not True: raise ValueError('explicit --execute required')
    output = Path(args.output)
    if output.exists() or output.is_symlink(): raise FileExistsError('output already exists')
    uuid = bytes.fromhex(args.uuid.replace('-', ''))
    if len(uuid) != 16 or not any(uuid) or args.driver < 0: raise ValueError('expected UUID and raw driver required')
    error = None
    result = None
    provenance = None
    try:
        with lease_context() as lease:
            if not isinstance(lease, BorrowedLease): raise ValueError('canonical borrowed lease required')
            lease.verify()
            device = None
            cases_started = False
            failures = []
            try:
                spy, cases, provenance = setup(args)  # import only after lock
                adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
                if len(adapters) != 1 or adapters[0].vendor_id != args.vendor or adapters[0].device_id != args.device:
                    raise ValueError('ambiguous or unexpected Vulkan adapter')
                device = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=adapters[0].luid)
                # Same-backend actual UUID and typed handles before any buffer allocation.
                cases.identity(spy, device, uuid, args.driver)
                cases_started = True  # run_cases owns and closes device in its finally
                result = cases.run_cases(spy, device, lease=lease, expected_uuid=uuid, expected_driver=args.driver)
            except Exception as exc:
                failures.append(exc)
            finally:
                if device is not None and not cases_started:
                    try: device.close()
                    except Exception as exc: failures.append(exc)
            if provenance is not None:
                try: verify_runtime(args, provenance)
                except Exception as exc: failures.append(exc)
            try: lease.verify()
            except Exception as exc: failures.append(exc)
            if len(failures) == 1: raise failures[0]
            if failures: raise ExceptionGroup('smoke and teardown/lease failure', failures)
    except Exception as exc:
        error = exc
    payload = {'status': 'failure' if error else 'exploratory_only',
               'scope': 'not provenance, allocation attestation, performance or GPU qualification',
               'observations': provenance}
    if error: payload.update(error_type=type(error).__name__, error=str(error))
    else: payload['result'] = result
    write_once(output, payload)
    if error: raise error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    for name in ('uuid', 'output', 'extension', 'extension-sha', 'libsgl', 'libsgl-sha', 'compiler', 'compiler-sha'):
        p.add_argument('--'+name, required=True)
    for name in ('driver', 'vendor', 'device'):
        p.add_argument('--'+name, type=lambda s: int(s, 0), required=True)
    execute(p.parse_args())

if __name__ == '__main__': main()
