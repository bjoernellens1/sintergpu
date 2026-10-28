# SPDX-License-Identifier: Apache-2.0
"""Fail-closed camera collector; hardware use requires independent review.

No background scheduling or overwrite. Caller holds the GPU lease until device
close. Runtime must expose the actual selected Vulkan physical-device UUID.
"""
import fcntl
import hashlib
import json
import os
import platform
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.camera_registration_fixture_history import MANIFEST_SHA256
from benchmarks.camera_registration_timing_support import source_identities
from benchmarks.device_snapshot import parse_snapshot
from sintergpu.timings import dependency_identities

ROOT = Path(__file__).resolve().parents[1]
DEVICE_UUID = '96d2dba0-95fd-5c00-e849-2e0610f24e06'
ICD = '/usr/share/vulkan/icd.d/nvidia_icd.json'
LOCK = '/tmp/sintergpu-gpu.lock'


def verify_selected_device(device, snapshot):
    """Reject absent selected-device UUID, rather than infer it from a name."""
    info = device.info
    uuid = getattr(info, 'physical_device_uuid', None)
    if not isinstance(uuid, str) or uuid.lower() != DEVICE_UUID:
        raise RuntimeError('actual selected Vulkan physical-device UUID unavailable or mismatched')
    if snapshot['physical_device_uuid'].lower() != uuid.lower(): raise RuntimeError('snapshot UUID mismatch')
    if str(info.api_name).casefold() != 'vulkan' or str(info.adapter_name) != snapshot['adapter_name']:
        raise RuntimeError('selected adapter mismatch')
    if os.environ.get('VK_DRIVER_FILES') != ICD or os.environ.get('VK_ICD_FILENAMES'):
        raise RuntimeError('NVIDIA ICD selection mismatch')
    # SlangPy's property provenance is not yet independently established.
    raise RuntimeError('selected physical-device UUID bridge is not verified')


def loaded_runtime_identity():
    """Identify installed and actually mapped SlangPy runtime/compiler libraries."""
    import slangpy
    directory = Path(slangpy.__file__).resolve().parent
    compiler = list(directory.glob('libslang-compiler*'))
    runtime = directory / 'libsgl.so'
    if len(compiler) != 1 or not compiler[0].is_file() or not runtime.is_file():
        raise RuntimeError('cannot identify unique SlangPy runtime and compiler')
    maps = Path('/proc/self/maps').read_text()
    result = {}
    for name, path in [('compiler', compiler[0]), ('runtime', runtime)]:
        path = path.resolve()
        if not any(line.split()[-1] == str(path) for line in maps.splitlines() if line.split() and line.split()[-1].startswith('/')):
            raise RuntimeError('loaded mapping unavailable: ' + name)
        result[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def identities():
    return {'v1_whole_trees': dependency_identities(), 'camera_source': source_identities(),
            'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def archive_identity(directory):
    directory = Path(directory)
    files = sorted(directory.iterdir())
    if not files or any(not p.is_file() or p.is_symlink() for p in files):
        raise RuntimeError('archive contains invalid entries')
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def fresh_snapshot():
    proc = subprocess.run(['vulkaninfo', '--summary'], check=True, capture_output=True, text=True, timeout=30)
    snapshot = parse_snapshot(proc.stdout, hostname=platform.node(), captured_at_utc=datetime.now(timezone.utc).isoformat())
    if snapshot['physical_device_uuid'].lower() != DEVICE_UUID: raise RuntimeError('probe UUID mismatch')
    return snapshot


def publish_receipt(output, payload):
    """Publish complete JSON atomically without replacing an existing path.

    A write/fsync/link failure leaves no newly published partial receipt. A
    storage failure cannot guarantee a failure receipt; the exception is surfaced.
    """
    output = Path(output)
    data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(prefix='.' + output.name + '.', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output, follow_symlinks=False)
    finally:
        os.unlink(name)


def collect(*, output, archive):
    if os.getenv('SINTERGPU_GPU_TESTS') != '1': raise RuntimeError('explicit GPU opt-in required')
    if os.getenv('VK_DRIVER_FILES') != ICD or os.getenv('VK_ICD_FILENAMES'):
        raise RuntimeError('NVIDIA ICD selection mismatch')
    output = Path(output)
    if os.path.lexists(output): raise FileExistsError(output)
    reservation = output.with_name(output.name + '.reservation')
    reserve_fd = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    lock_fd = None
    try:
        if os.path.lexists(output): raise FileExistsError(output)
        try:
            lock_fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode): raise RuntimeError('canonical lock must be a regular file')
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = identities()
            archive_before = archive_identity(archive)
            # Complete corpus and CPU differential/size checks precede device creation.
            from benchmarks.vkl_camera_registration import checked_preflight, run_checked
            checked_preflight(archive=archive, manifest_sha256=MANIFEST_SHA256)
            snapshot_before = fresh_snapshot()
            import slangpy as spy
            device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
            try:
                identity = verify_selected_device(device, snapshot_before)
                runtime_before = loaded_runtime_identity()
                from benchmarks.reprojection_support import runtime_environment
                environment = runtime_environment(device)
                for field in ('physical_device_uuid', 'driver', 'adapter_name'):
                    if environment['device_snapshot'][field] != snapshot_before[field]:
                        raise RuntimeError('fresh driver snapshot mismatch: ' + field)
                result = run_checked(device=device, snapshot=snapshot_before, archive=archive,
                                     manifest_sha256=MANIFEST_SHA256, iterations=50, warmups=5)
                runtime_after = loaded_runtime_identity()
                if runtime_after != runtime_before: raise RuntimeError('loaded runtime identity changed')
            finally:
                device.close()
            snapshot_after = fresh_snapshot()
            for field in ('physical_device_uuid', 'driver', 'adapter_name', 'vendor_id', 'device_id'):
                if snapshot_after[field] != snapshot_before[field]: raise RuntimeError('device snapshot changed: ' + field)
            after = identities()
            if after != before: raise RuntimeError('source identity changed during collection')
            if archive_identity(archive) != archive_before: raise RuntimeError('donor archive changed during collection')
            payload = {'status': 'diagnostic_only', 'identity': identity, 'runtime': runtime_before,
                       'snapshot_before': snapshot_before, 'snapshot_after': snapshot_after,
                       'source_before': before, 'source_after': after, 'archive_sha256': archive_before,
                       'limitations': ['Physical backing allocation disjointness unverified',
                                       'Quantized timestamp data are diagnostic only'], 'result': result}
        except BaseException as exc:
            publish_receipt(output, {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        publish_receipt(output, payload)
    finally:
        if lock_fd is not None: os.close(lock_fd)
        os.close(reserve_fd)
        os.unlink(reservation)
