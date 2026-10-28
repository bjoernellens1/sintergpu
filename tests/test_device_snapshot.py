"""Frozen vulkaninfo summary properties from the 8060S qualification host."""
import hashlib
import json

import pytest

from benchmarks.device_snapshot import load_snapshot, parse_snapshot

RAW = """==========
VULKANINFO
==========
Vulkan Instance Version: 1.4.341
Devices:
========
GPU0:
\tapiVersion         = 1.4.354
\tdriverVersion      = 26.2.3
\tvendorID           = 0x1002
\tdeviceID           = 0x1586
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon 8060S Graphics (RADV STRIX_HALO)
\tdriverID           = DRIVER_ID_MESA_RADV
\tdriverName         = radv
\tdriverInfo         = Mesa 26.2.3
\tconformanceVersion = 1.4.5.3
\tdeviceUUID         = 00000000-c400-0000-0000-000000000000
\tdriverUUID         = 414d442d-4d45-5341-2d44-525600000000
"""
NAME = 'AMD Radeon 8060S Graphics (RADV STRIX_HALO)'
STAMP = '2026-10-28T10:00:00+00:00'


def snapshot():
    return parse_snapshot(RAW, hostname='host', captured_at_utc=STAMP)


def write_snapshot(tmp_path, value):
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(value))
    return path


def load(path, **kwargs):
    return load_snapshot(path, **{'hostname': 'host', 'api': 'VULKAN', 'adapter_name': NAME, **kwargs})


def test_roundtrip_and_file_hash(tmp_path):
    value = snapshot()
    assert value['driver'] == {'name': 'radv', 'info': 'Mesa 26.2.3', 'version': '26.2.3', 'uuid': '414d442d-4d45-5341-2d44-525600000000'}
    assert value['physical_device_uuid'] == '00000000-c400-0000-0000-000000000000'
    assert value['raw_vulkaninfo_sha256'] == hashlib.sha256(RAW.encode()).hexdigest()
    path = write_snapshot(tmp_path, value)
    assert load(path) == {**value, 'snapshot_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('raw', ['', RAW.replace('GPU0:', 'Device:'), RAW + '\nGPU1:\n', RAW + '\tdriverName = duplicate\n', RAW.replace('radv\n', '\n'), RAW.replace('00000000-c400-0000-0000-000000000000', 'invalid')])
def test_reject_bad_raw(raw):
    with pytest.raises(ValueError):
        parse_snapshot(raw, hostname='host', captured_at_utc=STAMP)


@pytest.mark.parametrize('field', ['apiVersion', 'driverVersion', 'vendorID', 'deviceID', 'deviceName', 'driverName', 'driverInfo', 'deviceUUID', 'driverUUID'])
def test_require_each_property(field):
    raw = '\n'.join(line for line in RAW.splitlines() if line.strip().split(' ')[0] != field)
    with pytest.raises(ValueError):
        parse_snapshot(raw, hostname='host', captured_at_utc=STAMP)


@pytest.mark.parametrize('stamp', ['', 'garbage', '2026-10-28T10:00:00', '2026-10-28T10:00:00+01:00'])
def test_require_utc_timestamp(stamp):
    with pytest.raises(ValueError):
        parse_snapshot(RAW, hostname='host', captured_at_utc=stamp)


@pytest.mark.parametrize('field', ['schema_version', 'hostname', 'captured_at_utc', 'api', 'adapter_name', 'physical_device_uuid', 'driver', 'vendor_id', 'device_id', 'vulkan_api_version', 'raw_vulkaninfo', 'raw_vulkaninfo_sha256'])
def test_require_snapshot_fields(tmp_path, field):
    value = snapshot()
    del value[field]
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, value))


@pytest.mark.parametrize('field', ['schema_version', 'api', 'adapter_name', 'physical_device_uuid', 'vendor_id', 'device_id', 'vulkan_api_version', 'raw_vulkaninfo_sha256', 'raw_vulkaninfo', 'captured_at_utc'])
def test_reject_metadata_tampering(tmp_path, field):
    value = snapshot()
    value[field] = 'forged'
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, value))


@pytest.mark.parametrize('field', ['name', 'info', 'version', 'uuid'])
def test_reject_driver_tampering(tmp_path, field):
    value = snapshot()
    value['driver'][field] = 'forged'
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, value))


@pytest.mark.parametrize('kwargs', [{'hostname': 'other'}, {'api': 'cuda'}, {'adapter_name': 'other'}])
def test_reject_caller_mismatch(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, snapshot()), **kwargs)


def test_missing_and_invalid_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / 'missing')
    path = tmp_path / 'bad.json'
    path.write_text('not json')
    with pytest.raises(ValueError):
        load(path)


def test_raw_edit_with_recomputed_hash_still_rejects_stale_metadata(tmp_path):
    value = snapshot()
    value['raw_vulkaninfo'] = RAW.replace('Mesa 26.2.3', 'Mesa 99.0.0')
    value['raw_vulkaninfo_sha256'] = hashlib.sha256(value['raw_vulkaninfo'].encode()).hexdigest()
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, value))


@pytest.mark.parametrize('value', [None, [], 'snapshot', 1])
def test_reject_nonobject_json(tmp_path, value):
    with pytest.raises(ValueError):
        load(write_snapshot(tmp_path, value))


def test_utc_z_timestamp_and_no_age_policy():
    assert parse_snapshot(RAW, hostname='host', captured_at_utc='2099-01-01T00:00:00Z')['captured_at_utc'] == '2099-01-01T00:00:00Z'
