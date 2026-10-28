"""Validate persisted, single-device Vulkan summary evidence without probing hardware."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

_PROPERTIES = ('apiVersion', 'driverVersion', 'vendorID', 'deviceID', 'deviceName',
               'driverName', 'driverInfo', 'deviceUUID', 'driverUUID')
_UUID = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')


def _timestamp(value):
    _nonempty(value, 'captured_at_utc')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('captured_at_utc must be an ISO UTC timestamp') from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError('captured_at_utc must have UTC timezone')


def parse_snapshot(raw: str, *, hostname, captured_at_utc) -> dict:
    """Parse exactly one GPU from actual ``vulkaninfo --summary`` stdout."""
    _nonempty(raw, 'raw_vulkaninfo')
    _nonempty(hostname, 'hostname')
    _timestamp(captured_at_utc)
    blocks = list(re.finditer(r'^GPU\d+:', raw, re.MULTILINE))
    if len(blocks) != 1:
        raise ValueError('vulkaninfo must contain exactly one GPU block')
    body = raw[blocks[0].end():]
    properties = {}
    for key in _PROPERTIES:
        values = re.findall(r'^\s*' + key + r'[^\S\n]*=[^\S\n]*(.*)$', body, re.MULTILINE)
        if len(values) != 1 or not values[0].strip():
            raise ValueError(f'vulkaninfo requires one nonempty {key}')
        properties[key] = values[0].strip()
    for key in ('deviceUUID', 'driverUUID'):
        if not _UUID.fullmatch(properties[key]):
            raise ValueError(f'invalid {key}')
    return {
        'schema_version': 1,
        'hostname': hostname,
        'captured_at_utc': captured_at_utc,
        'api': 'vulkan',
        'adapter_name': properties['deviceName'],
        'physical_device_uuid': properties['deviceUUID'],
        'driver': {key: properties[prop] for key, prop in (
            ('name', 'driverName'), ('info', 'driverInfo'),
            ('version', 'driverVersion'), ('uuid', 'driverUUID'))},
        'vendor_id': properties['vendorID'],
        'device_id': properties['deviceID'],
        'vulkan_api_version': properties['apiVersion'],
        'raw_vulkaninfo': raw,
        'raw_vulkaninfo_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
    }


def load_snapshot(path, *, hostname, api, adapter_name) -> dict:
    """Read and cross-check persisted metadata against its embedded raw evidence."""
    data = Path(path).read_bytes()
    snapshot = json.loads(data)
    if not isinstance(snapshot, dict):
        raise ValueError('snapshot must be a JSON object')
    required = ('raw_vulkaninfo', 'hostname', 'captured_at_utc')
    if any(key not in snapshot for key in required):
        raise ValueError('missing snapshot fields')
    parsed = parse_snapshot(snapshot['raw_vulkaninfo'], hostname=snapshot['hostname'],
                            captured_at_utc=snapshot['captured_at_utc'])
    if type(snapshot.get('schema_version')) is not int:
        raise ValueError('schema_version must be integer 1')
    if any(key not in snapshot or snapshot[key] != value for key, value in parsed.items()):
        raise ValueError('snapshot metadata does not match raw vulkaninfo')
    if hostname != parsed['hostname']:
        raise ValueError('snapshot hostname does not match caller')
    if not isinstance(api, str) or api.casefold() != 'vulkan':
        raise ValueError('snapshot requires Vulkan API')
    if adapter_name != parsed['adapter_name']:
        raise ValueError('snapshot adapter does not match caller')
    return {**snapshot, 'snapshot_file_sha256': hashlib.sha256(data).hexdigest()}
