"""Validate the 0.2.6 camera corpus against its exact historical source closure.

Only the package version is historical: all executable closure inputs must match
current bytes. The original archive and manifest are never modified.
"""
import hashlib
import json
import tempfile
from pathlib import Path

from benchmarks.camera_registration_fixture_archive import ROOT, load_all, source_identities

MANIFEST_SHA256 = '00dde261bee03952bdc134b10f85c309f02869f00810ad967b8fd3ad4d862c54'
VERSION_PATH = 'src/sintergpu/data/VERSION'
FROZEN_VERSION = ROOT / 'qualification/evidence/vkl-camera-registration-frozen-version-0.2.6.txt'


def load_canonical(directory, *, root=ROOT, frozen_version=FROZEN_VERSION):
    directory, root = Path(directory), Path(root)
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('canonical camera fixture manifest digest mismatch')
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError('canonical camera fixture manifest digest mismatch')
    expected = json.loads(manifest_bytes)['profile']['source_sha256']
    current = source_identities(root)
    if set(current) != set(expected) or any(current[k] != expected[k] for k in current if k != VERSION_PATH):
        raise ValueError('frozen fixture executable source closure mismatch')
    frozen_version = Path(frozen_version)
    if frozen_version.is_symlink():
        raise ValueError('historical version must not be a symlink')
    version_bytes = frozen_version.read_bytes()
    if hashlib.sha256(version_bytes).hexdigest() != expected[VERSION_PATH]:
        raise ValueError('historical fixture version mismatch')
    with tempfile.TemporaryDirectory(prefix='sintergpu-camera-closure-') as temp:
        historical_root = Path(temp)
        for relative in expected:
            target = historical_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(version_bytes if relative == VERSION_PATH else (root / relative).read_bytes())
        # The strict loader hashes every copied source, checks the frozen donor,
        # payload identity, all archive bytes and output layouts before returning.
        return load_all(directory, root=historical_root)
