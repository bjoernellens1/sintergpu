"""Offline source-application checks; no compile or runtime claim."""
import hashlib
import os
import tempfile
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path('/home/bjoern/git/eligsir-refactor-execution/quality-current-receipt-sdk-build-v1')
CASES = [
    ('rhi-source/include/slang-rhi.h', 'include/slang-rhi.h', 'rhi-allocation-observation-draft.patch', '516a4390911cae8b3ca93e9191ef145c39f232aaef80fa9f4c8c1ed2111f9111', '7ba61df188aae63f5dd1b86873a822472b9abd967bf78e5553be12679aa9b24c'),
    ('slangpy-source/src/slangpy_ext/device/resource.cpp', 'src/slangpy_ext/device/resource.cpp', 'slangpy-allocation-observation-draft.patch', 'a1d0082a7440f17d4a51c231fcc14ffb66088e8fdb869058d4a21aec735335d4', '31461ce90a1a9819b3d8508e0ff47a008e297e4296474aa07c38ada1a3e88e09'),
]

@pytest.mark.parametrize('original,target,patch,expected,expected_output', CASES)
def test_patch_applies_to_local_audited_source(original, target, patch, expected, expected_output):
    source = AUDIT / original
    if not source.is_file():
        pytest.skip('local-only source application check: audited source absent; not CI proof')
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    patch_path = ROOT / 'tools/diagnostics/patches' / patch
    env = os.environ.copy()
    env.pop('GIT_DIR', None)
    env.pop('GIT_WORK_TREE', None)
    env.pop('GIT_INDEX_FILE', None)
    env['GIT_CEILING_DIRECTORIES'] = '/tmp'
    with tempfile.TemporaryDirectory(prefix='sinter-patch-', dir='/tmp') as directory:
        dest = Path(directory) / target
        dest.parent.mkdir(parents=True)
        dest.write_bytes(data)
        subprocess.run(['git', 'apply', '--check', str(patch_path)], cwd=directory, env=env, check=True)
        subprocess.run(['git', 'apply', str(patch_path)], cwd=directory, env=env, check=True)
        result = dest.read_bytes()
        assert result != data
        assert hashlib.sha256(result).hexdigest() == expected_output
        if patch.startswith('rhi'):
            assert b'class IVulkanBufferAllocationObservationV1 : public ISlangUnknown' in result
        else:
            assert b'vulkan_allocation_observation_v1' in result
            assert b'return nb::none();' in result
        dest.write_bytes(data.replace(data[:50], b'mismatched-source', 1))
        assert subprocess.run(['git', 'apply', '--check', str(patch_path)], cwd=directory, env=env, capture_output=True).returncode != 0
