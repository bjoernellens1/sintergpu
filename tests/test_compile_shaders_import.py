"""The catalog compiler must resolve packaged shader imports."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('version', ('2026.12', '2026.19'))
def test_catalog_compiler_resolves_packaged_import(version, tmp_path):
    key = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    value = os.environ.get(key)
    compiler = Path(value) if value else ROOT / '.tools' / ('slang-' + version) / 'bin/slangc'
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        if key in os.environ: pytest.fail(f'invalid compiler override: {key}')
        pytest.skip(f'Slang {version} unavailable')
    info = subprocess.run([str(compiler), '-version'], capture_output=True, text=True, check=True)
    assert (info.stdout + info.stderr).strip() == version
    output = tmp_path / 'compile'
    command = [sys.executable, str(ROOT / 'tools/compile_shaders.py'), '--slangc', str(compiler), '--operation', 'optimization.point_plane_aosoa_256.main', '--output', str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / 'results.json').read_text())
    assert data['status'] == 'passed' and len(data['entries']) == 1
    assert data['entries'][0]['operation'] == 'optimization.point_plane_aosoa_256.main'
    assert data['entries'][0]['reflection']['entryPoints'][0]['threadGroupSize'] == [256, 1, 1]
