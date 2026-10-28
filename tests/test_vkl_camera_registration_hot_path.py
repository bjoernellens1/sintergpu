"""Compile-only ABI and semantic guards; GPU numerical parity is separate.

Set SINTERGPU_SLANGC_2026_12 and SINTERGPU_SLANGC_2026_19 to pinned
compiler executables, and SINTERGPU_SPIRV_DIS to spirv-dis. Without compiler
overrides only versioned executables in the project .tools directory are used.
SINTERGPU_CAMERA_COMPILER_VERSION limits a matrix job to its one pinned lane.
"""
import json
import os
import shutil
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / 'src/sintergpu/kernels/vkl/camera_registration.slang'
BROWN = ROOT / 'src/sintergpu/kernels/common/brown_conrady.slang'
ENTRIES = ('rectifyGrid', 'rectifyFromGrid', 'depthCameraRays', 'clearRegistration',
           'projectRegistration', 'registrationMin', 'registrationWinnerCount',
           'finalizeRegistration', 'finalizeRegistrationDebug')
VERSIONS = ('2026.12', '2026.19')
SELECTED_VERSION = os.environ.get('SINTERGPU_CAMERA_COMPILER_VERSION')
if SELECTED_VERSION and SELECTED_VERSION not in VERSIONS:
    raise ValueError(f'unsupported camera compiler version: {SELECTED_VERSION}')
COMPILERS = (SELECTED_VERSION,) if SELECTED_VERSION else VERSIONS


def _resolve_tool(env_name, default, description):
    """An explicit override is a requirement; only absent defaults may skip."""
    configured = env_name in os.environ
    value = os.environ[env_name] if configured else default
    path = Path(value) if value else None
    if not path or not path.is_file() or not os.access(path, os.X_OK):
        if configured:
            pytest.fail(f'{env_name} must name an executable file: {value!r}')
        pytest.skip(f'{description} unavailable: {value or "not found"}')
    return path


@pytest.fixture
def compiler(request):
    version = request.param
    env_name = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    default = ROOT / '.tools' / f'slang-{version}' / 'bin/slangc'
    path = _resolve_tool(env_name, default, f'pinned Slang {version} compiler')
    result = subprocess.run([str(path), '-version'], text=True, capture_output=True, check=True)
    actual = (result.stdout + result.stderr).strip()
    assert actual == version, f'{path} is not Slang {version}: {actual!r}'
    return path


@pytest.fixture
def disassembler():
    return _resolve_tool('SINTERGPU_SPIRV_DIS', shutil.which('spirv-dis'), 'spirv-dis')


@pytest.mark.parametrize('env_name', ('SINTERGPU_SLANGC_2026_12', 'SINTERGPU_SPIRV_DIS'))
@pytest.mark.parametrize('invalid', ('empty', 'missing', 'nonexecutable'))
def test_explicit_tool_override_must_be_executable(monkeypatch, tmp_path, env_name, invalid):
    path = tmp_path / 'tool'
    if invalid == 'nonexecutable':
        path.write_text('not executable')
        path.chmod(0o644)
    monkeypatch.setenv(env_name, '' if invalid == 'empty' else str(path))
    with pytest.raises(pytest.fail.Exception, match=env_name):
        _resolve_tool(env_name, None, 'tool')


@pytest.mark.parametrize('env_name', ('SINTERGPU_SLANGC_2026_12', 'SINTERGPU_SPIRV_DIS'))
def test_unconfigured_missing_tool_skips(monkeypatch, tmp_path, env_name):
    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(pytest.skip.Exception, match='unavailable'):
        _resolve_tool(env_name, tmp_path / 'missing', 'tool')


def test_source_guards():
    source = SHADER.read_text()
    brown = BROWN.read_text()
    assert 'brownForward' in brown and 'brownInverse' in brown
    assert re.search(r'for\s*\([^;]+;\s*\w+\s*<\s*5\s*;', brown)
    for name in ENTRIES:
        assert re.search(r'void\s+' + name + r'\s*\(', source)
    assert 'InterlockedMin' in source and 'InterlockedAdd' in source
    assert 'RWStructuredBuffer<uint2> winnerSourceIndexOut, collisionCountOut' in source
    assert 'uint2(winnerIndex[i],0u)' in source and 'uint2(0xffffffffu,0xffffffffu)' in source
    assert 'uint2(collisionCount[i],0u)' in source
    assert 'NoContraction' in source
    assert not any(x in source for x in ('cuda', 'to_numpy', 'submit', 'wait'))


@pytest.mark.parametrize('compiler', COMPILERS, ids=COMPILERS, indirect=True)
@pytest.mark.parametrize('entry', ENTRIES)
def test_compile_entry(compiler, entry, tmp_path):
    result = subprocess.run([str(compiler), str(SHADER), '-entry', entry, '-target', 'spirv',
                             '-profile', 'glsl_460', '-o', str(tmp_path / 'test.spv')],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('compiler', COMPILERS, ids=COMPILERS, indirect=True)
def test_reflection_matches_plans_and_int64_capabilities(compiler, disassembler, tmp_path):
    from sintergpu.camera_registration import camera_rectify_plan, depth_register_plan
    plans = [camera_rectify_plan(1,3,3,4,5,6,max_workgroups_x=100,distortion=True,debug=True),
             depth_register_plan(1,3,4,5,6,max_workgroups_x=100,depth_distortion=True,rgb_distortion=True,debug=True),
             depth_register_plan(1,3,4,5,6,max_workgroups_x=100)]
    stages = {stage['entrypoint']: stage for plan in plans for stage in plan['stages']}
    for name, stage in stages.items():
        spv, reflection = tmp_path / (name+'.spv'), tmp_path / (name+'.json')
        result = subprocess.run([str(compiler), str(SHADER), '-entry', name, '-target', 'spirv', '-profile', 'glsl_460', '-o', str(spv), '-reflection-json', str(reflection)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        data = json.loads(reflection.read_text())
        used = {v['name'] for v in data['entryPoints'][0]['bindings'] if v['binding'].get('used') == 1}
        assert used == set(stage['bindings']) | {stage['params_binding']}, (name, used, stage)
        param = next(v for v in data['parameters'] if v['name'] == stage['params_binding'])
        assert {x['name'] for x in param['type']['elementType']['fields']} == set(stage['params'])
        asm = subprocess.run([disassembler, str(spv)], text=True, capture_output=True, check=True).stdout
        assert 'OpCapability Int64' not in asm
        if name == 'finalizeRegistrationDebug':
            assert 'ArrayStride 8' in asm
            for binding in ('winnerSourceIndexOut', 'collisionCountOut'):
                typ = next(v for v in data['parameters'] if v['name'] == binding)['type']['resultType']
                assert typ['kind'] == 'vector' and typ['elementCount'] == 2
                assert typ['elementType']['scalarType'] == 'uint32'
