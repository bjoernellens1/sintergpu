"""Compile-only ABI and semantic guards; GPU numerical parity is separate."""
import json
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
COMPILERS = (Path('/home/bjoern/.codex/worktrees/kernel-consolidation/sintergpu/.tools/slang-2026.12/bin/slangc'),
             Path('/home/bjoern/.local/opt/slang-2026.19/bin/slangc'))


def test_source_guards():
    source = SHADER.read_text()
    brown = BROWN.read_text()
    assert 'brownForward' in brown and 'brownInverse' in brown
    assert re.search(r'for\s*\([^;]+;\s*\w+\s*<\s*5\s*;', brown)
    for name in ENTRIES:
        assert re.search(r'void\s+' + name + r'\s*\(', source)
    assert 'InterlockedMin' in source and 'InterlockedAdd' in source
    assert 'RWStructuredBuffer<int64_t>' in source
    assert 'NoContraction' in source
    assert not any(x in source for x in ('cuda', 'to_numpy', 'submit', 'wait'))


@pytest.mark.parametrize('compiler', COMPILERS, ids=('2026.12', '2026.19'))
@pytest.mark.parametrize('entry', ENTRIES)
def test_compile_entry(compiler, entry, tmp_path):
    assert compiler.is_file(), f'pinned compiler missing: {compiler}'
    result = subprocess.run([str(compiler), str(SHADER), '-entry', entry, '-target', 'spirv',
                             '-profile', 'glsl_460', '-o', str(tmp_path / 'test.spv')],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('compiler', COMPILERS, ids=('2026.12', '2026.19'))
def test_reflection_matches_plans_and_int64_capabilities(compiler, tmp_path):
    from sintergpu.camera_registration import camera_rectify_plan, depth_register_plan
    plans = [camera_rectify_plan(1,3,3,4,5,6,max_workgroups_x=100,distortion=True,debug=True),
             depth_register_plan(1,3,4,5,6,max_workgroups_x=100,depth_distortion=True,rgb_distortion=True,debug=True),
             depth_register_plan(1,3,4,5,6,max_workgroups_x=100)]
    stages = {stage['entrypoint']: stage for plan in plans for stage in plan['stages']}
    disassembler = '/home/bjoern/.local/opt/locus-spirv-tools/usr/bin/spirv-dis'
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
        assert ('OpCapability Int64' in asm) == (name == 'finalizeRegistrationDebug')
