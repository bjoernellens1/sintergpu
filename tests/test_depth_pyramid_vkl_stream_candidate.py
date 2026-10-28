# SPDX-License-Identifier: Apache-2.0
"""ABI and SPIR-V checks only; GPU parity/performance are separate gates."""
import json
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'src/sintergpu/kernels/common'
SHADERS=(ROOT/'src/sintergpu/kernels/vkl/depth_pyramid.slang',
         ROOT/'benchmarks/shaders/depth_pyramid_vkl_candidate.slang',
         ROOT/'benchmarks/shaders/depth_pyramid_vkl_stream_candidate.slang')


def test_new_candidate_has_inline_per_sample_update_and_original_entry():
    assert SHADERS[2].is_file(), 'stream candidate missing'
    source=SHADERS[2].read_text()
    assert 'depthPyramidVklAccumulate' in source
    assert 'depthPyramidLevel' in source
    assert 'float d[4]' not in source and 'uint v[4]' not in source
    assert 'float c[4]' not in source
    assert not any(x in source for x in ('groupshared', 'Interlocked', 'Barrier'))


@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_original_and_candidates_have_identical_reflection_and_compile(version,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_')
    compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail(f'{key} unavailable')
        pytest.skip('pinned compiler unavailable')
    ver=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (ver.stdout+ver.stderr).strip()==version
    dis=os.getenv('SINTERGPU_SPIRV_DIS')
    if not dis: pytest.skip('disassembler not configured')
    reflection=[]; counts=[]
    for i,path in enumerate(SHADERS):
        spv=tmp_path/f'{i}.spv'; ref=tmp_path/f'{i}.json'
        run=subprocess.run([str(compiler),str(path),'-I',str(CORE),'-entry','depthPyramidLevel',
            '-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(ref)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        data=json.loads(ref.read_text())
        reflection.append((data['parameters'],data['entryPoints'][0]))
        asm=subprocess.run([dis,str(spv)],capture_output=True,text=True,check=True).stdout
        assert 'OpControlBarrier' not in asm and 'OpMemoryBarrier' not in asm and 'OpAtomic' not in asm
        if i == 2:
            assert 'OpFunctionCall' not in asm
            assert 'OpTypeArray' not in asm
        counts.append(sum(' Op' in line for line in asm.splitlines()))
    assert reflection[0]==reflection[1]==reflection[2]
    print(f'Slang {version} SPIR-V instruction counts (incumbent, array candidate, stream candidate): {counts}')
