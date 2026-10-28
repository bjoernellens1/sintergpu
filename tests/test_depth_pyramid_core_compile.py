"""Compile-only checks for additive shared block policies; no GPU qualification."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src/sintergpu/kernels/common/depth_pyramid_core.slang'
WRAPPER = r'''
#include "depth_pyramid_core.slang"
struct Params { float threshold; };
ConstantBuffer<Params> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
RWStructuredBuffer<float> outDepth;
RWStructuredBuffer<uint> outValid;
RWStructuredBuffer<float> outConfidence;
[shader("compute")][numthreads(256,1,1)]
void vklCore(uint3 id : SV_DispatchThreadID) {
    float d[4]; uint v[4]; float c[4];
    for (uint i=0; i<4; ++i) { d[i]=depth[id.x*4+i]; v[i]=valid[id.x*4+i]; c[i]=confidence[id.x*4+i]; }
    DepthPyramidVklResult r = depthPyramidVklBlock(d, v, c, params.threshold);
    outDepth[id.x]=r.depth; outValid[id.x]=r.valid; outConfidence[id.x]=r.confidence;
}
[shader("compute")][numthreads(16,16,1)]
void slamCore(uint3 id : SV_DispatchThreadID) {
    float d[4]; for (uint i=0; i<4; ++i) d[i]=depth[id.x*4+i];
    outDepth[id.x]=depthPyramidSlamBlock(d, params.threshold);
}
'''


def test_core_is_resource_free_and_exposes_distinct_static_policies():
    assert CORE.is_file(), 'shared depth pyramid core is missing'
    source = CORE.read_text()
    for name in ('depthPyramidVklBlock', 'depthPyramidSlamBlock', 'DepthPyramidVklResult'):
        assert name in source
    for forbidden in ('StructuredBuffer', 'RWStructuredBuffer', 'groupshared', 'Barrier', 'Interlocked', 'SV_DispatchThreadID'):
        assert forbidden not in source


@pytest.mark.parametrize('version', ['2026.12', '2026.19'])
@pytest.mark.parametrize('entry,workgroup,bindings', [
    ('vklCore', [256, 1, 1], {'params', 'depth', 'valid', 'confidence', 'outDepth', 'outValid', 'outConfidence'}),
    ('slamCore', [16, 16, 1], {'params', 'depth', 'outDepth'}),
])
def test_static_policy_compiles_and_reflects(version, entry, workgroup, bindings, tmp_path):
    key = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    path = Path(os.environ.get(key, ROOT / '.tools' / ('slang-' + version) / 'bin/slangc'))
    if not path.is_file():
        if key in os.environ:
            pytest.fail(f'{key} is not an executable compiler: {path}')
        pytest.skip(f'Slang {version} unavailable')
    identity = subprocess.run([str(path), '-version'], capture_output=True, text=True, check=True)
    assert (identity.stdout + identity.stderr).strip() == version
    wrapper = tmp_path / 'wrapper.slang'
    wrapper.write_text(WRAPPER)
    spv = tmp_path / 'core.spv'
    reflection = tmp_path / 'reflection.json'
    proc = subprocess.run([str(path), str(wrapper), '-I', str(CORE.parent), '-entry', entry,
                           '-target', 'spirv', '-profile', 'glsl_460', '-o', str(spv),
                           '-reflection-json', str(reflection)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    reflected = json.loads(reflection.read_text())
    info = reflected['entryPoints'][0]
    assert info['threadGroupSize'] == workgroup
    used = {b['name'] for b in info['bindings'] if b['binding'].get('used') == 1}
    assert used == bindings
    parameters = {p['name']: p for p in reflected['parameters']}
    assert parameters['params']['binding']['index'] == 0
    fields = parameters['params']['type']['elementType']['fields']
    assert [(f['name'], f['type']['scalarType'], f['binding']['offset']) for f in fields] == [('threshold', 'float32', 0)]
    indices = {'depth': 1, 'valid': 2, 'confidence': 3, 'outDepth': 4, 'outValid': 5, 'outConfidence': 6}
    for name in bindings - {'params'}:
        param = parameters[name]
        assert param['binding']['index'] == indices[name]
        typ = param['type']
        assert typ['baseShape'] == 'structuredBuffer'
        assert typ['resultType']['scalarType'] == ('uint32' if name in {'valid', 'outValid'} else 'float32')
        assert (typ.get('access') == 'readWrite') == name.startswith('out')
    dis = os.environ.get('SINTERGPU_SPIRV_DIS') or shutil.which('spirv-dis')
    if not dis:
        pytest.skip('SPIR-V disassembler unavailable')
    asm = subprocess.run([dis, str(spv)], capture_output=True, text=True, check=True).stdout
    assert 'OpEntryPoint GLCompute' in asm
    assert 'OpControlBarrier' not in asm and 'OpMemoryBarrier' not in asm
    assert 'OpAtomic' not in asm
    # Distinct entry resources prove no runtime policy selector or cross-policy resources.
    assert 'policy' not in {b['name'].lower() for b in info['bindings']}
