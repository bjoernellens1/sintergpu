"""Mutation-sensitive source guards for the experimental VKL fast paths.

These catch accidental ABI/dispatch regressions; numerical tests remain separate.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT/'src/sintergpu/kernels/vkl'


def guard_projection(source):
    for name in ('visible', 'provisional', 'depthCandidate'):
        assert re.search(r'StructuredBuffer<uint>\s+' + name + r'\s*;', source), name
    for name in ('eligibleOut','projectedOut','sensorOut','supportOut','conflictOut','residualOut'):
        assert len(re.findall(r'\b'+name+r'\[i\]\s*=',source)) == 1, name
    assert 'if (i >= params.batches * params.points) return;' in source
    assert 'if (projected)' in source
    assert 'ru < float(params.width)' in source and 'rv < float(params.height)' in source
    assert 'roundEven(' in source and 'fraction == 0.5' in source
    assert 'residual < params.supportThreshold' in source
    assert 'z < lookup-params.conflictThreshold' in source
    # No barriers, atomics, workgroup arrays, or point loops in this lane-local operation.
    assert not re.search(r'\b(for|while|groupshared|Interlocked\w*|GroupMemoryBarrier\w*)\b', re.sub(r'//[^\n]*','',source))


def test_projection_guard_and_mutations():
    source = (KERNELS/'project_depth_evidence.slang').read_text()
    guard_projection(source)
    for old,new in [('StructuredBuffer<uint> visible', 'StructuredBuffer<float> visible'),
                    ('eligibleOut[i] = uint(eligible);', ''),
                    ('if (projected)', 'if (true)'),
                    ('residual < params.supportThreshold', 'residual <= params.supportThreshold'),
                    ('ru < float(params.width)', 'true')]:
        assert old in source
        with pytest.raises(AssertionError):
            guard_projection(source.replace(old,new))


def test_no_product_host_or_hidden_layout_work():
    # Product VKL ports are shader resources only; host allocations/copies exist
    # solely in the explicit CPU oracle and opt-in numerical/benchmark harness.
    shaders = list(KERNELS.glob('*.slang'))
    assert len(shaders) >= 4
    for path in shaders:
        source = re.sub(r'//[^\n]*','',path.read_text())
        assert re.search(r'\[numthreads\(\s*256\s*,\s*1\s*,\s*1\s*\)\]', source)
        assert not re.search(r'\b(Device|create_buffer|submit|to_numpy|contiguous|wait_for_idle)\b', source)
        assert 'ConstantBuffer<' in source
    isp = (KERNELS/'photometric_isp.slang').read_text()
    assert 'c<3' in isp and 'params.useColor' in isp
    pyramid = (KERNELS/'depth_pyramid.slang').read_text()
    assert 'dy<2' in pyramid and 'dx<2' in pyramid and 'count==4' in pyramid


def guard_geometry_depth(source,geometry=False):
    source=re.sub(r'//[^\n]*','',source)
    assert not re.search(r'\b(while|do|groupshared|Interlocked\w*|GroupMemoryBarrier\w*)\b',source)
    loops=re.findall(r'for\s*\(([^)]*)\)',source)
    for loop in loops:
        assert re.fullmatch(r'uint\s+(\w+)\s*=\s*0;\s*\1\s*<\s*'+(r'3' if geometry else r'params\.sources')+r';\s*\1\+\+',loop),loop
    assert 'if(i>=params.batches*plane)return;' in source if geometry else 'if(i>=params.count)return;' in source
    assert len(loops)<=(2 if geometry else 7), 'unexpected loop nesting/extra full-field passes'
    if not geometry:
        for name in ('support_countOut','temporal_support_countOut','depth_varianceOut','conflict_owner_sourceOut','dominant_contributorOut','temporal_weight_fractionOut','source_disagreementOut'):
            assert len(re.findall(r'\b'+name+r'\[i\]\s*=',source))==1,name


@pytest.mark.parametrize('filename,geometry',[('geometry.slang',True),('depth_fusion.slang',False)])
def test_geometry_depth_guards_reject_image_loops_and_missing_stores(filename,geometry):
    source=(KERNELS/filename).read_text();guard_geometry_depth(source,geometry)
    loop='c<3' if geometry else 's<params.sources'
    for changed in (source.replace(loop,loop.split('<')[0]+'<params.count',1),source+'\ngroupshared float scratch[256];',source.replace('if(i>=params.batches*plane)return;' if geometry else 'if(i>=params.count)return;','')):
        with pytest.raises(AssertionError):guard_geometry_depth(changed,geometry)
    if not geometry:
        with pytest.raises(AssertionError):guard_geometry_depth(source.replace('support_countOut[i]=support;',''),False)
