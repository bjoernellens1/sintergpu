"""Mutation-sensitive guards for camera byte layout and the fusion boundary."""
from pathlib import Path
import ast
import re
import pytest

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'src/sintergpu/kernels/image/camera_color.slang'
REFERENCE=ROOT/'src/sintergpu/references/color.py'


def violations(source):
    errors=[]
    if re.search(r'\.(Store|Store2|Store3|Store4)\s*\(',source): errors.append('unsafe packed word store')
    writer=source[source.index('void writeByte('):source.index('uint quantize(')]
    if 'InterlockedAnd' not in writer or 'InterlockedOr' not in writer: errors.append('missing race-free byte mask atomics')
    decoder=source[source.index('uint3 decodePixel('):source.index('uint3 readYUYV(')]
    if 'return quantizeRGB(' not in decoder: errors.append('decode must quantize before remap')
    fused=source[source.index('void fusedYUYVRemap('):]
    if fused.count('readYUYV(')!=4 or 'bilinear(' not in fused: errors.append('fused path must sample four decoded corners')
    if 'all(isfinite(coordinate))' not in source: errors.append('nonfinite maps need explicit black border')
    if re.search(r'for\s*\([^)]*Params\.',source): errors.append('unbounded work per output')
    return errors


def test_packed_shader_and_fusion_guards():
    assert violations(SHADER.read_text())==[]


@pytest.mark.parametrize('old,new',[
    ('buffer.InterlockedAnd(address & ~3u, ~mask, previous);','buffer.Store(address & ~3u, value);'),
    ('return quantizeRGB(float3(r,g,b));','return uint3(float3(r,g,b));'),
    ('if(!all(isfinite(coordinate))) return false;',''),
    ('readYUYV(lo),readYUYV(uint2(hi.x,lo.y))','uint3(0),readYUYV(uint2(hi.x,lo.y))'),
])
def test_guard_rejects_unsafe_mutations(old,new):
    source=SHADER.read_text(); assert old in source
    assert violations(source.replace(old,new))


def test_reference_module_never_owns_gpu_runtime():
    tree=ast.parse(REFERENCE.read_text())
    forbidden={'Device','create_device','create_buffer','create_command_encoder','submit','dispatch','to_numpy','readback'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=getattr(node.func,'attr',getattr(node.func,'id',''))
            assert name not in forbidden
