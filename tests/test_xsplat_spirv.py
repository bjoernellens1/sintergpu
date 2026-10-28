"""The pixel model must fail closed if compiler precision assumptions change."""
import importlib,importlib.util,struct
from pathlib import Path
import pytest


def inspector():
    assert importlib.util.find_spec('benchmarks.xsplat_spirv'), 'measured-program precision inspector missing'
    return importlib.import_module('benchmarks.xsplat_spirv')


def artifact():
    root=Path(__file__).resolve().parents[1]/'qualification/evidence'
    return (root/'xsplat-project-program-0.2.3.spv').read_bytes(),(root/'xsplat-project-program-0.2.3.spv-asm').read_text()


def test_actual_runtime_program_has_covered_fp32_operations_and_no_relaxed_precision():
    m=inspector();binary,assembly=artifact();result=m.inspect(binary,assembly)
    assert result['float_widths']==[32]
    assert result['operations']['OpVectorTimesMatrix']==1
    assert result['operations']['OpFDiv']>=1
    assert len(result['binary_sha256'])==64


def test_precision_decorations_and_float_width_changes_are_rejected():
    m=inspector();binary,assembly=artifact();words=list(struct.unpack('<'+'I'*(len(binary)//4),binary))
    # OpDecorate target=1, RelaxedPrecision=0 or FPFastMathMode=40.
    for decoration in (0,40):
        mutated=words+[(3<<16)|71,1,decoration]
        if decoration==40:mutated[-3]=(4<<16)|71;mutated.append(1)
        with pytest.raises(ValueError,match='precision|fast-math'):m.inspect(struct.pack('<'+'I'*len(mutated),*mutated),assembly)
    offset=5
    while offset<len(words):
        count=words[offset]>>16;opcode=words[offset]&0xffff
        if opcode==22:
            mutated=words.copy();mutated[offset+2]=16
            with pytest.raises(ValueError,match='float width'):m.inspect(struct.pack('<'+'I'*len(mutated),*mutated),assembly)
            break
        offset+=count
    else:pytest.fail('retained program lacks float type')


def test_truncated_program_and_unreviewed_identity_are_rejected(tmp_path):
    m=inspector();binary,assembly=artifact()
    with pytest.raises(ValueError):m.inspect(binary[:-3],assembly)
    path=tmp_path/'unexpected.spv';path.write_bytes(binary+b'\x00\x00\x00\x00')
    with pytest.raises(RuntimeError,match='reviewed'):m.require_reviewed_program(tmp_path,m.inspect(binary,assembly)['binary_sha256'])
    path.write_bytes(binary)
    assert m.require_reviewed_program(tmp_path,m.inspect(binary,assembly)['binary_sha256'])['binary_sha256']==m.inspect(binary,assembly)['binary_sha256']
