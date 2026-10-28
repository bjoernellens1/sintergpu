# SPDX-License-Identifier: Apache-2.0
"""Inspect actual compiler output and require the reviewed projection program."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import struct

# Core SPIR-V opcodes/decorations, stable in the Khronos unified specification.
OPERATIONS={129:'OpFAdd',133:'OpFMul',136:'OpFDiv',142:'OpVectorTimesScalar',144:'OpVectorTimesMatrix'}


def inspect(binary,assembly=None):
    if len(binary)<20 or len(binary)%4:raise ValueError('truncated SPIR-V')
    words=struct.unpack('<'+'I'*(len(binary)//4),binary)
    if words[0]!=0x07230203:raise ValueError('invalid SPIR-V magic')
    widths=set();operations=Counter();offset=5
    while offset<len(words):
        count=words[offset]>>16;opcode=words[offset]&0xffff
        if count==0 or offset+count>len(words):raise ValueError('invalid SPIR-V instruction extent')
        operands=words[offset+1:offset+count]
        if opcode==22:
            if len(operands)!=2 or operands[1]!=32:raise ValueError('unsupported float width for FP32 pixel model')
            widths.add(operands[1])
        if opcode in (71,72):
            index=1 if opcode==71 else 2
            if len(operands)<=index:raise ValueError('truncated precision decoration')
            if operands[index]==0:raise ValueError('RelaxedPrecision violates pixel model precision assumptions')
            if operands[index]==40:raise ValueError('fast-math decoration requires separate review')
        if opcode in OPERATIONS:operations[OPERATIONS[opcode]]+=1
        offset+=count
    if widths!={32} or any(operations[name]==0 for name in OPERATIONS.values()):raise ValueError('projection program lacks covered FP32 camera/perspective operations')
    if assembly is not None:
        if re.search(r'\b(RelaxedPrecision|FPFastMathMode|FPFastMathDefault)\b',assembly):raise ValueError('precision-reducing assembly mode requires separate review')
        text_counts=Counter(re.findall(r'\bOp[A-Za-z0-9_]+\b',assembly))
        if any(text_counts[name]!=operations[name] for name in OPERATIONS.values()):raise ValueError('binary/assembly operation counts differ')
    return dict(binary_sha256=hashlib.sha256(binary).hexdigest(),assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest() if assembly is not None else None,spirv_version=hex(words[1]),float_widths=sorted(widths),operations=dict(operations),relaxed_precision=False)


def require_reviewed_program(directory,expected_sha256):
    matches=[]
    for path in Path(directory).glob('*.spv'):
        binary=path.read_bytes()
        if hashlib.sha256(binary).hexdigest()==expected_sha256:matches.append(binary)
    if not matches:raise RuntimeError('actual compiler output does not match the reviewed projection SPIR-V; qualification pending')
    return inspect(matches[0])
