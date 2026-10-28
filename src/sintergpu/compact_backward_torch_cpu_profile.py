# SPDX-License-Identifier: Apache-2.0
"""Experimental frozen VKL Torch CPU-observed gradient-bit profile.

For each floating component, an upstream negative zero is changed to positive
zero and signaling NaNs are quieted while retaining sign and payload. This is
CPU-observed behavior, not a guarantee for GPU execution or other Torch builds.
Outputs are dense logical gradients; no view accumulation or adapter is supplied.
"""
from .compact_backward_plan import plan_backward

DTYPES={'float16':(2,2),'float32':(4,4),'float64':(8,8),'complex64':(8,4),'complex128':(16,8)}


def plan_torch_cpu_profile(shape,capacity,dtype,**kwargs):
    if dtype not in DTYPES: raise ValueError('unsupported profile dtype')
    item,component=DTYPES[dtype]
    plan=plan_backward(shape,capacity,item,**kwargs)
    plan['params']['componentBytes']=component
    for stage in plan['stages']:
        stage['entry']=stage['entry'].replace('backward','backwardProfile')
        stage['source']='benchmarks/shaders/vkl_compact_backward_torch_cpu_profile.slang'
    return plan


def component_bits_plus_zero(bits,component_bytes):
    """Pure integer CPU model for a single IEEE component bit pattern."""
    if component_bytes not in (2,4,8): raise ValueError('unsupported component')
    if type(bits) is not int or not 0<=bits<1<<(8*component_bytes): raise ValueError('component bits out of range')
    sign=1<<(8*component_bytes-1)
    mantissa_bits={2:10,4:23,8:52}[component_bytes]
    exponent_bits={2:5,4:8,8:11}[component_bytes]
    exponent=((1<<exponent_bits)-1)<<mantissa_bits; fraction=(1<<mantissa_bits)-1
    if bits==sign: return 0
    if bits&exponent==exponent and bits&fraction: return bits|(1<<(mantissa_bits-1))
    return bits


def torch_cpu_profile_bits(logical,component_bytes):
    """Transform a byte buffer componentwise, retaining NaN signs and payloads."""
    if component_bytes not in (2,4,8): raise ValueError('unsupported component')
    source=bytes(logical)
    if len(source)%component_bytes: raise ValueError('incomplete component')
    out=bytearray(len(source))
    for start in range(0,len(source),component_bytes):
        bits=int.from_bytes(source[start:start+component_bytes],'little')
        out[start:start+component_bytes]=component_bits_plus_zero(bits,component_bytes).to_bytes(component_bytes,'little')
    return bytes(out)
