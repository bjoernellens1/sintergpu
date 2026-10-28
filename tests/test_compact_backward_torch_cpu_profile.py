# SPDX-License-Identifier: Apache-2.0
"""Frozen Torch CPU observed bit profile; compilation is not GPU parity."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from sintergpu.compact_backward_torch_cpu_profile import DTYPES,plan_torch_cpu_profile,torch_cpu_profile_bits
from sintergpu.references.compact_backward import compact_value_backward
from test_vkl_compact_backward_reference import donor

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'benchmarks/shaders/vkl_compact_backward_torch_cpu_profile.slang'
ENTRIES={'backwardProfileInit':{'params','inverse'},'backwardProfileInverse':{'params','slotSource','inverse'},'backwardProfileGather':{'params','inverse','gradWords','outputWords'}}

@pytest.mark.parametrize('version',['2026.12','2026.19'])
@pytest.mark.parametrize('entry',ENTRIES)
def test_compiles_and_reflects(version,entry,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_'); compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail('explicit compiler missing')
        pytest.skip('compiler unavailable')
    result=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True); assert (result.stdout+result.stderr).strip()==version
    refl=tmp_path/'reflection.json'; spv=tmp_path/'shader.spv'
    result=subprocess.run([str(compiler),str(SHADER),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    data=json.loads(refl.read_text()); fields=next(x for x in data['parameters'] if x['name']=='params')['type']['elementType']['fields']
    assert [x['name'] for x in fields]==['batches','items','capacity','bytesPerItem','logicalBytes','gradBase','componentBytes']
    assert [x['type']['scalarType'] for x in fields]==['uint32']*7
    assert [x['binding']['offset'] for x in fields]==list(range(0,28,4))
    ep=data['entryPoints'][0]; assert ep['threadGroupSize']==[256,1,1]
    assert {b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}==ENTRIES[entry]
    assert spv.stat().st_size>0


def bits_of_torch_result(donor,dtype,upstream,mask):
    torch,select=donor; b,n=mask.shape; capacity=upstream.shape[1]
    values=torch.zeros((b,n),dtype=getattr(torch,dtype),requires_grad=True)
    select(values,torch.from_numpy(mask),capacity=capacity)['values'].backward(torch.from_numpy(upstream))
    return values.grad.numpy().tobytes()


def compare(donor,dtype,upstream,mask):
    component=DTYPES[dtype][1]
    logical=compact_value_backward(mask.shape,mask,upstream)
    assert torch_cpu_profile_bits(logical.tobytes(),component)==bits_of_torch_result(donor,dtype,upstream,mask)


def test_exhaustive_float16_components(donor):
    # All 65,536 IEEE encodings, including both signed zeros, NaN payloads and infinities.
    upstream=np.arange(65536,dtype=np.uint16).view(np.float16).reshape(1,-1)
    compare(donor,'float16',upstream,np.ones((1,65536),bool))

@pytest.mark.parametrize('dtype',['float16','float32','float64','complex64','complex128'])
@pytest.mark.parametrize('capacity',[0,1,3,7])
@pytest.mark.parametrize('strided',[False,True])
def test_special_random_bits_capacity_and_upstream_strides(donor,dtype,capacity,strided):
    torch,_=donor; width,component=DTYPES[dtype]; b,n=2,5
    rng=np.random.default_rng(410); raw=rng.integers(0,256,size=b*capacity*width*2,dtype=np.uint8)
    # Keep float bit patterns entirely intact, including signaling NaNs.
    if capacity:
        special={2:[0x0000,0x8000,0x7c00,0xfc00,0x7c01,0xfc01,0x7e55,0xfe55],
                 4:[0,0x80000000,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc12345,0xffc12345],
                 8:[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff0000000000001,0xfff0000000000001,0x7ff8123456789abc,0xfff8123456789abc]}[component]
        for i,bits in enumerate(special[:len(raw)//component]): raw[i*component:(i+1)*component]=np.frombuffer(bits.to_bytes(component,'little'),np.uint8)
    array=raw.view(np.dtype(dtype)).reshape(b,capacity*2)
    upstream=array[:,::2] if strided else array[:,:capacity].copy()
    mask=np.array([[1,0,1,1,1],[0,1,1,0,1]],bool)
    compare(donor,dtype,upstream,mask)


def test_plan_rejects_unsupported_and_bounds():
    kw=dict(grad_base=1,grad_storage_bytes=4,max_buffer_bytes=4096,max_workgroups_x=100,max_threads_per_group=256,max_group_size_x=256)
    with pytest.raises(ValueError): plan_torch_cpu_profile((1,3),1,'int32',**kw)
    with pytest.raises(ValueError): plan_torch_cpu_profile((1,3),1,'float32',**kw)
    kw['grad_storage_bytes']=8
    p=plan_torch_cpu_profile((1,3),1,'float32',**kw)
    assert p['params']['componentBytes']==4
    assert [s['entry'] for s in p['stages']]==list(ENTRIES)
    assert all(s['barrier_before'] for s in p['stages'])


def test_source_word_ownership_and_byte_profile():
    s=SHADER.read_text(); assert 'outputWords[word]=packed;' in s
    assert 'Interlocked' not in s and 'GroupMemoryBarrier' not in s


def shader_byte_model(component):
    """Host simulation of the shader's separate high/low words and lane writes."""
    c=len(component); low=int.from_bytes(component[:4],'little'); high=int.from_bytes(component[4:],'little')
    if c==2: negative_zero=low==0x8000; isnan=low&0x7c00==0x7c00 and low&0x3ff!=0
    elif c==4: negative_zero=low==0x80000000; isnan=low&0x7f800000==0x7f800000 and low&0x7fffff!=0
    else: negative_zero=high==0x80000000 and low==0; isnan=high&0x7ff00000==0x7ff00000 and ((high&0xfffff)!=0 or low!=0)
    out=bytearray(component)
    if negative_zero: out[c-1]=0
    if isnan: out[{2:1,4:2,8:6}[c]]|={2:2,4:64,8:8}[c]
    return bytes(out)

@pytest.mark.parametrize('component',[2,4,8])
def test_byte_lane_model_against_independent_profile(component):
    rng=np.random.default_rng(29+component)
    values=range(65536) if component==2 else []
    if component!=2:
        raw=rng.integers(0,256,size=(4096,component),dtype=np.uint8)
        values=[int.from_bytes(row.tobytes(),'little') for row in raw]
        values += ([0x80000000,0x7f800001,0xffc12345] if component==4 else [0x8000000000000000,0x7ff0000000000001,0xfff8123456789abc])
    for bits in values:
        src=bits.to_bytes(component,'little')
        assert shader_byte_model(src)==torch_cpu_profile_bits(src,component)

@pytest.mark.parametrize('dtype',['float16','float32','float64','complex64','complex128'])
@pytest.mark.parametrize('strides',[(5,1),(10,2),(0,1),(2,1)])
def test_logical_profile_for_strided_and_overlapping_donor_values(donor,dtype,strides):
    torch,select=donor; width,component=DTYPES[dtype]
    base=torch.zeros(20,dtype=getattr(torch,dtype),requires_grad=True)
    values=torch.as_strided(base,(2,5),strides)
    mask=np.array([[1,0,1,1,0],[0,1,1,0,1]],bool)
    raw=np.zeros((2,3*width),np.uint8)
    if component==2: special=[0x8000,0x7c01,0xfe55]
    elif component==4: special=[0x80000000,0x7f800001,0xffc12345]
    else: special=[0x8000000000000000,0x7ff0000000000001,0xfff8123456789abc]
    for batch in range(2):
        for i in range(min(3*width//component,len(special))):
            raw[batch,i*component:(i+1)*component]=np.frombuffer(special[i].to_bytes(component,'little'),np.uint8)
    upstream=raw.view(dtype).reshape(2,3)
    out=select(values,torch.from_numpy(mask),capacity=3)['values']
    actual=torch.autograd.grad(out,values,torch.from_numpy(upstream))[0].numpy().tobytes()
    expected=torch_cpu_profile_bits(compact_value_backward((2,5),mask,upstream).tobytes(),component)
    assert actual==expected
