# SPDX-License-Identifier: Apache-2.0
"""CPU models and actual compiler reflection; no GPU execution."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from sintergpu.compact_backward_plan import plan_backward, SENTINEL
from sintergpu.references.compact_backward import compact_value_backward
from test_vkl_compact_backward_reference import donor

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'benchmarks/shaders/vkl_compact_backward_candidate.slang'
ENTRIES={'backwardInit':{'params','inverse'},'backwardInverse':{'params','slotSource','inverse'},'backwardGather':{'params','inverse','gradWords','outputWords'}}

@pytest.mark.parametrize('entry',ENTRIES)
@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_compiler_reflection(version,entry,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_')
    compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail('explicit compiler missing')
        pytest.skip('compiler unavailable')
    identity=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (identity.stdout+identity.stderr).strip()==version
    refl=tmp_path/'reflection.json'; spv=tmp_path/'shader.spv'
    process=subprocess.run([str(compiler),str(SHADER),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    data=json.loads(refl.read_text()); params=next(x for x in data['parameters'] if x['name']=='params')
    fields=params['type']['elementType']['fields']
    assert [x['name'] for x in fields]==['batches','items','capacity','bytesPerItem','logicalBytes','gradBase']
    assert [x['type']['scalarType'] for x in fields]==['uint32']*6
    assert [x['binding']['offset'] for x in fields]==list(range(0,24,4))
    ep=data['entryPoints'][0]; assert ep['threadGroupSize']==[256,1,1]
    assert {b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}==ENTRIES[entry]
    dis=os.environ.get('SINTERGPU_SPIRV_DIS')
    if dis:
        asm=subprocess.run([dis,str(spv)],capture_output=True,text=True,check=True).stdout
        assert 'OpAtomic' not in asm and 'OpControlBarrier' not in asm


def model(plan,slots,grad):
    """Independent host interpreter of specified word ownership and inverse passes."""
    p=plan['params']; b=p['batches']; n=p['items']; cap=p['capacity']; size=p['bytesPerItem']
    inverse=[SENTINEL]*(b*n)
    for slot,source in enumerate(slots):
        if source!=SENTINEL:
            assert source<n and inverse[(slot//cap)*n+source]==SENTINEL
            inverse[(slot//cap)*n+source]=slot
    logical=p['logicalBytes']; output=bytearray((logical+3)//4*4)
    for word in range(len(output)//4):
        for lane in range(4):
            pos=word*4+lane
            if pos>=logical: continue
            slot=inverse[pos//size]
            if slot!=SENTINEL: output[pos]=grad[p['gradBase']+slot*size+pos%size]
    return output,inverse

@pytest.mark.parametrize('width',[1,2,3,4,8,16])
@pytest.mark.parametrize('capacity',[0,1,3,7])
def test_word_model_exact_bits_and_padding(width,capacity):
    shape=(2,4,3); payload=3*width; size=((1+2*capacity*payload+3)//4)*4
    plan=plan_backward(shape,capacity,width,grad_base=1,grad_storage_bytes=size,max_buffer_bytes=4096,max_workgroups_x=100,max_group_size_x=256,max_threads_per_group=256)
    mask=np.array([[1,0,1,1],[0,1,1,1]],bool); slots=[SENTINEL]*(2*capacity)
    for b in range(2):
        for rank,index in enumerate(np.flatnonzero(mask[b])[:capacity]): slots[b*capacity+rank]=int(index)
    raw=np.arange(size,dtype=np.uint8)*np.uint8(73)
    result,inverse=model(plan,slots,raw)
    expected=np.zeros((2,4,payload),np.uint8)
    for b in range(2):
        for rank,index in enumerate(np.flatnonzero(mask[b])[:capacity]):
            start=1+(b*capacity+rank)*payload; expected[b,index]=raw[start:start+payload]
    assert result[:expected.size]==expected.tobytes()
    assert not any(result[expected.size:])
    assert len(inverse)==8

@pytest.mark.parametrize('capacity',[0,2,5])
@pytest.mark.parametrize('dtype,torch_dtype',[(np.float16,'float16'),(np.float32,'float32'),(np.float64,'float64'),(np.complex64,'complex64'),(np.complex128,'complex128')])
def test_model_logical_gradient_matches_torch(donor,capacity,dtype,torch_dtype):
    torch,select=donor; shape=(2,4,3); mask=np.array([[1,0,1,1],[0,1,1,1]],bool)
    width=np.dtype(dtype).itemsize; upstream=np.arange(2*capacity*3,dtype=np.float64).astype(dtype).reshape(2,capacity,3)
    if np.dtype(dtype).kind=='c': upstream+=1j*upstream
    values=torch.zeros(shape,dtype=getattr(torch,torch_dtype),requires_grad=True)
    select(values,torch.from_numpy(mask),capacity=capacity)['values'].backward(torch.from_numpy(upstream))
    size=((1+upstream.nbytes+3)//4)*4
    raw=bytearray(size); raw[1:1+upstream.nbytes]=upstream.tobytes()
    slots=[SENTINEL]*(2*capacity)
    for b in range(2):
        for rank,index in enumerate(np.flatnonzero(mask[b])[:capacity]): slots[b*capacity+rank]=int(index)
    p=plan_backward(shape,capacity,width,grad_base=1,grad_storage_bytes=size,max_buffer_bytes=4096,max_workgroups_x=100,max_group_size_x=256,max_threads_per_group=256)
    output,_=model(p,slots,raw)
    logical=np.frombuffer(output,dtype=dtype,count=np.prod(shape)).reshape(shape)
    np.testing.assert_array_equal(logical,values.grad.numpy(),strict=True)
    np.testing.assert_array_equal(logical,compact_value_backward(shape,mask,upstream),strict=True)


def test_plan_empty_bounds_and_overflow():
    kw=dict(grad_base=0,grad_storage_bytes=0,max_buffer_bytes=4096,max_workgroups_x=100,max_group_size_x=256,max_threads_per_group=256)
    plan=plan_backward((2,0,3),3,4,**kw)
    assert plan['stages']==[] and all(x>=4 for x in plan['buffers'].values())
    plan=plan_backward((2,4,3),0,4,**kw)
    assert [s['entry'] for s in plan['stages']]==['backwardInit','backwardGather']
    with pytest.raises(ValueError): plan_backward((2,0xffffffff,3),1,4,**kw)
    with pytest.raises(ValueError): plan_backward((2,4,3),1,4,**kw)


def test_opaque_nonfinite_and_signed_zero_bits_preserved():
    # Opaque payload comparison preserves NaN payloads without interpreting floats.
    bits=np.array([0x7fc12345,0xffc23456,0x80000000,0x7f800000],dtype='<u4')
    raw=bytearray(1+bits.nbytes+3); raw[1:1+bits.nbytes]=bits.tobytes()
    plan=plan_backward((1,4),4,4,grad_base=1,grad_storage_bytes=len(raw),max_buffer_bytes=4096,max_workgroups_x=100,max_group_size_x=256,max_threads_per_group=256)
    slots=[0,1,2,3]; original=raw[:]; output,_=model(plan,slots,raw)
    assert output==bits.tobytes() and raw==original and slots==[0,1,2,3]


def test_shader_has_word_owned_output_and_no_internal_sync():
    source=SHADER.read_text()
    assert 'outputWords[word]=packed;' in source
    assert 'Interlocked' not in source and 'GroupMemoryBarrier' not in source
    assert 'inverse[(slot/params.capacity)*params.items+source]=slot;' in source
