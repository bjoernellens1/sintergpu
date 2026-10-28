# SPDX-License-Identifier: Apache-2.0
"""Frozen donor and independent models; compile-only, no GPU execution."""
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import numpy as np
import pytest
from sintergpu.references import caspar_layout_numeric as reference
from sintergpu.caspar_layout_plan import plan_layout

ROOT=Path(__file__).resolve().parents[1]
DONOR=ROOT/'references/donors/caspar-slang/src/caspar_slang/layout.py'
SHADER=ROOT/'src/sintergpu/kernels/experimental/caspar_layout.slang'
LIMITS={'max_buffer_bytes':1<<20,'max_workgroups_x':65535,'max_threads_per_group':256,'max_group_size_x':256}

@pytest.fixture(scope='module')
def donor():
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest()=='d32325aaefaa37df324645683b2e2ff5dfaa39391b53b04bae46c034ff0d93c2'
    spec=importlib.util.spec_from_file_location('frozen_caspar_layout',DONOR)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

@pytest.mark.parametrize('n',[0,1,5,257])
@pytest.mark.parametrize('d',[1,2,3,4,5,6,7,11,16])
def test_reference_exact_donor_roundtrip_and_bits(donor,n,d):
    words=np.arange(n*d,dtype=np.uint32).reshape(n,d)*np.uint32(65537)
    if words.size: words.flat[:min(words.size,5)]=np.array([0x80000000,0x7fc12345,0x7f800001,0xff800000,0],np.uint32)[:min(words.size,5)]
    a=words.view(np.float32); expected=donor.pack(a); actual=reference.pack(a)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(reference.unpack(actual,n,d).view(np.uint32),words)
    np.testing.assert_array_equal(donor.unpack(actual,n,d).view(np.uint32),words)

@pytest.mark.parametrize('view',['slice','reverse','broadcast'])
def test_donor_cpu_views_and_custom_permutation(donor,view):
    base=np.arange(60,dtype=np.float32).reshape(5,12)
    a={'slice':base[:,::2],'reverse':base[::-1,::-2],'broadcast':np.broadcast_to(base[:1,::2],(5,6))}[view]
    layout=((3,1,4),(5,2,0))
    x=reference.pack(a,layout); np.testing.assert_array_equal(x.view(np.uint32),donor.pack(a,layout).view(np.uint32))
    np.testing.assert_array_equal(reference.unpack(x,5,6,layout).view(np.uint32),a.view(np.uint32))


def model_shader(plan,source):
    """Independent host model of scalar lane owners; guards remain poisoned."""
    p=plan['params']; desc=plan['descriptor_u32']; total=p['outputCount']; out=np.full(total+7,0xdeadbeef,np.uint32)
    for i in range(total):
        value=0
        if plan['mode']=='pack':
            lo=0; hi=p['chunkCount']; iterations=0
            while lo<hi:
                mid=lo+(hi-lo)//2; iterations+=1
                if desc[mid*7]<=i: lo=mid+1
                else: hi=mid
            assert iterations<=13 and lo>0
            off,width,live,*mapping=desc[(lo-1)*7:lo*7]
            lane=(i-off)%width; row=(i-off)//width
            if lane<live: value=source[p['sourceBase']+row*p['rowStride']+mapping[lane]*p['componentStride']]
        else:
            row,component=divmod(i,p['components'])
            start=p['chunkCount']*7+component*3
            off,width,lane=desc[start:start+3]
            value=source[p['sourceBase']+off+row*width+lane]
        out[i]=value
    return out

@pytest.mark.parametrize('n',[0,1,5,257])
@pytest.mark.parametrize('d,layout',[(3,None),(7,None),(5,((3,1,4),(2,0)))])
def test_descriptor_word_models_match_donor_and_preserve_guards(donor,n,d,layout):
    count=n*(d*2+1); source=np.arange(max(count+3,4),dtype=np.uint32)*np.uint32(61)
    if n: source[1]=0x7f800001
    data=np.ndarray((n,d),dtype=np.float32,buffer=source,offset=4,strides=((d*2+1)*4,8))
    packed=donor.pack(data,layout)
    p=plan_layout('pack',n,d,layout,source_base_bytes=4,source_storage_bytes=source.nbytes,
        row_stride_bytes=(d*2+1)*4,component_stride_bytes=8,**LIMITS)
    actual=model_shader(p,source)
    np.testing.assert_array_equal(actual[:len(packed)],packed.view(np.uint32))
    np.testing.assert_array_equal(actual[len(packed):],np.full(7,0xdeadbeef,np.uint32))
    inp=np.concatenate((np.array([77],np.uint32),packed.view(np.uint32)))
    u=plan_layout('unpack',n,d,layout,source_base_bytes=4,source_storage_bytes=inp.nbytes,**LIMITS)
    restored=model_shader(u,inp)
    np.testing.assert_array_equal(restored[:n*d],data.view(np.uint32).reshape(-1))
    np.testing.assert_array_equal(restored[n*d:],np.full(7,0xdeadbeef,np.uint32))


def test_plan_rejects_invalid_layout_ranges_and_strides():
    def pack(**kw): return plan_layout('pack',1,3,**(LIMITS|{'source_storage_bytes':12,'row_stride_bytes':12,'component_stride_bytes':4}|kw))
    for layout in [((0,1),(1,)),((0,1),),((0,1,2,3,4),)]:
        with pytest.raises(ValueError): pack(layout=layout)
    with pytest.raises(ValueError): pack(row_stride_bytes=-4)
    with pytest.raises(ValueError): pack(component_stride_bytes=2)
    with pytest.raises(ValueError): pack(source_storage_bytes=8)
    with pytest.raises(ValueError): plan_layout('pack',1,4097,source_storage_bytes=4,row_stride_bytes=0,component_stride_bytes=0,**LIMITS)
    with pytest.raises(ValueError): plan_layout('pack',0xffffffff,3,source_storage_bytes=4,row_stride_bytes=0,component_stride_bytes=0,**LIMITS)
    with pytest.raises(ValueError): plan_layout('unpack',1,3,source_storage_bytes=16,row_stride_bytes=4,**LIMITS)

@pytest.mark.parametrize('version',['2026.12','2026.19'])
@pytest.mark.parametrize('entry',['packAosoa','unpackAosoa'])
def test_compiler_reflection(version,entry,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_'); compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail('explicit compiler missing')
        pytest.skip('compiler unavailable')
    identity=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (identity.stdout+identity.stderr).strip()==version
    output=tmp_path/'a.spv'; refl=tmp_path/'r.json'
    run=subprocess.run([str(compiler),str(SHADER),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(output),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    data=json.loads(refl.read_text()); ep=data['entryPoints'][0]
    assert ep['threadGroupSize']==[256,1,1]
    assert {x['name'] for x in ep['bindings'] if x['binding'].get('used')==1}=={'params','sourceWords','layoutWords','outputWords'}
    fields=next(x for x in data['parameters'] if x['name']=='params')['type']['elementType']['fields']
    assert [f['name'] for f in fields]==['n','components','chunkCount','sourceBase','rowStride','componentStride','outputCount']
    assert [f['binding']['offset'] for f in fields]==list(range(0,28,4))
    assert all(f['type']['scalarType']=='uint32' for f in fields) and output.stat().st_size>0
    dis=os.environ.get('SINTERGPU_SPIRV_DIS')
    if dis:
        asm=subprocess.run([dis,str(output)],capture_output=True,text=True,check=True).stdout
        arrays=set(re.findall(r'(%\S+)\s*=\s*OpTypeRuntimeArray\s+%uint\b',asm))
        stride4=set(re.findall(r'OpDecorate\s+(%\S+)\s+ArrayStride\s+4\b',asm))
        structs={match.group(1):match.group(2) for match in re.finditer(r'(%\S+)\s*=\s*OpTypeStruct\s+(%\S+)\s*$',asm,re.MULTILINE)}
        pointers={match.group(1):match.group(2) for match in re.finditer(r'(%\S+)\s*=\s*OpTypePointer\s+(?:Uniform|StorageBuffer)\s+(%\S+)',asm)}
        variables={match.group(1):match.group(2) for match in re.finditer(r'(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+(?:Uniform|StorageBuffer)',asm)}
        names={match.group(2):match.group(1) for match in re.finditer(r'OpName\s+(%\S+)\s+"(sourceWords|layoutWords|outputWords)"',asm)}
        assert set(names)=={'sourceWords','layoutWords','outputWords'}
        for name in names:
            arr=structs[pointers[variables[names[name]]]]
            assert arr in arrays and arr in stride4, name+' must use uint32 ArrayStride 4'


def test_cpu_validation_matches_donor(donor):
    for implementation in (donor,reference):
        with pytest.raises(TypeError): implementation.pack(np.ones((2,3),np.float64))
        with pytest.raises(ValueError): implementation.pack(np.ones((3,),np.float32))
        with pytest.raises(ValueError): implementation.pack(np.ones((2,0),np.float32))
        with pytest.raises(ValueError): implementation.pack(np.ones((2,3),np.float32),((0,1),(1,)))
        with pytest.raises(ValueError): implementation.unpack(np.ones(9,np.float32),3,3)
        with pytest.raises(ValueError): implementation.unpack(np.ones(0,np.float32),-1,3)


def test_gpu_model_supports_zero_stride_broadcast():
    n,d=3,3; source=np.array([0x80000000,0x7f800001,0x7fc12345],np.uint32)
    p=plan_layout('pack',n,d,source_storage_bytes=source.nbytes,row_stride_bytes=0,component_stride_bytes=4,**LIMITS)
    out=model_shader(p,source)
    expected=np.tile(np.r_[source,np.uint32(0)],n)
    np.testing.assert_array_equal(out[:len(expected)],expected)
    np.testing.assert_array_equal(out[len(expected):],np.full(7,0xdeadbeef,np.uint32))

@pytest.mark.parametrize('mode',['pack','unpack'])
def test_empty_input_is_explicitly_no_dispatch(mode):
    from sintergpu.adapters.slangpy import SlangPyAdapter
    opts={'row_stride_bytes':12,'component_stride_bytes':4} if mode=='pack' else {}
    plan=plan_layout(mode,0,3,source_storage_bytes=0,**(LIMITS|opts))
    assert plan['thread_count']==(0,1,1) and plan['dispatch_required'] is False
    class Device:
        def load_module(self,*args): pytest.fail('empty should not compile')
    SlangPyAdapter(Device(),allow_experimental=True).record('unused',thread_count=plan['thread_count'],bindings={},command_encoder=object())


def test_maximum_4096_chunks_binary_search_and_direct_lookup(donor):
    n,d=1,4096; layout=tuple((i,) for i in reversed(range(d)))
    values=(np.arange(d,dtype=np.uint32)*np.uint32(0x9e3779b9)).view(np.float32).reshape(1,d)
    packed=donor.pack(values,layout)
    p=plan_layout('pack',n,d,layout,source_storage_bytes=values.nbytes,row_stride_bytes=values.nbytes,component_stride_bytes=4,**LIMITS)
    assert p['params']['chunkCount']==4096
    assert len(p['descriptor_u32'])==7*4096+3*4096
    out=model_shader(p,values.view(np.uint32).reshape(-1))
    np.testing.assert_array_equal(out[:d],packed.view(np.uint32))
    u=plan_layout('unpack',n,d,layout,source_storage_bytes=packed.nbytes,**LIMITS)
    unpacked=model_shader(u,packed.view(np.uint32))
    np.testing.assert_array_equal(unpacked[:d],values.view(np.uint32).reshape(-1))
    np.testing.assert_array_equal(unpacked[d:],np.full(7,0xdeadbeef,np.uint32))
