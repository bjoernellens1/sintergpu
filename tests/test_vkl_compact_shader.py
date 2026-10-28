# SPDX-License-Identifier: Apache-2.0
"""Compile/reflection and independent byte-model tests; not GPU parity."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from sintergpu.compact_plan import plan_compact, value_address
from sintergpu.references.compact_numeric import select_compact

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'benchmarks/shaders/vkl_compact_candidate.slang'
ENTRIES={
 'compactInit':{'params','slotToSource','indicesOut','validOut','actionsOut','countsOut'},
 'compactFlags':{'params','maskWords','flags','segmentIds'},
 'compactScatterMetadata':{'params','maskWords','actionWords','prefix','slotToSource','indicesOut','validOut','actionsOut'},
 'compactFinalizeCounts':{'params','prefix','countsOut'},
 'compactGatherWords':{'params','valueWords','shapeStrides','slotToSource','valuesOut'},
}


def test_candidate_source_is_additive_and_word_owned():
    assert SHADER.is_file(), 'experimental compact shader missing'
    source=SHADER.read_text()
    for entry in ENTRIES: assert 'void '+entry+'(' in source
    assert 'Interlocked' not in source and 'groupshared' not in source
    assert 'valuesOut[wordIndex]=' in source


@pytest.mark.parametrize('version',['2026.12','2026.19'])
@pytest.mark.parametrize('entry',list(ENTRIES))
def test_compilation_reflection_and_workgroups(version,entry,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_')
    compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail(f'{key} missing')
        pytest.skip('compiler unavailable')
    info=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (info.stdout+info.stderr).strip()==version
    refl=tmp_path/'r.json'; spv=tmp_path/'out.spv'
    p=subprocess.run([str(compiler),str(SHADER),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    data=json.loads(refl.read_text())
    params=next(x for x in data['parameters'] if x['name']=='params')
    fields=params['type']['elementType']['fields']
    names=['batches','items','capacity','rank','itemBytes','payloadItems','logicalValueBytes','actionBytes','valueBase','maskBase','maskBatchStride','maskItemStride','actionBase','actionBatchStride','actionItemStride']
    assert [field['name'] for field in fields]==names
    assert [field['type']['scalarType'] for field in fields]==['uint32']*len(names)
    assert [field['binding']['offset'] for field in fields]==list(range(0,4*len(names),4))
    ep=data['entryPoints'][0]
    assert ep['threadGroupSize']==[256,1,1]
    assert {b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}==ENTRIES[entry]
    dis=os.environ.get('SINTERGPU_SPIRV_DIS')
    if dis:
        asm=subprocess.run([dis,str(spv)],capture_output=True,text=True,check=True).stdout
        assert 'OpAtomic' not in asm and 'OpControlBarrier' not in asm


@pytest.mark.parametrize('width',[1,2,3,4,8,16])
@pytest.mark.parametrize('broadcast',[False,True])
@pytest.mark.parametrize('action_width',[0,4,8])
def test_byte_model_matches_independent_compaction(width,broadcast,action_width):
    """Exercise the specified per-word gather, not compiled shader execution."""
    rng=np.random.default_rng(310+width)
    source=bytearray(rng.integers(0,256,160,dtype=np.uint8).tobytes())
    mask_bytes=bytearray([0,1,0,1,0,0,1,1,0,0,0,0])
    actions=bytearray(rng.integers(0,256,64,dtype=np.uint8).tobytes())
    shape=(2,3,2); strides=(0 if broadcast else 64,11,width)
    action_strides=(24,5) if action_width else None
    plan=plan_compact(shape,capacity=4,item_bytes=width,value_strides=strides,value_base_byte=1,value_storage_bytes=len(source),
        mask_strides=(5,1),mask_base_byte=1,mask_storage_bytes=len(mask_bytes),
        action_item_bytes=action_width,action_strides=action_strides,action_base_byte=1 if action_width else 0,
        action_storage_bytes=len(actions) if action_width else 0,max_workgroups_x=65535,max_buffer_bytes=4096,max_threads_per_group=256,max_group_size_x=256)
    values=np.ndarray(shape,dtype=np.dtype('V'+str(width)),buffer=source,offset=1,strides=strides)
    mask=np.ndarray((2,3),dtype=np.bool_,buffer=mask_bytes,offset=1,strides=(5,1))
    codes=None if not action_width else np.ndarray((2,3),dtype='<i'+str(action_width),buffer=actions,offset=1,strides=action_strides)
    reference=select_compact(values,mask,action_codes=codes,capacity=4)
    slots=[None]*8; counts=[]; out_indices=np.zeros((2,4),np.int64); out_valid=np.zeros((2,4),bool)
    code_bits=np.zeros((2,4),dtype=np.int64 if action_width==8 else np.int32)
    for b in range(2):
        rank=0
        for n in range(3):
            if mask[b,n]:
                rank+=1
                if rank<=4:
                    slot=b*4+rank-1; slots[slot]=n
                    out_indices[b,rank-1]=n; out_valid[b,rank-1]=True
                    if codes is not None: code_bits[b,rank-1]=codes[b,n]
        counts.append(min(rank,4))
    result=bytearray()
    logical=plan['outputs']['values']['logical_bytes']; per_slot=2*width
    for word in range((logical+3)//4):
        packed=0
        for lane in range(4):
            destination=word*4+lane
            if destination>=logical: continue
            slot,inside=divmod(destination,per_slot)
            if slots[slot] is None: continue
            batch=slot//4; item,byte=divmod(inside,width)
            packed|=source[value_address(plan,batch,slots[slot],item)+byte]<<(8*lane)
        result.extend(packed.to_bytes(4,'little'))
    assert bytes(result[:logical])==reference['values'].tobytes()
    assert not any(result[logical:])
    np.testing.assert_array_equal(out_indices,reference['indices'])
    np.testing.assert_array_equal(out_valid,reference['valid'])
    np.testing.assert_array_equal(code_bits,reference['action_codes'])
    np.testing.assert_array_equal(np.array(counts,np.int64),reference['count'])
