# SPDX-License-Identifier: Apache-2.0
"""Independent CPU stage model and case generation for compact GPU tests."""
import numpy as np
from sintergpu.compact_plan import plan_compact, value_address
from sintergpu.references.compact_numeric import select_compact

GUARD=8
POISON=np.uint32(0xa5b6c7d8)


def padded(n): return max(4,((n+3)//4)*4)


def make_case(n,capacity,width,action,*,max_workgroups_x=65535,max_buffer_bytes=32<<20,max_threads_per_group=256,max_group_size_x=256):
    b=2; shape=(b,n,2); itemstride=width*2; batchstride=n*itemstride
    value_size=padded(1+b*batchstride); mask_size=padded(1+b*n); action_size=padded(1+b*n*action) if action else 0
    v=np.arange(value_size,dtype=np.uint8)*np.uint8(17); mask=np.zeros(mask_size,np.uint8)
    for batch in range(b):
        for i in range(n): mask[1+batch*n+i]=int((i+batch)%3!=0)
    codes=(np.arange(action_size,dtype=np.uint8)*np.uint8(13)) if action else np.zeros(0,np.uint8)
    plan=plan_compact(shape,capacity=capacity,item_bytes=width,value_strides=(batchstride,itemstride,width),value_base_byte=1,value_storage_bytes=value_size,
        mask_strides=(n,1),mask_base_byte=1,mask_storage_bytes=mask_size,
        action_item_bytes=action,action_strides=(n*action,action) if action else None,action_base_byte=1 if action else 0,action_storage_bytes=action_size,
        max_workgroups_x=max_workgroups_x,max_buffer_bytes=max_buffer_bytes,max_threads_per_group=max_threads_per_group,max_group_size_x=max_group_size_x)
    return {'plan':plan,'values_bytes':v,'mask_bytes':mask,'action_bytes':codes,'shape':shape,'width':width,'capacity':capacity,'action':action}


def initial_state(case):
    p=case['plan']; state={name:np.full(desc['allocated_bytes']//4+GUARD,POISON,np.uint32) for name,desc in p['resources'].items()}
    for name,key in [('value_words','values_bytes'),('mask_words','mask_bytes'),('action_words','action_bytes')]:
        raw=case[key]
        if raw.size: state[name][:raw.size//4]=raw.view('<u4')
        else: state[name][0]=0
    state['shape_strides'][:len(p['shape_strides_u32'])]=p['shape_strides_u32']
    return state


def byte(words,offset): return (int(words[offset//4])>>(8*(offset%4)))&255


def model_stage(case,state,stage):
    p=case['plan']; param=next((s['params'] for s in p['stages'] if s['entry'].startswith('compact')),None)
    b,n,cap=case['shape'][0],case['shape'][1],case['capacity']; slots=b*cap; entry=stage['entry']
    if entry=='compactInit':
        state['slot_to_source'][:slots]=0xffffffff; state['indices_out'][:2*slots]=0; state['valid_out'][:slots]=0
        state['actions_out'][:slots*(2 if case['action']==8 else 1)]=0; state['counts_out'][:2*b]=0
    elif entry=='compactFlags':
        for i in range(b*n):
            batch,index=divmod(i,n); state['values'][i]=int(byte(state['mask_words'],1+batch*n+index)!=0); state['segment_ids'][i]=batch
    elif entry=='scanBlocks':
        q=stage['bindings']; count=stage['params']['count']; vals=state[q['values']]; ids=state[q['segmentIds']]; out=state[q['output']]
        for start in range(0,count,256):
            accum=0; previous=None; end=min(start+256,count)
            for i in range(start,end):
                if previous!=int(ids[i]): accum=0
                accum+=int(vals[i]); out[i]=accum; previous=int(ids[i])
            block=start//256; state[q['blockValues']][block]=accum; state[q['blockIds']][block]=previous
    elif entry=='scanCarry':
        q=stage['bindings']; count=stage['params']['count']
        for i in range(count):
            block=i//256
            if block and state[q['segmentIds']][i]==state[q['blockIds']][block-1]:
                state[q['output']][i]+=state[q['scannedBlocks']][block-1]
    elif entry=='compactScatterMetadata':
        for i in range(b*n):
            batch,index=divmod(i,n)
            if not byte(state['mask_words'],1+i): continue
            rank=int(state['output'][i])
            if not rank or rank>cap: continue
            slot=batch*cap+rank-1; state['slot_to_source'][slot]=index; state['indices_out'][2*slot]=index
            state['indices_out'][2*slot+1]=0; state['valid_out'][slot]=1
            width=case['action']
            if width:
                offset=1+i*width
                for w in range(width//4):
                    word=sum(byte(state['action_words'],offset+4*w+k)<<(8*k) for k in range(4))
                    state['actions_out'][slot*(width//4)+w]=word
    elif entry=='compactFinalizeCounts':
        for batch in range(b):
            state['counts_out'][2*batch]=min(int(state['output'][(batch+1)*n-1]),cap) if n else 0
            state['counts_out'][2*batch+1]=0
    elif entry=='compactGatherWords':
        logical=p['outputs']['values']['logical_bytes']; width=case['width']; per_slot=2*width
        for word in range((logical+3)//4):
            packed=0
            for lane in range(4):
                pos=word*4+lane
                if pos>=logical: continue
                slot,inside=divmod(pos,per_slot); source=int(state['slot_to_source'][slot])
                if source==0xffffffff: continue
                item,inbyte=divmod(inside,width)
                offset=value_address(p,slot//cap,source,item)+inbyte
                packed|=byte(state['value_words'],offset)<<(lane*8)
            state['values_out'][word]=packed
    else: raise ValueError('unexpected stage '+entry)


def final_reference(case):
    b,n,_=case['shape']; width=case['width']; v=case['values_bytes']; m=case['mask_bytes']; c=case['action_bytes']
    values=np.ndarray(case['shape'],dtype=np.dtype('V'+str(width)),buffer=v,offset=1,strides=(n*width*2,width*2,width))
    mask=np.ndarray((b,n),dtype=np.bool_,buffer=m,offset=1,strides=(n,1))
    codes=None if not case['action'] else np.ndarray((b,n),dtype='<i'+str(case['action']),buffer=c,offset=1,strides=(n*case['action'],case['action']))
    return select_compact(values,mask,action_codes=codes,capacity=case['capacity'])


def verify_final(case,state,reference):
    p=case['plan']; b,n,_=case['shape']; cap=case['capacity']; slots=b*cap
    actual=state['values_out'].view(np.uint8)[:p['outputs']['values']['logical_bytes']]
    np.testing.assert_array_equal(actual,np.frombuffer(reference['values'].tobytes(),np.uint8))
    np.testing.assert_array_equal(state['indices_out'][:slots*2].view('<i8').reshape(b,cap),reference['indices'])
    np.testing.assert_array_equal(state['valid_out'][:slots].reshape(b,cap),reference['valid'].astype(np.uint32))
    dtype='<i8' if case['action']==8 else '<i4'
    np.testing.assert_array_equal(state['actions_out'][:slots*(2 if case['action']==8 else 1)].view(dtype).reshape(b,cap),reference['action_codes'])
    np.testing.assert_array_equal(state['counts_out'][:2*b].view('<i8'),reference['count'])


def frozen_reference(case):
    """Execute the frozen Torch CPU donor for Torch-representable element widths."""
    import hashlib
    import importlib.util
    from pathlib import Path
    import torch
    width=case['width']
    if width not in (1,2,4,8):
        return None
    path=Path(__file__).resolve().parents[1]/'references/donors/vkl/src/vkl/kernels/reference/compact.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff':
        raise ValueError('frozen donor source mismatch')
    spec=importlib.util.spec_from_file_location('frozen_compact_gpu_reference',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    b,n,_=case['shape']; action=case['action']
    values=np.ndarray(case['shape'],dtype='<i'+str(width),buffer=case['values_bytes'],offset=1,strides=(n*width*2,width*2,width)).copy()
    mask=np.ndarray((b,n),dtype=np.bool_,buffer=case['mask_bytes'],offset=1,strides=(n,1)).copy()
    codes=None if not action else np.ndarray((b,n),dtype='<i'+str(action),buffer=case['action_bytes'],offset=1,strides=(n*action,action)).copy()
    result=module.select_compact(torch.from_numpy(values),torch.from_numpy(mask),action_codes=None if codes is None else torch.from_numpy(codes),capacity=case['capacity'])
    return {key:value.numpy() for key,value in result.items()}
