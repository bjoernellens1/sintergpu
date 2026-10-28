# SPDX-License-Identifier: Apache-2.0
"""Design-only VKL compact graph and scalar32 byte-packing ABI.

There is no implementation of the compact-prefixed passes yet; no recording,
allocation, dispatch, synchronization or GPU qualification is supplied here.
Source semantics: bjoernellens1/vkl revision 3384023a311651ea4260cf12ddab1567972f89f3,
reference compact.py SHA-256 f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff;
Apache-2.0 owner authorization, see provenance/vkl.json. The proposed segmented
scan uses the existing separately noticed MIT-licensed primitives/scan_uint.slang.
"""
from math import prod
from .primitives import scan_layout

UINT_MAX=0xffffffff
MAX_BUFFER=0xfffffffc


def _uint(name,value,maximum=UINT_MAX):
    if type(value) is not int or not 0<=value<=maximum: raise ValueError(name+' must be a bounded nonnegative integer')
    return value


def _buffer(name,bytes_,limit):
    _uint(name,bytes_,min(MAX_BUFFER,limit))
    if bytes_%4: raise ValueError(name+' must include padding to uint32 words')
    return bytes_


def _span(name,shape,strides,base,item_bytes,storage,limit):
    if not isinstance(strides,(tuple,list)) or len(strides)!=len(shape): raise ValueError(name+' stride rank mismatch')
    strides=tuple(_uint(name+' stride',v) for v in strides)
    base=_uint(name+' base',base)
    _buffer(name+' storage',storage,limit)
    if all(shape):
        end=base+sum((n-1)*s for n,s in zip(shape,strides))+item_bytes
        if end>UINT_MAX or ((end+3)//4)*4>storage: raise ValueError(name+' address exceeds storage or uint32 range')
    elif base>storage: raise ValueError(name+' base exceeds storage')
    return {'shape':tuple(shape),'strides_bytes':strides,'base_byte':base,'item_bytes':item_bytes,'storage_bytes':storage}


def plan_compact(shape,*,capacity,item_bytes,value_strides,value_storage_bytes,mask_strides,mask_storage_bytes,
                 max_workgroups_x,max_buffer_bytes,max_threads_per_group,max_group_size_x,value_base_byte=0,mask_base_byte=0,
                 action_item_bytes=0,action_strides=None,action_storage_bytes=0,action_base_byte=0):
    """Specify fixed-capacity output and ordered passes; no shaders are selected.

    Inputs are nonnegative-stride, byte-addressed little-endian buffers, padded
    to complete uint32 words; negative strides, object/reference values, and
    non-byte-addressable layouts are unsupported. Mask bytes must be 0 or 1.
    Logical values are opaque fixed-size item bytes; trailing dimensions are
    flattened in C order, while input addresses use all supplied strides.
    Indices/counts and 64-bit actions are low/high uint32 words; valid is uint32.
    Each gather invocation owns exactly one complete destination uint32 word.
    """
    if not isinstance(shape,(tuple,list)) or len(shape)<2: raise ValueError('shape must contain batch and items')
    shape=tuple(_uint('dimension',n) for n in shape)
    capacity=_uint('capacity',capacity); item_bytes=_uint('item_bytes',item_bytes)
    if item_bytes==0: raise ValueError('item bytes must be positive')
    groups=_uint('max_workgroups_x',max_workgroups_x); limit=_uint('max_buffer_bytes',max_buffer_bytes,MAX_BUFFER)
    if not groups or not limit: raise ValueError('positive device limits required')
    if _uint('max_threads_per_group',max_threads_per_group)<256 or _uint('max_group_size_x',max_group_size_x)<256:
        raise ValueError('device must support 256-thread workgroups')
    if len(shape)*8>limit: raise ValueError('shape/stride descriptor exceeds device limit')
    b,n,*tail=shape
    candidates=_uint('batch items',b*n); slots=_uint('batch capacity',b*capacity)
    payload_items=_uint('trailing elements',prod(tail))
    logical_bytes=_uint('output value bytes',slots*payload_items*item_bytes,MAX_BUFFER)
    rounded_bytes=((logical_bytes+3)//4)*4
    if rounded_bytes>min(MAX_BUFFER,limit): raise ValueError('value output buffer exceeds device limit')
    for size in (candidates*4,slots*4,slots*8,b*8):
        if size>min(MAX_BUFFER,limit): raise ValueError('metadata buffer exceeds uint32 or device limit')
    values=_span('values',shape,value_strides,value_base_byte,item_bytes,value_storage_bytes,limit)
    mask=_span('mask',(b,n),mask_strides,mask_base_byte,1,mask_storage_bytes,limit)
    if action_item_bytes not in (0,4,8) or type(action_item_bytes) is not int: raise ValueError('actions must be absent, int32 or int64')
    if action_item_bytes:
        actions=_span('actions',(b,n),action_strides,action_base_byte,action_item_bytes,action_storage_bytes,limit)
    else:
        if action_strides is not None or action_storage_bytes!=0 or action_base_byte!=0: raise ValueError('absent actions cannot have a layout')
        actions=None
    scan_sizes,scan_passes=scan_layout(candidates)
    for size in scan_sizes.values():
        if size*4>min(MAX_BUFFER,limit): raise ValueError('scan scratch exceeds device limit')
    stages=[]
    def add(entry,count,*,bindings=None,implemented=False,params=None,source=None,specialization=None):
        if not count: return
        if (count+255)//256>groups: raise ValueError('dispatch exceeds device workgroup limit')
        stages.append({'entry':entry,'logical_count':count,'thread_count':(((count+255)//256)*256,1,1),'barrier_before':True,
                       'bindings':bindings or {},'implemented':implemented,
                       'params':params,'source':source,'specialization':specialization})
    if slots or b: add('compactInit',max(slots,b))
    if candidates and capacity:
        add('compactFlags',candidates)
        for p in scan_passes:
            add(p.entry,p.count,bindings=dict(p.bindings),implemented=True,
                params={'count':p.count,'product':0},
                source='src/sintergpu/kernels/primitives/scan_uint.slang',
                specialization='uint32_segmented_inclusive_sum')
        add('compactScatterMetadata',candidates)
        add('compactFinalizeCounts',b)
    if logical_bytes: add('compactGatherWords',(logical_bytes+3)//4)
    # Every reflected buffer has valid scalar storage even when the shader's
    # logical branch performs no access. Writable dummies are separate resources.
    resources={}
    def resource(name,logical_bytes,access):
        allocated=max(4,logical_bytes)
        if allocated>limit: raise ValueError('reflected resource exceeds device limit')
        resources[name]={'logical_bytes':logical_bytes,'allocated_bytes':allocated,'access':access}
    resource('value_words',value_storage_bytes,'read')
    resource('mask_words',mask_storage_bytes,'read')
    resource('action_words',action_storage_bytes,'read')
    resource('shape_strides',len(shape)*8,'read')
    resource('slot_to_source',slots*4,'write')
    resource('indices_out',slots*8,'write')
    resource('valid_out',slots*4,'write')
    resource('actions_out',slots*(8 if action_item_bytes==8 else 4),'write')
    resource('counts_out',b*8,'write')
    resource('values_out',rounded_bytes,'write')
    for name,size in scan_sizes.items(): resource(name,size*4,'scratch')
    descriptor=[item for pair in zip(shape,values['strides_bytes']) for item in pair]
    params={'batches':b,'items':n,'capacity':capacity,'rank':len(shape),
            'itemBytes':item_bytes,'payloadItems':payload_items,'logicalValueBytes':logical_bytes,
            'actionBytes':action_item_bytes,'valueBase':value_base_byte,'maskBase':mask_base_byte,
            'maskBatchStride':mask['strides_bytes'][0],'maskItemStride':mask['strides_bytes'][1],
            'actionBase':action_base_byte,'actionBatchStride':actions['strides_bytes'][0] if actions else 0,
            'actionItemStride':actions['strides_bytes'][1] if actions else 0}
    compact_bindings={
      'compactInit':{'slotToSource':'slot_to_source','indicesOut':'indices_out','validOut':'valid_out','actionsOut':'actions_out','countsOut':'counts_out'},
      'compactFlags':{'maskWords':'mask_words','flags':'values','segmentIds':'segment_ids'},
      'compactScatterMetadata':{'maskWords':'mask_words','actionWords':'action_words','prefix':'output','slotToSource':'slot_to_source','indicesOut':'indices_out','validOut':'valid_out','actionsOut':'actions_out'},
      'compactFinalizeCounts':{'prefix':'output','countsOut':'counts_out'},
      'compactGatherWords':{'valueWords':'value_words','shapeStrides':'shape_strides','slotToSource':'slot_to_source','valuesOut':'values_out'},
    }
    for stage in stages:
        if stage['entry'] in compact_bindings:
            stage['bindings']=compact_bindings[stage['entry']]
            stage['params']=dict(params)
            stage['source']='benchmarks/shaders/vkl_compact_candidate.slang'
    return {'status':'design_only_no_gpu_implementation','inputs':{'values':values,'mask':mask,'actions':actions},
            'outputs':{'values':{'shape':(b,capacity,*tail),'item_bytes':item_bytes,'logical_bytes':logical_bytes,'buffer_bytes':rounded_bytes,'encoding':'packed_bytes_little_endian_words'},
                       'indices':{'shape':(b,capacity),'elements':slots*2,'encoding':'little_endian_i64_from_u32_words'},
                       'valid':{'shape':(b,capacity),'elements':slots,'encoding':'uint32_logical'},
                       'actions':{'shape':(b,capacity),'elements':slots*(2 if action_item_bytes==8 else 1),'encoding':'little_endian_i64_from_u32_words' if action_item_bytes==8 else 'int32_bits'},
                       'counts':{'shape':(b,),'elements':b*2,'encoding':'little_endian_i64_from_u32_words'}},
            'scratch':{'slot_to_source_u32_elements':slots,'scan_u32_elements':scan_sizes},'resources':resources,'shape_strides_u32':descriptor,'stages':stages,'capacity':capacity,
            'sentinel':UINT_MAX,'autograd':'backward_required_before_differentiable_use',
            'ownership':'all buffers and command submission belong to caller; no hidden readback',
            'limitations':'No shader/adapter implementation; positive and zero strides only; packed outputs use low-address byte in low uint32 bits; caller supplies actual device limits.'}


def value_address(plan,batch,index,trailing_flat):
    """Calculate a logical source item byte address for contract verification."""
    desc=plan['inputs']['values']; shape=desc['shape']; strides=desc['strides_bytes']; tail=shape[2:]
    if type(batch) is not int or type(index) is not int or type(trailing_flat) is not int or not 0<=batch<shape[0] or not 0<=index<shape[1] or not 0<=trailing_flat<prod(tail):
        raise ValueError('logical index out of range')
    coords=[]; remaining=trailing_flat
    for dim in reversed(tail): coords.append(remaining%dim); remaining//=dim
    return desc['base_byte']+batch*strides[0]+index*strides[1]+sum(c*s for c,s in zip(reversed(coords),strides[2:]))
