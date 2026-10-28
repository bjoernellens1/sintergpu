# SPDX-License-Identifier: Apache-2.0
"""Experimental caller-owned logical gradient graph; no GPU adapter or autograd.

Forward slotSource must remain alive, immutable and validated: each non-sentinel
index is in-range and unique in its batch. Upstream is tightly packed by slot;
output is dense logical row-major. Both are opaque little-endian bytes. Inputs,
outputs and scratch must be caller-owned with no writable overlap. Explicit
barriers are required between stages. View/base gradient accumulation belongs
to the caller's autograd framework. No hidden submission or synchronization.
"""
from math import prod

MAX=0xfffffffc
SENTINEL=0xffffffff


def plan_backward(shape,capacity,item_bytes,*,grad_base,grad_storage_bytes,max_buffer_bytes,max_workgroups_x,max_group_size_x,max_threads_per_group):
    def uint(name,x,limit=MAX):
        if type(x) is not int or not 0<=x<=limit: raise ValueError(name+' out of range')
        return x
    if not isinstance(shape,(tuple,list)) or len(shape)<2: raise ValueError('invalid shape')
    dims=tuple(uint('dimension',x) for x in shape)
    cap=uint('capacity',capacity); width=uint('item bytes',item_bytes)
    if not width: raise ValueError('item bytes must be positive')
    limit=uint('buffer limit',max_buffer_bytes)
    groups=uint('workgroups',max_workgroups_x)
    if not limit or not groups or uint('group size',max_group_size_x)<256 or uint('threads',max_threads_per_group)<256: raise ValueError('device limits')
    base=uint('grad base',grad_base); storage=uint('gradient storage',grad_storage_bytes)
    if storage%4 or storage>limit: raise ValueError('gradient storage padding/range')
    batches,items,*tail=dims
    logical_items=uint('logical items',batches*items); slots=uint('slots',batches*cap)
    payload=uint('bytes per item',prod(tail)*width)
    logical=uint('logical bytes',logical_items*payload)
    upstream=uint('upstream bytes',slots*payload if logical_items else 0)
    if base+upstream>MAX or ((base+upstream+3)//4)*4>storage: raise ValueError('upstream range')
    sizes={'slotSource':max(4,slots*4),'gradWords':max(4,storage),'inverse':max(4,logical_items*4),'outputWords':max(4,((logical+3)//4)*4)}
    if any(v>limit or v>MAX for v in sizes.values()): raise ValueError('buffer range')
    entries=[]
    for entry,count in [('backwardInit',logical_items),('backwardInverse',slots if logical_items else 0),('backwardGather',(logical+3)//4)]:
        if not count: continue
        if (count+255)//256>groups or ((count+255)//256)*256>0xffffffff: raise ValueError('dispatch range')
        entries.append({'entry':entry,'logical_count':count,'thread_count':((count+255)//256*256,1,1),'barrier_before':True})
    return {'params':{'batches':batches,'items':items,'capacity':cap,'bytesPerItem':payload,'logicalBytes':logical,'gradBase':base},'buffers':sizes,'stages':entries,'sentinel':SENTINEL}
