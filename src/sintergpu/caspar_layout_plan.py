# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded scalar32 Caspar AoS/AoSoA layout conversion plan.

Sources must be little-endian float32 words. Pack supports nonnegative, 4-byte
aligned input strides including zero; unpack takes contiguous packed input.
Negative or byte-unaligned strides, more than 4096 components and gradients are unsupported. Caller owns
buffers, descriptor upload, synchronization and dispatch; writable output must
not overlap any input/descriptor. No allocation, submission or GPU qualification.
"""
MAX=0xfffffffc
MAX_COMPONENTS=4096
SENTINEL=0xffffffff


def _uint(name,value,limit=MAX):
    if type(value) is not int or not 0<=value<=limit: raise ValueError(name+' out of range')
    return value


def plan_layout(mode,n,components,layout=None,*,source_base_bytes=0,source_storage_bytes,
                row_stride_bytes=None,component_stride_bytes=None,max_buffer_bytes,max_workgroups_x,
                max_threads_per_group,max_group_size_x):
    if mode not in ('pack','unpack'): raise ValueError('mode')
    n=_uint('n',n); d=_uint('components',components)
    if not d or d>MAX_COMPONENTS: raise ValueError('components must be in 1..4096')
    limit=_uint('max_buffer_bytes',max_buffer_bytes); groups=_uint('max_workgroups_x',max_workgroups_x)
    if not limit or not groups or _uint('max_threads_per_group',max_threads_per_group)<256 or _uint('max_group_size_x',max_group_size_x)<256:
        raise ValueError('device limits')
    if layout is None or layout==() or layout==[]: chunks=tuple(tuple(range(i,min(i+4,d))) for i in range(0,d,4))
    else:
        if not isinstance(layout,(tuple,list)): raise ValueError('layout')
        chunks=[]
        for c in layout:
            if not isinstance(c,(tuple,list)) or not 1<=len(c)<=4: raise ValueError('invalid chunk')
            chunks.append(tuple(_uint('component index',v) for v in c))
        chunks=tuple(chunks)
    if sorted(x for c in chunks for x in c)!=list(range(d)): raise ValueError('layout must be a permutation')
    rows=_uint('logical output size',n*d); stored=_uint('packed size',n*sum(4 if len(c)==3 else len(c) for c in chunks))
    base=_uint('source base',source_base_bytes); storage=_uint('source storage',source_storage_bytes)
    if base%4 or storage%4 or storage>limit: raise ValueError('source alignment/range')
    if mode=='pack':
        row=_uint('row stride',row_stride_bytes); col=_uint('component stride',component_stride_bytes)
        if row%4 or col%4: raise ValueError('source strides must be aligned')
        end=base+(n-1)*row+(d-1)*col+4 if n else base
        output=stored
    else:
        if row_stride_bytes is not None or component_stride_bytes is not None: raise ValueError('unpack source must be contiguous')
        row=col=0; end=base+stored*4; output=rows
    if end>MAX or end>storage: raise ValueError('source out of bounds')
    descriptor=[]; component_lookup=[None]*d; offset=0
    for chunk in chunks:
        width=4 if len(chunk)==3 else len(chunk)
        descriptor.extend([offset,width,len(chunk),*chunk,*([SENTINEL]*(4-len(chunk)))])
        for lane,component in enumerate(chunk): component_lookup[component]=(offset,width,lane)
        offset+=n*width
    for entry in component_lookup: descriptor.extend(entry)
    sizes={'sourceWords':max(4,storage),'layoutWords':max(4,len(descriptor)*4),'outputWords':max(4,output*4)}
    if any(v>MAX or v>limit for v in sizes.values()): raise ValueError('buffer range')
    groups_needed=(output+255)//256
    if groups_needed>groups or groups_needed*256>0xffffffff: raise ValueError('dispatch range')
    params={'n':n,'components':d,'chunkCount':len(chunks),'sourceBase':base//4,
            'rowStride':row//4,'componentStride':col//4,'outputCount':output}
    return {'status':'experimental_unqualified','mode':mode,'source':'src/sintergpu/kernels/experimental/caspar_layout.slang',
            'entry':'packAosoa' if mode=='pack' else 'unpackAosoa','params':params,'descriptor_u32':descriptor,
            'dispatch_required':bool(output),'descriptor_stride_bytes':4,'descriptor_layout':'seven words per chunk followed by three words per component',
            'buffer_bytes':sizes,'thread_count':(groups_needed*256,1,1),
            'layout':chunks,'ownership':'caller supplies verified disjoint resources, descriptor contents and explicit synchronization',
            'limitations':'float32 little-endian word bits; nonnegative aligned strides only; no gradient or GPU qualification'}
