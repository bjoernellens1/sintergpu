# SPDX-License-Identifier: Apache-2.0
"""Independent CPU Caspar float32 chunk-layout reference.

Frozen donor: bjoernellens1/caspar-slang revision
cab53390aa111cd5526cb30c9aef484e73521186,
src/caspar_slang/layout.py SHA-256 d32325aaefaa37df324645683b2e2ff5dfaa39391b53b04bae46c034ff0d93c2.
CPU only; no GPU execution or autograd support.
"""
import numpy as np


def _chunks(d,layout):
    if d<=0: raise ValueError('components must be positive')
    chunks=tuple(tuple(c) for c in layout) if layout else tuple(tuple(range(i,min(d,i+4))) for i in range(0,d,4))
    if any(not 1<=len(c)<=4 for c in chunks) or sorted(x for c in chunks for x in c)!=list(range(d)):
        raise ValueError('layout must contain every component once in chunks of 1 to 4')
    return chunks


def pack(array,layout=None):
    data=np.asarray(array)
    if data.dtype!=np.float32: raise TypeError('Caspar arrays must have dtype float32')
    if data.ndim!=2: raise ValueError('expected shape (N, D)')
    n,d=data.shape; chunks=_chunks(d,layout); src=data.view(np.uint32)
    total=n*sum(4 if len(chunk)==3 else len(chunk) for chunk in chunks)
    out=np.zeros(total,dtype=np.uint32); base=0
    for chunk in chunks:
        width=4 if len(chunk)==3 else len(chunk)
        for row in range(n):
            for lane,component in enumerate(chunk): out[base+row*width+lane]=src[row,component]
        base+=n*width
    return out.view(np.float32)


def unpack(array,n,components,layout=None):
    if n<0: raise ValueError('n must be nonnegative')
    data=np.asarray(array)
    if data.dtype!=np.float32: raise TypeError('Caspar arrays must have dtype float32')
    chunks=_chunks(components,layout); size=n*sum(4 if len(c)==3 else len(c) for c in chunks)
    if data.ndim!=1 or data.size!=size: raise ValueError('packed buffer size does not match dimensions')
    src=data.view(np.uint32); out=np.empty((n,components),np.uint32); base=0
    for chunk in chunks:
        width=4 if len(chunk)==3 else len(chunk)
        for row in range(n):
            for lane,component in enumerate(chunk): out[row,component]=src[base+row*width+lane]
        base+=n*width
    return out.view(np.float32)
