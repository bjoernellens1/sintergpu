# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reference for VKL stable fixed-capacity compaction.

Frozen donor: bjoernellens1/vkl revision
3384023a311651ea4260cf12ddab1567972f89f3,
``src/vkl/kernels/reference/compact.py`` SHA-256
f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff.
Original owner code authorized for Apache-2.0; see provenance/vkl.json.
NumPy CPU arrays only: no GPU dispatch, autograd, device validation, or timing.
"""
from __future__ import annotations
import operator
import numpy as np


def select_compact(values, mask, *, action_codes=None, capacity):
    """Stable per-batch selection with zero-filled, fixed-capacity outputs."""
    if (not isinstance(values,np.ndarray) or not isinstance(mask,np.ndarray) or
        values.ndim<2 or mask.shape!=values.shape[:2] or mask.dtype!=np.dtype(bool)):
        raise ValueError('values [B,N,...] and matching bool mask are required')
    if action_codes is not None and (not isinstance(action_codes,np.ndarray) or
        action_codes.shape!=mask.shape or action_codes.dtype not in (np.dtype(np.int32),np.dtype(np.int64))):
        raise ValueError('action_codes must match mask and use int32 or int64')
    if capacity<0: raise ValueError('capacity must be non-negative')
    capacity=operator.index(capacity)
    batch,count=mask.shape
    out_values=np.zeros((batch,capacity,*values.shape[2:]),dtype=values.dtype)
    out_indices=np.zeros((batch,capacity),dtype=np.int64)
    out_valid=np.zeros((batch,capacity),dtype=bool)
    out_codes=np.zeros((batch,capacity),dtype=np.int32 if action_codes is None else action_codes.dtype)
    out_count=np.zeros((batch,),dtype=np.int64)
    for b in range(batch):
        dst=0
        for index in range(count):
            if not mask[b,index] or dst>=capacity: continue
            out_values[b,dst]=values[b,index]
            out_indices[b,dst]=index
            out_valid[b,dst]=True
            if action_codes is not None: out_codes[b,dst]=action_codes[b,index]
            dst+=1
        out_count[b]=dst
    return {'values':out_values,'indices':out_indices,'action_codes':out_codes,'valid':out_valid,'count':out_count}
