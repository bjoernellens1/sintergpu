# SPDX-License-Identifier: Apache-2.0
"""CPU-only VKL compact backward oracle; no GPU implementation.

Frozen donor: bjoernellens1/vkl 3384023a311651ea4260cf12ddab1567972f89f3,
source src/vkl/kernels/reference/compact.py, SHA-256
f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff.
"""
import operator
import numpy as np


def compact_value_backward(shape, mask, grad_output):
    """Return logical input gradients; selection is stable and capacity-bounded."""
    shape=tuple(operator.index(x) for x in shape)
    if len(shape)<2 or any(x<0 for x in shape): raise ValueError('invalid shape')
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=shape[:2]: raise ValueError('invalid mask')
    if not isinstance(grad_output,np.ndarray) or grad_output.ndim!=len(shape) or grad_output.shape[0]!=shape[0] or grad_output.shape[2:]!=shape[2:]: raise ValueError('invalid output gradient')
    if grad_output.dtype.kind not in 'fc': raise ValueError('gradient must be floating or complex')
    result=np.zeros(shape,dtype=grad_output.dtype)
    capacity=grad_output.shape[1]
    for batch in range(shape[0]):
        retained=0
        for item in range(shape[1]):
            if mask[batch,item] and retained<capacity:
                result[batch,item]=grad_output[batch,retained]
                retained+=1
    return result


def accumulate_view_gradient(logical, *, base_size, strides, offset=0):
    """Scatter logical gradients into an element-addressed base, summing aliases.

    Strides and offset are in elements. The returned gradient has logical dtype;
    this models a view of one storage, not arbitrary Torch view operations.
    Aliases accumulate in NumPy index order using the gradient dtype. Floating
    point results may differ from another implementation with a different
    reduction order; compare with an appropriate tolerance, not bitwise.
    """
    if not isinstance(logical,np.ndarray) or logical.dtype.kind not in 'fc': raise ValueError('invalid logical gradient')
    if logical.dtype.kind=='c' and not np.isfinite(logical).all(): raise ValueError('nonfinite complex accumulation is unsupported')
    base_size=operator.index(base_size); offset=operator.index(offset)
    strides=tuple(operator.index(x) for x in strides)
    if base_size<0 or len(strides)!=logical.ndim: raise ValueError('invalid storage')
    result=np.zeros(base_size,dtype=logical.dtype)
    for index in np.ndindex(logical.shape):
        address=offset+sum(i*s for i,s in zip(index,strides))
        if address<0 or address>=base_size: raise ValueError('view outside storage')
        result[address]+=logical[index]
    return result
