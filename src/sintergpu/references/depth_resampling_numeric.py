# SPDX-License-Identifier: Apache-2.0
"""Independent float32 NumPy oracles for VKL depth lift and propagation."""
import numpy as np


def depth_upsample(depth,valid,confidence,*,size):
    depth=np.asarray(depth,dtype=np.float32);confidence=np.asarray(confidence,dtype=np.float32)
    usable=np.asarray(valid,dtype=bool)&np.isfinite(depth)&(depth>0)
    h,w=depth.shape[-2:];oh,ow=size
    ys=np.minimum(np.floor(np.arange(oh,dtype=np.float32)*np.float32(h/oh)).astype(np.int64),h-1)
    xs=np.minimum(np.floor(np.arange(ow,dtype=np.float32)*np.float32(w/ow)).astype(np.int64),w-1)
    lift=lambda a:a[:,:,ys[:,None],xs[None,:]]
    out_valid=lift(usable)
    return dict(depth=lift(np.where(usable,depth,np.float32(0))),valid=out_valid,
                confidence=lift(np.where(usable,confidence,np.float32(0))),
                direct_valid=np.zeros(out_valid.shape,bool),propagated_valid=out_valid,
                provenance=out_valid.astype(np.uint8)*2)


def depth_densify(depth,valid,confidence,rgb,*,direct_valid=None,max_rgb_difference=.15,max_relative_spread=.1,min_support=2,steps=1):
    depth=np.asarray(depth,dtype=np.float32);confidence=np.asarray(confidence,dtype=np.float32);rgb=np.asarray(rgb,dtype=np.float32)
    usable=np.asarray(valid,dtype=bool)&np.isfinite(depth)&(depth>0)
    direct=usable.copy() if direct_valid is None else np.asarray(direct_valid,dtype=bool)&usable
    current_d=np.where(usable,depth,np.float32(0));current_c=np.where(usable,confidence,np.float32(0));current_v=usable.copy()
    batches,_,h,w=depth.shape
    for _ in range(steps):
        out_d=current_d.copy();out_c=current_c.copy();out_v=current_v.copy();support=np.zeros(depth.shape,np.int64)
        for b in range(batches):
            for y in range(h):
                for x in range(w):
                    values=[];weights=[]
                    for yy in range(max(0,y-1),min(h,y+2)):
                        for xx in range(max(0,x-1),min(w,x+2)):
                            with np.errstate(invalid='ignore'):
                                difference=np.max(np.abs(rgb[b,:,yy,xx]-rgb[b,:,y,x]))
                            if current_v[b,0,yy,xx] and difference<=max_rgb_difference:
                                values.append(current_d[b,0,yy,xx]);weights.append(current_c[b,0,yy,xx])
                    n=len(values);support[b,0,y,x]=n
                    if not current_v[b,0,y,x] and n>=min_support:
                        ordered=np.sort(np.array(values,np.float32));median=ordered[(n-1)//2]
                        spread=(ordered[-1]-ordered[0])/max(median,np.finfo(np.float32).eps)
                        if spread<=max_relative_spread:
                            out_v[b,0,y,x]=True;out_d[b,0,y,x]=median
                            out_c[b,0,y,x]=(np.sum(weights,dtype=np.float32)/np.float32(n))*(np.float32(n)/np.float32(9))
        current_d,current_c,current_v=out_d,out_c,out_v
    propagated=current_v&~direct
    return dict(depth=current_d,valid=current_v,confidence=current_c,direct_valid=direct,
                propagated_valid=propagated,provenance=direct.astype(np.uint8)+2*propagated.astype(np.uint8),support=support)


def depth_downsample(depth,valid,confidence,*,scale):
    # Reuse the existing independent per-level oracle rather than duplicating
    # its masked-INF/minimum/mean and first-tie confidence contracts.
    from sintergpu.references.vkl.numpy import depth_pyramid_level
    if scale not in (2,4,8):raise ValueError('scale must be2,4or8')
    current=dict(depth=depth,valid=valid,confidence=confidence)
    factor=1
    while factor<scale:
        if current['depth'].shape[-2:]==(1,1):raise ValueError('scale exceeds available depth pyramid levels')
        current=depth_pyramid_level(**current);factor*=2
    return dict(**current,direct_valid=current['valid'],propagated_valid=np.zeros(current['valid'].shape,bool),provenance=current['valid'].astype(np.uint8))
