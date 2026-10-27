# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy float32/int64 contracts for VKL binary features."""
import math
import numpy as np

MAX_DEFAULT_BITS=961*960


def default_pairs(bits):
    if type(bits) is not int or not 1<=bits<=MAX_DEFAULT_BITS:raise ValueError('invalid default BRIEF bit count')
    code=(np.arange(bits,dtype=np.int64)*104729+13007)%MAX_DEFAULT_BITS
    first=code//960;second=code%960;second+=second>=first
    xy=lambda p:np.stack((p%961%31-15,p//31-15),axis=-1)
    return np.stack((xy(first),xy(second)),axis=1)


def _mask(value,shape):
    if value is None:return np.ones(shape,bool)
    value=np.asarray(value)
    if value.shape!=shape or value.dtype!=np.bool_:raise ValueError('boolean mask shape mismatch')
    return value


def brief_describe(rgb,pixels_xy,*,pixels_valid=None,rgb_valid=None,pairs=None,num_bits=256):
    rgb=np.asarray(rgb);pixels=np.asarray(pixels_xy)
    if rgb.ndim!=4 or rgb.shape[1] not in (1,3) or rgb.dtype!=np.float32:raise ValueError('float32 B1|3HW required')
    b,c,h,w=rgb.shape
    if min(b,h,w)<1 or pixels.ndim!=3 or pixels.shape[0]!=b or pixels.shape[2]!=2 or pixels.dtype!=np.int64:raise ValueError('int64 BP2 and positive image extent required')
    pv=_mask(pixels_valid,pixels.shape[:2]);domain=np.isfinite(rgb).all(1)&_mask(rgb_valid,(b,1,h,w))[:,0]
    pairs=default_pairs(num_bits) if pairs is None else np.asarray(pairs)
    if pairs.ndim!=3 or pairs.shape[0]<1 or pairs.shape[1:]!=(2,2) or pairs.dtype!=np.int64 or np.any(np.all(pairs[:,0]==pairs[:,1],axis=-1)):raise ValueError('distinct int64 D22 pair offsets required')
    safe=np.where(domain[:,None],rgb,np.float32(0))
    gray=safe[:,0] if c==1 else (safe[:,0]*np.float32(.299)+safe[:,1]*np.float32(.587))+safe[:,2]*np.float32(.114)
    x,y=pixels[...,0],pixels[...,1];inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    batch=np.arange(b)[:,None]
    center=domain[batch,np.clip(y,0,h-1),np.clip(x,0,w-1)]
    samples=pixels[:,:,None,None,:]+pairs[None,None]
    sx,sy=samples[...,0],samples[...,1];supported=(sx>=0)&(sx<w)&(sy>=0)&(sy<h)
    xx=np.clip(sx,0,w-1);yy=np.clip(sy,0,h-1)
    sampled_domain=domain[np.arange(b)[:,None,None,None],yy,xx]
    valid=pv&inside&center&supported.all(axis=(-1,-2))&sampled_domain.all(axis=(-1,-2))
    values=gray[np.arange(b)[:,None,None,None],yy,xx]
    return dict(descriptor=(values[...,0]<values[...,1])&valid[...,None],valid=valid)


def binary_local_match(landmark_descriptor,candidate_descriptor,projected_xy,candidate_xy,*,radius_px,max_hamming_distance,mutual=True,landmark_valid=None,candidate_valid=None,candidate_chunk_size=None):
    a=np.asarray(landmark_descriptor);c=np.asarray(candidate_descriptor);xy=np.asarray(projected_xy);cp=np.asarray(candidate_xy)
    if a.ndim!=3 or c.ndim!=3 or a.dtype!=np.bool_ or c.dtype!=np.bool_:raise ValueError('boolean BMD/BPD required')
    b,m,d=a.shape;p=c.shape[1]
    if min(b,m,d,p)<1 or c.shape!=(b,p,d) or xy.shape!=(b,m,2) or xy.dtype!=np.float32 or cp.shape!=(b,p,2) or cp.dtype!=np.int64:raise ValueError('incompatible descriptor/float32 projection/int64 candidate shapes')
    if isinstance(radius_px,bool) or not isinstance(radius_px,(int,float)) or not math.isfinite(radius_px) or radius_px<=0:raise ValueError('positive finite radius required')
    if type(max_hamming_distance) is not int or not 0<=max_hamming_distance<=d or type(mutual) is not bool:raise ValueError('invalid distance/mutual')
    if candidate_chunk_size is not None and (type(candidate_chunk_size) is not int or candidate_chunk_size<1):raise ValueError('positive chunk size required')
    av=_mask(landmark_valid,(b,m))&np.isfinite(xy).all(-1);cv=_mask(candidate_valid,(b,p))
    step=p if candidate_chunk_size is None else min(p,candidate_chunk_size)
    best=np.full((b,m),d+1,np.int64);index=np.zeros((b,m),np.int64);reverse=np.zeros((b,p),np.int64)
    with np.errstate(over='ignore',invalid='ignore'):
        radius_squared=np.float32(radius_px*radius_px)
        for start in range(0,p,step):
            end=min(p,start+step);delta=xy[:,:,None]-cp[:,None,start:end].astype(np.float32)
            local=np.sum(delta*delta,axis=-1,dtype=np.float32)<=radius_squared
            local&=av[:,:,None]&cv[:,None,start:end]
            distances=np.logical_xor(a[:,:,None],c[:,None,start:end]).sum(-1,dtype=np.int64)
            distances=np.where(local,distances,d+1);offset=distances.argmin(-1);value=np.take_along_axis(distances,offset[...,None],axis=-1)[...,0]
            improve=value<best;best=np.where(improve,value,best);index=np.where(improve,offset+start,index)
            if mutual:reverse[:,start:end]=distances.argmin(1)
    valid=av&(best<=max_hamming_distance)
    if mutual:valid&=np.take_along_axis(reverse,index,axis=1)==np.arange(m)[None]
    return dict(matched_index=np.where(valid,index,-1),distance=np.where(valid,best,0),valid=valid)
