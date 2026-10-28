# SPDX-License-Identifier: Apache-2.0
"""Input-derived FP32 center/pixel enclosures, not covariance error bounds.

Assumptions: finite FP32 inputs, no overflow, normal-range positive perspective
 denominator, no RelaxedPrecision. A four-term camera dot has at most4products
and3adds; any ordering/contraction is covered by gamma7*sum(abs(terms)). Use
u=2**-23 (one-ULP relative bound, covering rounding-mode choice), not a fitted
GPU tolerance. Add7*smallest_normal for possible denormal flushing. Include the
much smaller FP64 center-computation bound. Perspective intervals cover both
multiplication orders and division forms,1ULP per add/multiply,2.5ULP per divide.

Khronos: https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
The model leaves uncertain clipping/viewport classifications unresolved (-1).
It neither certifies conics nor relaxes their existing comparison.
"""
import numpy as np

TINY=float(np.finfo(np.float32).tiny)
U32=2.**-23
U64=2.**-53
G7=np.nextafter(7*U32/(1-7*U32),np.inf)
G7_64=np.nextafter(7*U64/(1-7*U64),np.inf)


def _outward(lo,hi):
    return np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf)


def _round(bounds,ulps=1.):
    lo,hi=bounds
    magnitude=np.maximum(np.abs(lo),np.abs(hi))
    if not np.all(np.isfinite(magnitude)) or np.any(magnitude>=np.finfo(np.float32).max):raise ValueError('finite no-overflow arithmetic required')
    rounded=magnitude.astype(np.float32)
    spacing=np.nextafter(rounded,np.float32(np.inf)).astype(np.float64)-rounded.astype(np.float64)
    error=ulps*spacing+TINY
    return _outward(lo-error,hi+error)


def _mul(a,b):
    terms=np.array([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
    return _round(_outward(np.min(terms,axis=0),np.max(terms,axis=0)))


def _add(a,b):
    return _round(_outward(a[0]+b[0],a[1]+b[1]))


def _divide(a,b):
    if np.any(b[0]<=0):raise ValueError('positive bounded denominator required')
    if np.any(b[0]<TINY) or np.any(b[1]>2.**126):raise ValueError('normal-range division denominator required')
    terms=np.array([a[0]/b[0],a[0]/b[1],a[1]/b[0],a[1]/b[1]])
    return _round(_outward(np.min(terms,axis=0),np.max(terms,axis=0)),2.5)


def projected_pixel_bounds(mean,worldToCamera,focal,principal,imageSize):
    # Casting matches the actual uploaded data, not any higher-precision caller.
    points,view,f,p=(np.asarray(x,dtype=np.float32).astype(np.float64) for x in (mean,worldToCamera,focal,principal))
    image=np.asarray(imageSize,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or view.shape!=(4,4) or f.shape!=(2,) or p.shape!=(2,) or image.shape!=(2,):raise ValueError('projection dimensions invalid')
    if not all(np.all(np.isfinite(x)) for x in (points,view,f,p,image)) or np.any(image<0) or np.any(image>2.**24) or np.any(image!=np.floor(image)):raise ValueError('finite inputs and exactly represented viewport required')
    if any(np.any((np.abs(x)>0)&(np.abs(x)<TINY)) for x in (points,view,f,p)):raise ValueError('subnormal source inputs are outside this model')
    terms=points[:,None,:]*view[None,:3,:3]
    absolute_sum=np.sum(np.abs(terms),axis=2)+np.abs(view[:3,3])
    center=np.sum(terms,axis=2)+view[:3,3]
    if np.any(absolute_sum>=np.finfo(np.float32).max):raise ValueError('intermediate camera arithmetic may overflow')
    sum_upper=np.nextafter(absolute_sum/(1-G7_64),np.inf)
    factor=np.nextafter(G7+G7_64,np.inf)
    rounding=np.nextafter(factor*sum_upper,np.inf)
    flushing=np.nextafter(7*TINY/(1-7*U32),np.inf)
    radius=np.nextafter(rounding+flushing,np.inf)
    clo,chi=_outward(center-radius,center+radius)
    if np.any(np.maximum(np.abs(clo),np.abs(chi))>=np.finfo(np.float32).max):raise ValueError('camera transform may overflow')
    threshold=float(np.float32(1e-4));valid=clo[:,2]>threshold;rejected=chi[:,2]<=threshold
    lower=np.zeros((len(points),2),np.float64);upper=lower.copy();classification=np.full(len(points),-1,np.int8);classification[rejected]=0
    if np.any(valid):
        xy=(clo[valid,:2],chi[valid,:2]);z=(clo[valid,2,None],chi[valid,2,None])
        focal_interval=(np.broadcast_to(f,xy[0].shape),)*2;principal_interval=(np.broadcast_to(p,xy[0].shape),)*2
        reciprocal=_divide((np.ones_like(z[0]),np.ones_like(z[1])),z)
        variants=[_add(_mul(_mul(focal_interval,xy),reciprocal),principal_interval),_add(_mul(focal_interval,_mul(xy,reciprocal)),principal_interval),_add(_mul(focal_interval,_divide(xy,z)),principal_interval),_add(_divide(_mul(focal_interval,xy),z),principal_interval)]
        lower[valid]=np.min([x[0] for x in variants],axis=0);upper[valid]=np.max([x[1] for x in variants],axis=0)
        inside=np.all((lower[valid]>=0)&(upper[valid]<image),axis=1)
        outside=np.any((upper[valid]<0)|(lower[valid]>=image),axis=1)
        indices=np.flatnonzero(valid);classification[indices[inside]]=1;classification[indices[outside]]=0
    return dict(lower=lower,upper=upper,center_lower=clo,center_upper=chi,classification=classification)


def validate_pixels(pixels,visible,bounds):
    pixels=np.asarray(pixels);visible=np.asarray(visible)
    assert pixels.shape==bounds['lower'].shape and visible.shape==bounds['classification'].shape
    assert np.all(bounds['classification']>=0), 'depth/viewport classification is numerically unresolved; qualification pending'
    assert np.all(np.isfinite(pixels))
    np.testing.assert_array_equal(visible,bounds['classification'])
    assert np.all(pixels>=bounds['lower']) and np.all(pixels<=bounds['upper']), 'pixel coordinate outside input-derived FP32 enclosure'
