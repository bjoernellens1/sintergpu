# SPDX-License-Identifier: Apache-2.0
"""Independent float32 NumPy contracts for VKL geometry and depth-fusion ports.

These are numerical oracles, not runtime fallback or transaction policy. Source
field zero denotes the target in temporal fusion. Storage/profile conversions
are explicit at caller boundaries; this module owns only its CPU test arrays.
"""
import numpy as np

EPS=np.finfo(np.float32).eps


def world_transform(xyz,pose):
    xyz=np.asarray(xyz,dtype=np.float32);pose=np.asarray(pose,dtype=np.float32)
    if xyz.ndim!=4 or xyz.shape[1]!=3 or pose.shape!=(xyz.shape[0],4,4):
        raise ValueError('expected xyz [B,3,H,W] and pose [B,4,4]')
    return {'xyz_world':np.einsum('bij,bjhw->bihw',pose[:,:3,:3],xyz)+pose[:,:3,3,None,None]}


def depth_geometry(depth,intrinsics,pose=None,confidence=None):
    depth=np.asarray(depth,dtype=np.float32);k=np.asarray(intrinsics,dtype=np.float32)
    if depth.ndim!=4 or depth.shape[1]!=1 or k.shape!=(depth.shape[0],3,3):
        raise ValueError('expected depth [B,1,H,W] and intrinsics [B,3,3]')
    b,_,h,w=depth.shape
    y,x=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing='ij')
    rays=np.stack(((x[None]-k[:,0,2,None,None])/k[:,0,0,None,None],
                   (y[None]-k[:,1,2,None,None])/k[:,1,1,None,None],np.ones((b,h,w),np.float32)),axis=1)
    valid=np.isfinite(depth)&(depth>0)
    xyz=rays*np.where(valid,depth,np.float32(0))
    dx=np.zeros_like(xyz);dy=np.zeros_like(xyz)
    dx[:,:,:,:-1]=xyz[:,:,:,1:]-xyz[:,:,:,:-1]
    dy[:,:,:-1,:]=xyz[:,:,1:,:]-xyz[:,:,:-1,:]
    stencil=np.zeros_like(valid)
    with np.errstate(invalid='ignore',over='ignore'):
        stencil[:,:,:-1,:-1]=(valid[:,:,:-1,:-1]&valid[:,:,:-1,1:]&valid[:,:,1:,:-1]
            &(np.abs(depth[:,:,:-1,1:]-depth[:,:,:-1,:-1])<=np.abs(depth[:,:,:-1,:-1])*.25)
            &(np.abs(depth[:,:,1:,:-1]-depth[:,:,:-1,:-1])<=np.abs(depth[:,:,:-1,:-1])*.25))
        n=np.cross(dx,dy,axis=1)
        length=np.sqrt(np.sum(n*n,axis=1,keepdims=True))
        n=n/np.maximum(length,np.float32(1e-8))
        n=np.where(np.sum(n*rays,axis=1,keepdims=True)>0,-n,n)
        normals=np.where(stencil,n,np.float32(0))
    gc=stencil.astype(np.float32)
    if confidence is not None:
        with np.errstate(invalid='ignore'):gc=gc*np.asarray(confidence,dtype=np.float32)
    out={'valid':valid,'rays_camera':rays,'xyz_camera':xyz,'normals':normals,'geometry_confidence':gc}
    if pose is not None: out['xyz_world']=np.where(valid,world_transform(xyz,pose)['xyz_world'],np.float32(0))
    return out


def _fields(depths,valids,confidences):
    d=np.asarray(depths,dtype=np.float32);v=np.asarray(valids,dtype=bool);c=np.asarray(confidences,dtype=np.float32)
    if d.ndim!=5 or d.shape[2]!=1 or d.shape[0]<1 or d.shape!=v.shape or d.shape!=c.shape:
        raise ValueError('expected matching [S,B,1,H,W] fields')
    return d,v,c


def fuse_depth(depths,valids,confidences,*,relative=False,consistency_threshold=.05):
    d,v,c=_fields(depths,valids,confidences)
    shape=d.shape[1:]
    out={k:np.zeros(shape,np.float32) for k in ('depth','confidence')}
    out['valid']=np.zeros(shape,bool);out['winning_source']=np.full(shape,-1,np.int64)
    for index in np.ndindex(shape):
        p=(slice(None),)+index;z=d[p];conf=c[p]
        use=v[p]&np.isfinite(z)&(z>0)&(conf>0)
        if not use.any():continue
        weights=np.where(use,conf,np.float32(0));safe=np.where(use,z,np.float32(0))
        mean=np.sum(weights*safe,dtype=np.float32)/max(np.sum(weights,dtype=np.float32),EPS)
        agree=use&(np.abs(z-mean)<=np.float32(consistency_threshold)*max(mean,EPS))
        score=np.where(use,conf*np.float32(.5 if relative else 1),np.float32(0))
        winner=int(np.argmax(score));out['winning_source'][index]=winner;out['valid'][index]=True
        if np.any(agree) and np.count_nonzero(agree)==np.count_nonzero(use):
            w=np.where(agree,weights,np.float32(0));a=np.where(agree,z,np.float32(0))
            total=np.sum(w,dtype=np.float32)
            out['depth'][index]=np.sum(w*a,dtype=np.float32)/max(total,EPS)
            out['confidence'][index]=min(total,np.float32(1))
        else:
            out['depth'][index]=safe[winner];out['confidence'][index]=score[winner]
    out['metric_status']='RELATIVE' if relative else 'METRIC_ESTIMATED'
    return out


def temporal_depth_fuse(depths,valids,confidences,visibility,consistency,trusted,*,consistency_threshold=.05,min_support=2,min_temporal_support=1):
    d,v,c=_fields(depths,valids,confidences)
    vis=np.asarray(visibility,bool);con=np.asarray(consistency,bool);trust=np.asarray(trusted,bool)
    shape=d.shape[1:]
    floats=('depth','confidence','depth_variance','temporal_weight_fraction','source_disagreement')
    integers=('support_count','temporal_support_count','winning_source','conflict_owner_source','dominant_contributor')
    out={k:np.zeros(shape,np.float32) for k in floats};out.update({k:np.zeros(shape,np.int64) for k in integers});out['valid']=np.zeros(shape,bool)
    for index in np.ndindex(shape):
        p=(slice(None),)+index;z=d[p];conf=c[p];t=trust[p]
        evidence=(vis[p]&con[p]).copy();evidence[0]=True
        use=v[p]&evidence&np.isfinite(z)&(z>0)&(conf>0)
        score=np.where(use,np.where(t,np.float32(1)+conf,conf*np.float32(.5)),np.float32(0))
        if np.any(use&t):winner=int(np.argmax(score))
        else:
            scores=np.zeros(len(z),np.float32)
            for s in range(len(z)):
                if not use[s]:continue
                agree=use&use[s]&(np.abs(z-z[s])<=np.float32(consistency_threshold)*max(z[s],EPS))
                scores[s]=np.sum(np.where(agree,conf,np.float32(0)),dtype=np.float32)
            winner=int(np.argmax(scores))
        ref=z[winner]
        agree=use&(np.abs(z-ref)<=np.float32(consistency_threshold)*max(ref,EPS))
        support=np.count_nonzero(agree);temporal=np.count_nonzero(agree[1:])
        out['support_count'][index]=support;out['temporal_support_count'][index]=temporal
        valid=np.any(agree&t) or (support>=min_support and temporal>=min_temporal_support)
        if not valid:
            for k in ('winning_source','conflict_owner_source','dominant_contributor'):out[k][index]=-1
            continue
        weights=np.where(agree,conf,np.float32(0));total=np.sum(weights,dtype=np.float32)
        safe=np.where(agree,z,np.float32(0));fused=np.sum(weights*safe,dtype=np.float32)/max(total,EPS)
        residual=np.where(agree,z-fused,np.float32(0))
        dominant=int(np.argmax(weights))
        out['valid'][index]=True;out['depth'][index]=fused;out['confidence'][index]=min(total,np.float32(1))
        out['winning_source'][index]=dominant;out['dominant_contributor'][index]=dominant;out['conflict_owner_source'][index]=winner
        out['depth_variance'][index]=np.sum(weights*residual*residual,dtype=np.float32)/max(total,EPS)
        out['temporal_weight_fraction'][index]=np.sum(weights[1:],dtype=np.float32)/max(total,EPS)
        out['source_disagreement'][index]=np.max(np.where(use,np.abs(z-ref),np.float32(0)))
    return out
