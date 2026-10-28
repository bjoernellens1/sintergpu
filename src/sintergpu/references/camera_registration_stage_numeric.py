# SPDX-License-Identifier: Apache-2.0
"""Independent CPU models for individual camera dispatches.

Bindings are shader binding names mapped to preallocated, flat NumPy arrays.
Only live elements of written buffers are modified; padded tails are preserved.
Diagnostic int64 buffers use the host logical representation of shader uint2.
"""
import numpy as np

from .camera_registration_numeric import _brown, _fmaf

_F = np.float32
_EPS = _F(np.finfo(np.float32).eps)


def _brown_flat(x, y, coefficients, batches):
    # The shared independent scalar polynomial expects [batch, height, width].
    shape = x.shape
    a, b = _brown(x.reshape(batches, 1, -1), y.reshape(batches, 1, -1),
                  coefficients.reshape(batches, 5))
    return a.reshape(shape), b.reshape(shape)


def _xy(k, batches, height, width):
    yy, xx = np.indices((height, width), dtype=np.float32)
    k = k[:batches * 9].reshape(batches, 3, 3)
    return xx, yy, ((xx[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]), ((yy[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None])


def run_stage(entrypoint, params, bindings):
    """Execute exactly one modeled dispatch, returning its written bindings."""
    p = params
    b = bindings
    n = p['batches']
    written = []
    def put(name, values):
        values = np.asarray(values).reshape(-1)
        b[name][:values.size] = values
        written.append(name)
    with np.errstate(all='ignore'):
        if entrypoint == 'rectifyGrid':
            h, w = p['targetHeight'], p['targetWidth']
            x, y, rx, ry = _xy(b['targetIntrinsics'], n, h, w)
            dx, dy = (_brown_flat(rx, ry, b['sourceDistortion'][:n*5], n)
                      if p['distortion'] else (rx, ry))
            k = b['sourceIntrinsics'][:n*9].reshape(n,3,3)
            u = dx*k[:,0,0,None,None] + k[:,0,2,None,None]
            v = dy*k[:,1,1,None,None] + k[:,1,2,None,None]
            tol = _F(16)*_EPS*_F(max(p['sourceHeight'],p['sourceWidth']))
            valid = np.isfinite(u)&np.isfinite(v)&(u >= -tol)&(u <= _F(p['sourceWidth']-1)+tol)&(v >= -tol)&(v <= _F(p['sourceHeight']-1)+tol)
            put('sourceUVOut', np.stack((u,v), axis=-1)); put('validOut', valid)
            if p['debug']:
                put('targetRayOut', np.stack((rx,ry,np.ones_like(rx)),axis=-1))
                put('displacementOut', np.sqrt((u-x)*(u-x)+(v-y)*(v-y)))
        elif entrypoint == 'depthCameraRays':
            h,w = p['depthHeight'],p['depthWidth']
            _,_,xd,yd = _xy(b['depthIntrinsics'],n,h,w)
            x,y=xd,yd
            if p['depthDistortion']:
                for _ in range(5):
                    px,py=_brown_flat(x,y,b['depthDistortion'][:n*5],n)
                    x,y=x+xd-px,y+yd-py
            put('depthRaysOut',np.stack((x,y),axis=-1))
        elif entrypoint == 'rectifyFromGrid':
            c,sh,sw,th,tw = (p[k] for k in ('channels','sourceHeight','sourceWidth','targetHeight','targetWidth'))
            image=b['image'][:n*c*sh*sw].reshape(n,c,sh,sw)
            uv=b['sourceUV'][:n*th*tw*2].reshape(n,th,tw,2)
            valid=b['valid'][:n*th*tw].reshape(n,th,tw)
            out=b['imageOut']
            for bi in range(n):
                for j in range(th):
                    for i in range(tw):
                        u,v=uv[bi,j,i]
                        u=_F(np.clip(u if np.isfinite(u) else 0,0,sw-1)); v=_F(np.clip(v if np.isfinite(v) else 0,0,sh-1))
                        gx=_F(_F(_F(2)*u)/_F(max(sw-1,1))-_F(1)); gy=_F(_F(_F(2)*v)/_F(max(sh-1,1))-_F(1))
                        x=_F(_F(gx+_F(1))*_F(sw-1)/_F(2)); y=_F(_F(gy+_F(1))*_F(sh-1)/_F(2))
                        for channel in range(c):
                            idx=((bi*c+channel)*th+j)*tw+i
                            if not valid[bi,j,i]: out[idx]=0; continue
                            if p['sampling']==0:
                                out[idx]=image[bi,channel,int(np.rint(y)),int(np.rint(x))]; continue
                            x0,y0=int(np.floor(x)),int(np.floor(y)); dx,dy=_F(x-x0),_F(y-y0); value=_F(0)
                            for yy,wy in ((y0,_F(1)-dy),(y0+1,dy)):
                                for xx,wx in ((x0,_F(1)-dx),(x0+1,dx)):
                                    if 0<=xx<sw and 0<=yy<sh:
                                        value=_F(_fmaf(image[bi,channel,yy,xx],_F(wx*wy),value))
                            out[idx]=value
            written.append('imageOut')
        elif entrypoint == 'clearRegistration':
            count=n*p['targetHeight']*p['targetWidth']
            put('minDepthBits',np.full(count,0x7f800000,np.uint32))
            put('winnerIndex',np.full(count,p['points'],np.uint32))
            put('collisionCount',np.zeros(count,np.uint32))
        elif entrypoint == 'projectRegistration':
            points=p['points']; total=n*points
            d=b['depth'][:total].reshape(n,points)
            dv=np.isfinite(d)&(d>=_F(p['minDepth']))
            if p['hasMaxDepth']: dv &= d<=_F(p['maxDepth'])
            safe=np.where(dv,d,_F(0)); rays=b['depthRays'][:total*2].reshape(n,points,2)
            px,py=rays[:,:,0]*safe,rays[:,:,1]*safe
            t=b['transformRgbDepth'][:n*16].reshape(n,4,4)
            xyz=[((px*t[:,None,j,0]+py*t[:,None,j,1])+safe*t[:,None,j,2])+t[:,None,j,3] for j in range(3)]
            z=xyz[2]; positive=np.isfinite(z)&(z>_EPS); sz=np.where(positive,z,_F(1))
            x,y=xyz[0]/sz,xyz[1]/sz
            if p['rgbDistortion']: x,y=_brown_flat(x,y,b['rgbDistortion'][:n*5],n)
            k=b['rgbIntrinsics'][:n*9].reshape(n,3,3)
            u=x*k[:,None,0,0]+k[:,None,0,2]; v=y*k[:,None,1,1]+k[:,None,1,2]
            candidate=dv&positive&np.isfinite(u)&np.isfinite(v)&(u>=_F(-.5))&(v>=_F(-.5))&(u<=_F(p['targetWidth'])-_F(.5))&(v<=_F(p['targetHeight'])-_F(.5))
            ru=np.rint(np.where(candidate,u,_F(0))); rv=np.rint(np.where(candidate,v,_F(0)))
            ok=candidate&(ru>=0)&(ru<p['targetWidth'])&(rv>=0)&(rv<p['targetHeight'])
            destination=np.where(ok,np.arange(n,dtype=np.uint64)[:,None]*(p['targetHeight']*p['targetWidth'])+rv.astype(np.uint64)*p['targetWidth']+ru.astype(np.uint64),0)
            put('sourceUVOut',np.stack((u,v),axis=-1)); put('projectedZOut',z)
            put('sourceValidOut',ok); put('destinationIndexOut',destination)
        elif entrypoint in ('registrationMin','registrationWinnerCount'):
            total=n*p['points']; valid=b['sourceValidOut'][:total]!=0
            destinations=b['destinationIndexOut'][:total][valid].astype(np.intp)
            zbits=b['projectedZOut'][:total].view(np.uint32)[valid]
            if entrypoint=='registrationMin':
                np.minimum.at(b['minDepthBits'],destinations,zbits); written.append('minDepthBits')
            else:
                np.add.at(b['collisionCount'],destinations,np.uint32(1))
                indices=np.arange(total,dtype=np.uint64)[valid] % p['points']
                matches=zbits==b['minDepthBits'][destinations]
                np.minimum.at(b['winnerIndex'],destinations[matches],indices[matches].astype(np.uint32))
                written.extend(('winnerIndex','collisionCount'))
        elif entrypoint in ('finalizeRegistration','finalizeRegistrationDebug'):
            count=n*p['targetHeight']*p['targetWidth']
            z=b['minDepthBits'][:count].view(np.float32); valid=np.isfinite(z)
            put('depthOut',np.where(valid,z,_F(0))); put('validOut',valid)
            if entrypoint=='finalizeRegistrationDebug':
                put('winnerSourceIndexOut',np.where(valid,b['winnerIndex'][:count].astype(np.int64),-1))
                put('collisionCountOut',b['collisionCount'][:count].astype(np.int64))
        else:
            raise ValueError('unknown camera registration stage: '+str(entrypoint))
    return {name:b[name] for name in written}
