# SPDX-License-Identifier: Apache-2.0
"""Independent float64 mathematical oracles for prepared xsplat benchmark inputs.

Projection follows frozen xsplat-vk/src/reference.cpp algebra on finite, nonzero
quaternions and positive-definite covariances; rejected depth follows the shader
contract (the donor C++ reference instead returns zero there). Benchmark-only,
not an optimized CPU implementation or a runtime fallback.
"""
import numpy as np


def adam(values,gradients,first,second,*,step,learningRate,beta1,beta2,epsilon):
    # Uniforms originate as float32 on GPU. Round their input representation,
    # then evaluate independently in float64 rather than copying shader stages.
    lr,b1,b2,eps=map(lambda x:float(np.float32(x)),(learningRate,beta1,beta2,epsilon))
    x,g,m,v=(np.asarray(a,dtype=np.float64) for a in (values,gradients,first,second))
    m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g;t=max(step,1)
    updated=x-lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
    return tuple(a.astype(np.float32) for a in (updated,m,v))


def project(mean,logScale,rotation,worldToCamera,focal,principal,imageSize):
    mean,logScale,q,view,focal,principal=(np.asarray(a,dtype=np.float64) for a in (mean,logScale,rotation,worldToCamera,focal,principal))
    center=mean@view[:3,:3].T+view[:3,3]
    out=np.zeros((len(mean),7),np.float32);out[:,5]=center[:,2]
    valid=center[:,2]>float(np.float32(1e-4))
    if not np.any(valid):return out
    c=center[valid];q=q[valid]/np.linalg.norm(q[valid],axis=1,keepdims=True)
    x,y,z,w=q.T
    rot=np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=1).reshape(-1,3,3)
    scaled=rot*np.exp(logScale[valid])[:,None,:];cov=scaled@scaled.transpose(0,2,1)
    j=np.zeros((len(c),2,3));j[:,0,0]=focal[0]/c[:,2];j[:,1,1]=focal[1]/c[:,2]
    j[:,:,2]=-focal*c[:,:2]/c[:,2,None]**2
    t=j@view[:3,:3];cov2=t@cov@t.transpose(0,2,1)
    a=cov2[:,0,0]+float(np.float32(.3));b=cov2[:,0,1];d=cov2[:,1,1]+float(np.float32(.3))
    determinant=np.maximum(a*d-b*b,1e-12)
    pixel=c[:,:2]/c[:,2,None]*focal+principal
    out[valid,:2]=pixel;out[valid,2:5]=np.stack((d,-b,a),axis=1)/determinant[:,None]
    out[valid,6]=np.all((pixel>=0)&(pixel<np.asarray(imageSize)),axis=1)
    return out


def backward_probe():
    # Analytic derivative of a single alpha-compositing step, independent of AD.
    transmittance,alpha=.72,.31
    rgb=np.array([.8,.4,.2]);gradient=np.array([.7,-.2,.5]);dt=.13
    dot=float(rgb@gradient)
    return np.r_[transmittance*alpha*gradient,transmittance*(dot-dt),alpha*dot+(1-alpha)*dt].astype(np.float32)
