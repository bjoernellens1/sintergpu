// SPDX-License-Identifier: Apache-2.0
// G18 iteration4: keep arithmetic phase order behind ABI-identical adapters.
// Derived from the frozen Apache-2.0 Caspar exports named in candidate.json.
#ifndef SINTERGPU_POINT_PLANE_INLINED
#define SINTERGPU_POINT_PLANE_INLINED
public float sinterPointPlaneResidual<let AoSoASchedule : bool>(float3 p,float3 q,float3 n) {
    float value=fma(q.z,-1.0,p.z);
    if(AoSoASchedule) {
        float dx=fma(q.x,-1.0,p.x);
        value=fma(n.x,dx,n.z*value);
        float dy=fma(q.y,-1.0,p.y);
        value=fma(n.y,dy,value);
    } else {
        float dy=fma(q.y,-1.0,p.y);
        value=fma(n.y,dy,n.z*value);
        float dx=fma(q.x,-1.0,p.x);
        value=fma(n.x,dx,value);
    }
    return value;
}
public float3 sinterPointPlaneRotation(float3 p,float3 n) {
    float x=fma(-1.0,p.z*n.y,p.y*n.z);
    float y=fma(p.z,n.x,-1.0*(p.x*n.z));
    float z=fma(-1.0,p.y*n.x,p.x*n.y);
    return float3(x,y,z);
}
public void sinterPointPlaneScalars<let AoSoASchedule : bool>(
    float px,float py,float pz,float qx,float qy,float qz,float nx,float ny,float nz,
    out float r,out float j0,out float j1,out float j2,out float j3,out float j4,out float j5) {
    float3 p=float3(px,py,pz),q=float3(qx,qy,qz),n=float3(nx,ny,nz);
    r=sinterPointPlaneResidual<AoSoASchedule>(p,q,n);
    float3 rotation=sinterPointPlaneRotation(p,n);
    j0=rotation.x;j1=rotation.y;j2=rotation.z;j3=nx;j4=ny;j5=nz;
}
#endif
