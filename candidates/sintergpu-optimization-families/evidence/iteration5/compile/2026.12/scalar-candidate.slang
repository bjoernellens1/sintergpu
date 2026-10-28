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
[[vk::binding(0, 0)]] StructuredBuffer<float> points;
[[vk::binding(1, 0)]] StructuredBuffer<float> targets;
[[vk::binding(2, 0)]] StructuredBuffer<float> normals;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_0;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> out_1;
[[vk::binding(5, 0)]] StructuredBuffer<uint> points_indices;

struct CasparParams
{
        uint problem_size;
    uint points_num_alloc;
    uint targets_num_alloc;
    uint normals_num_alloc;
    uint out_0_num_alloc;
    uint out_1_num_alloc;
};

[[vk::binding(6, 0)]]
ConstantBuffer<CasparParams> params;


[shader("compute")][numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=params.problem_size)return;uint p=points_indices[i];
 float3 point=float3(points[p*4],points[p*4+1],points[p*4+2]);
 float3 target=float3(targets[i*4],targets[i*4+1],targets[i*4+2]);
 float3 normal=float3(normals[i*4],normals[i*4+1],normals[i*4+2]);
 float r,j0,j1,j2,j3,j4,j5;

 sinterPointPlaneScalars<false>(point.x,point.y,point.z,target.x,target.y,target.z,normal.x,normal.y,normal.z,r,j0,j1,j2,j3,j4,j5);

 out_0[i]=r;out_1[i*4]=j0;out_1[i*4+1]=j1;out_1[i*4+2]=j2;out_1[i*4+3]=j3;
 out_1[4*params.out_1_num_alloc+i*2]=j4;out_1[4*params.out_1_num_alloc+i*2+1]=j5;
}
