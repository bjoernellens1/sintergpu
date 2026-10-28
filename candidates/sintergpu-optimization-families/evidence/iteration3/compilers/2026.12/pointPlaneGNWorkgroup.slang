// SPDX-License-Identifier: Apache-2.0
// G18/G11 fused candidate: preserves original AoSoA output and reduction trees.
// No change to the downstream reducePartials or solveSystem implementations.
#include "point_plane.slang"
#include "normal_terms.slang"
struct PointPlaneGNParams {uint count;};
ConstantBuffer<PointPlaneGNParams> cfg;
StructuredBuffer<float> points, targets, normals;
StructuredBuffer<uint> points_indices;
RWStructuredBuffer<float> out_0, out_1, outputTerms;
groupshared float terms[43][64];

void sinterBuildPointPlane(uint i, uint lane, uint group, bool useWave) {
    float r=0.0;float J[6];for(uint c=0;c<6;++c)J[c]=0.0;
    if(i<cfg.count) {
        uint p=points_indices[i];
        SinterPointPlaneFactor f=sinterPointPlane<true>(
            float3(points[p*4],points[p*4+1],points[p*4+2]),
            float3(targets[i*4],targets[i*4+1],targets[i*4+2]),
            float3(normals[i*4],normals[i*4+1],normals[i*4+2]));
        r=f.residual;J[0]=f.rotation.x;J[1]=f.rotation.y;J[2]=f.rotation.z;
        J[3]=f.translation.x;J[4]=f.translation.y;J[5]=f.translation.z;
        out_0[i]=r;
        for(uint c=0;c<4;++c)out_1[i*4+c]=J[c];
        out_1[cfg.count*4+i*2]=J[4];out_1[cfg.count*4+i*2+1]=J[5];
    }
    for(uint t=0;t<43;++t) {
        float value=i<cfg.count?sinterNormalTerm<6>(t,r,J):0.0;
        if(useWave) {float sum=WaveActiveSum(value);value=WaveIsFirstLane()?sum:0.0;}
        terms[t][lane]=value;
    }
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=32;stride>0;stride>>=1) {
        if(lane<stride)for(uint t=0;t<43;++t)terms[t][lane]+=terms[t][lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(lane==0)for(uint t=0;t<43;++t)outputTerms[group*43+t]=terms[t][0];
}
[shader("compute")][numthreads(64,1,1)]
void pointPlaneGNWorkgroup(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint lane:SV_GroupIndex) {
    sinterBuildPointPlane(tid.x,lane,gid.x,false);
}
[shader("compute")][numthreads(64,1,1)]
void pointPlaneGNSubgroup(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint lane:SV_GroupIndex) {
    sinterBuildPointPlane(tid.x,lane,gid.x,true);
}
