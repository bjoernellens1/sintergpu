// SPDX-License-Identifier: Apache-2.0
// Exact integer minima/ties; stream candidate chunks using caller-owned scratch.
#include "../common/float32_bits.slang"
struct MatchParams { uint batches,landmarks,candidates,bits,words,chunkStart,chunkCount,maxDistance,mutual,useLandmarkValid,useCandidateValid; float radiusSquared; };
ConstantBuffer<MatchParams> Params;
StructuredBuffer<uint> landmarkDescriptor,candidateDescriptor,landmarkValid,candidateValid;
StructuredBuffer<float> projected;
StructuredBuffer<int64_t> candidateXY;
RWStructuredBuffer<uint> landmarkPacked,candidatePacked,pairDistance,bestDistance,bestIndex,reverseIndex;
RWStructuredBuffer<int64_t> matchedIndexOut,distanceOut;
RWStructuredBuffer<uint> matchValidOut;
groupshared uint minimumValue[256],minimumIndex[256];

[shader("compute")]
[numthreads(256,1,1)]
void packBinary(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x,totalA=Params.batches*Params.landmarks*Params.words;
    if(i>=totalA+Params.batches*Params.candidates*Params.words) return;
    bool first=i<totalA;uint at=first?i:i-totalA,word=at%Params.words,point=at/Params.words;
    uint value=0;
    for(uint k=0;k<32;k++) { uint bit=word*32+k;if(bit<Params.bits) { uint input;[branch] if(first) input=landmarkDescriptor[point*Params.bits+bit];else input=candidateDescriptor[point*Params.bits+bit];value|=uint(input!=0)<<k; } }
    if(first) landmarkPacked[at]=value;else candidatePacked[at]=value;
}

[shader("compute")]
[numthreads(256,1,1)]
void initializeMatch(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.batches*Params.landmarks) return;
    bestDistance[id.x]=Params.bits+1;bestIndex[id.x]=0;
}

[shader("compute")]
[numthreads(256,1,1)]
void matchPairs(uint3 id:SV_DispatchThreadID)
{
    uint i=id.x;if(i>=Params.batches*Params.landmarks*Params.chunkCount) return;
    uint c=i%Params.chunkCount+Params.chunkStart,m=(i/Params.chunkCount)%Params.landmarks,b=i/(Params.chunkCount*Params.landmarks);
    uint row=b*Params.landmarks+m,col=b*Params.candidates+c;
    float x=projected[row*2],y=projected[row*2+1];bool local=isfinite(x) && isfinite(y);
    if(Params.useLandmarkValid!=0) local=local && landmarkValid[row]!=0;
    if(Params.useCandidateValid!=0) local=local && candidateValid[col]!=0;
    if(local)
    {
        uint dx=sintergpuF32AddBits(asuint(x),asuint(float(candidateXY[col*2]))^0x80000000u);
        uint dy=sintergpuF32AddBits(asuint(y),asuint(float(candidateXY[col*2+1]))^0x80000000u);
        uint distance=sintergpuF32AddBits(sintergpuF32MulBits(dx,dx),sintergpuF32MulBits(dy,dy));
        local=distance<=asuint(Params.radiusSquared);
    }
    uint distance=Params.bits+1;
    if(local) { distance=0;for(uint w=0;w<Params.words;w++) distance+=countbits(landmarkPacked[row*Params.words+w]^candidatePacked[col*Params.words+w]); }
    pairDistance[i]=distance;
}

void reduceMinimum(uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=128;stride!=0;stride/=2)
    {
        if(lane<stride)
        {
            uint a=minimumValue[lane],b=minimumValue[lane+stride],bi=minimumIndex[lane+stride];
            if(b<a || (b==a && bi<minimumIndex[lane])) { minimumValue[lane]=b;minimumIndex[lane]=bi; }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256,1,1)]
void rowMinimum(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint row=group.x;if(row>=Params.batches*Params.landmarks) return;
    uint value=Params.bits+1,index=0;
    for(uint c=lane;c<Params.chunkCount;c+=256) { uint d=pairDistance[row*Params.chunkCount+c];if(d<value || (d==value && c<index)) { value=d;index=c; } }
    minimumValue[lane]=value;minimumIndex[lane]=index;reduceMinimum(lane);
    if(lane==0 && minimumValue[0]<bestDistance[row]) { bestDistance[row]=minimumValue[0];bestIndex[row]=minimumIndex[0]+Params.chunkStart; }
}

[shader("compute")]
[numthreads(256,1,1)]
void reverseMinimum(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint i=group.x;if(i>=Params.batches*Params.chunkCount) return;
    uint b=i/Params.chunkCount,c=i%Params.chunkCount,value=Params.bits+1,index=0;
    for(uint m=lane;m<Params.landmarks;m+=256) { uint d=pairDistance[(b*Params.landmarks+m)*Params.chunkCount+c];if(d<value || (d==value && m<index)) { value=d;index=m; } }
    minimumValue[lane]=value;minimumIndex[lane]=index;reduceMinimum(lane);
    if(lane==0) reverseIndex[b*Params.candidates+Params.chunkStart+c]=minimumIndex[0];
}

[shader("compute")]
[numthreads(256,1,1)]
void finalizeMatch(uint3 id:SV_DispatchThreadID)
{
    uint row=id.x;if(row>=Params.batches*Params.landmarks) return;
    uint b=row/Params.landmarks,m=row%Params.landmarks,index=bestIndex[row],distance=bestDistance[row];
    bool valid=isfinite(projected[row*2]) && isfinite(projected[row*2+1]) && index<Params.candidates && distance<=Params.maxDistance;
    if(Params.useLandmarkValid!=0) valid=valid && landmarkValid[row]!=0;
    if(valid && Params.mutual!=0) valid=reverseIndex[b*Params.candidates+index]==m;
    matchedIndexOut[row]=valid?int64_t(index):int64_t(-1);distanceOut[row]=valid?int64_t(distance):int64_t(0);matchValidOut[row]=uint(valid);
}
