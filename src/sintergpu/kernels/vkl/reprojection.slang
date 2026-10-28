// SPDX-License-Identifier: Apache-2.0
// D4/T0 float32 caller-owned stages. Planar B,C,H,W / B,N,C,H,W.
// No device state, allocation, readback, scheduling, or float atomics.
#include "../common/float32_divide.slang"
#include "../common/float32_multiply.slang"
struct ReprojectionParams
{
    uint pairs; uint height; uint width; uint targetHeight; uint targetWidth;
    uint geometryRepeat; uint sourceCameraRepeat; uint targetCameraRepeat;
    uint targetDepthRepeat; uint useTargetDepth; uint useConfidence;
    uint groupsPerPair; uint neighbors; uint batches; float threshold;
};
ConstantBuffer<ReprojectionParams> params;
StructuredBuffer<float> cameraXYZ;
StructuredBuffer<uint> geometryValid;
StructuredBuffer<float> sourcePose;
StructuredBuffer<float> targetIntrinsics;
StructuredBuffer<float> targetPose;
StructuredBuffer<float> targetDepth;
StructuredBuffer<float> sourceRGB;
StructuredBuffer<float> sourceConfidence;
StructuredBuffer<float> rasterConfidence;
StructuredBuffer<uint> rasterValid;
StructuredBuffer<float> fusedDepth;
StructuredBuffer<uint> fusedValid;
StructuredBuffer<float> fusedConfidence;
RWStructuredBuffer<float> projected_xyOut;
RWStructuredBuffer<float> projected_depthOut;
RWStructuredBuffer<uint> visibleOut;
RWStructuredBuffer<float> displacementOut;
RWStructuredBuffer<float> overlapOut;
RWStructuredBuffer<float> parallaxOut;
RWStructuredBuffer<float> depth_consistencyOut;
RWStructuredBuffer<uint> depth_consistency_validOut;
RWStructuredBuffer<uint> occludedOut;
RWStructuredBuffer<uint> occlusion_validOut;
RWStructuredBuffer<uint> partialCount;
RWStructuredBuffer<float> partialParallax;
RWStructuredBuffer<uint> minDepthBits;
RWStructuredBuffer<uint> maxConfidenceBits;
RWStructuredBuffer<uint> supportBits;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
RWStructuredBuffer<uint> support_countOut;
RWStructuredBuffer<float> aligned_rgbOut;
RWStructuredBuffer<uint> aligned_validOut;
RWStructuredBuffer<float> alignment_confidenceOut;
RWStructuredBuffer<float> packedDepth;
RWStructuredBuffer<uint> packedValid;
RWStructuredBuffer<float> packedConfidence;
RWStructuredBuffer<uint> packedTrusted;
RWStructuredBuffer<float> rgbOut;
static const float EPS=1.1920928955078125e-7;
groupshared uint countTree[256];
groupshared float parallaxTree[256];

float ratio(float a,float b) { return asfloat(sintergpuF32DivideBits(asuint(a),asuint(b))); }
float mul(float a,float b) { return sintergpuF32MulNoContract(a,b); }
float dotCamera(float a,float b,float c,float3 p)
{
    return (mul(a,p.x)+mul(b,p.y))+mul(c,p.z);
}
float normalizedRoundtrip(float pixel,uint size)
{
    float normalized=ratio(mul(2.0,pixel),float(max(size-1,1u)))-1.0;
    return mul(ratio(normalized+1.0,2.0),float(size-1));
}
int nearestEven(float value)
{
    float lower=floor(value),fraction=value-lower;
    int result=int(lower);
    if(fraction>.5||(fraction==.5&&(result&1)!=0))result++;
    return result;
}
float nearestDepth(uint pair,float2 pixel)
{
    float u=normalizedRoundtrip(pixel.x,params.targetWidth);
    float v=normalizedRoundtrip(pixel.y,params.targetHeight);
    if(!isfinite(u)||!isfinite(v)||u<-.5||v<-.5||u>float(params.targetWidth)-.5||v>float(params.targetHeight)-.5)return 0.0;
    int x=nearestEven(u),y=nearestEven(v);
    if(x<0||y<0||x>=int(params.targetWidth)||y>=int(params.targetHeight))return 0.0;
    return targetDepth[(pair/params.targetDepthRepeat)*params.targetHeight*params.targetWidth+uint(y)*params.targetWidth+uint(x)];
}
float bilinearTap(StructuredBuffer<float> field,uint base,uint h,uint w,int x,int y,float wx,float wy)
{
    if(x<0||y<0||x>=int(w)||y>=int(h))return 0.0;
    return mul(mul(field[base+uint(y)*w+uint(x)],wx),wy);
}
float bilinear(StructuredBuffer<float> field,uint base,float2 pixel)
{
    uint h=params.targetHeight,w=params.targetWidth;
    float u=normalizedRoundtrip(pixel.x,w),v=normalizedRoundtrip(pixel.y,h);
    if(!isfinite(u)||!isfinite(v))return asfloat(0x7fc00000);
    if(u< -1||v< -1||u>float(w)||v>float(h))return 0.0;
    int x=int(floor(u)),y=int(floor(v));
    float wx0=float(x+1)-u,wy0=float(y+1)-v,wx1=u-float(x),wy1=v-float(y);
    float result=bilinearTap(field,base,h,w,x,y,wx0,wy0);
    result+=bilinearTap(field,base,h,w,x+1,y,wx1,wy0);
    result+=bilinearTap(field,base,h,w,x,y+1,wx0,wy1);
    result+=bilinearTap(field,base,h,w,x+1,y+1,wx1,wy1);
    return result;
}
void reduceTrees(uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint stride=128;stride>0;stride>>=1)
    {
        if(lane<stride){countTree[lane]+=countTree[lane+stride];parallaxTree[lane]+=parallaxTree[lane+stride];}
        GroupMemoryBarrierWithGroupSync();
    }
}
[shader("compute")]
[numthreads(256,1,1)]
void projectPoints(uint3 group : SV_GroupID,uint3 local : SV_GroupThreadID)
{
    uint lane=local.x,plane=params.height*params.width;
    uint pair=group.x/params.groupsPerPair,q=(group.x%params.groupsPerPair)*256+lane;
    uint visible=0;float displacementContribution=0;
    if(pair<params.pairs&&q<plane)
    {
        uint geometry=pair/params.geometryRepeat,source=pair/params.sourceCameraRepeat,target=pair/params.targetCameraRepeat;
        uint sk=source*16,tk=target*16,k=target*9,i=pair*plane+q;
        float3 xyz=float3(cameraXYZ[(geometry*3)*plane+q],cameraXYZ[(geometry*3+1)*plane+q],cameraXYZ[(geometry*3+2)*plane+q]);
        // Apply translation unconditionally, including invalid depth's zero XYZ.
        float3 world=float3(dotCamera(sourcePose[sk],sourcePose[sk+1],sourcePose[sk+2],xyz)+sourcePose[sk+3],
                            dotCamera(sourcePose[sk+4],sourcePose[sk+5],sourcePose[sk+6],xyz)+sourcePose[sk+7],
                            dotCamera(sourcePose[sk+8],sourcePose[sk+9],sourcePose[sk+10],xyz)+sourcePose[sk+11]);
        float3 offset=world-float3(targetPose[tk+3],targetPose[tk+7],targetPose[tk+11]);
        float3 rel=float3(dotCamera(targetPose[tk],targetPose[tk+4],targetPose[tk+8],offset),
                          dotCamera(targetPose[tk+1],targetPose[tk+5],targetPose[tk+9],offset),
                          dotCamera(targetPose[tk+2],targetPose[tk+6],targetPose[tk+10],offset));
        float denominator=isnan(rel.z)?rel.z:max(rel.z,EPS);
        float2 xy=float2(ratio(mul(targetIntrinsics[k],rel.x),denominator)+targetIntrinsics[k+2],
                         ratio(mul(targetIntrinsics[k+4],rel.y),denominator)+targetIntrinsics[k+5]);
        visible=uint(geometryValid[geometry*plane+q]!=0&&rel.z>0&&xy.x>=0&&xy.x<=float(params.targetWidth-1)&&xy.y>=0&&xy.y<=float(params.targetHeight-1));
        float dx=xy.x-float(q%params.width),dy=xy.y-float(q/params.width);
        float displacement=sqrt(mul(dx,dx)+mul(dy,dy));
        projected_xyOut[(pair*2)*plane+q]=xy.x;projected_xyOut[(pair*2+1)*plane+q]=xy.y;
        projected_depthOut[i]=rel.z;visibleOut[i]=visible;displacementOut[i]=displacement;
        // Donor multiplication preserves 0*Inf/NaN in aggregate diagnostics.
        displacementContribution=mul(displacement,float(visible));
        if(params.useTargetDepth!=0)
        {
            float sample=nearestDepth(pair,xy);
            bool valid=visible!=0&&isfinite(sample)&&sample>0;
            depth_consistencyOut[i]=valid?abs(sample-rel.z):0.0;
            depth_consistency_validOut[i]=uint(valid);occlusion_validOut[i]=uint(valid);
            occludedOut[i]=uint(valid&&sample<rel.z-params.threshold);
        }
    }
    countTree[lane]=visible;parallaxTree[lane]=displacementContribution;
    reduceTrees(lane);
    if(lane==0&&pair<params.pairs){partialCount[group.x]=countTree[0];partialParallax[group.x]=parallaxTree[0];}
}
[shader("compute")]
[numthreads(256,1,1)]
void reduceProjection(uint3 group : SV_GroupID,uint3 local : SV_GroupThreadID)
{
    uint pair=group.x,lane=local.x;uint count=0;float sum=0;
    for(uint p=lane;p<params.groupsPerPair;p+=256)
    {
        uint i=pair*params.groupsPerPair+p;count+=partialCount[i];sum+=partialParallax[i];
    }
    countTree[lane]=count;parallaxTree[lane]=sum;reduceTrees(lane);
    if(lane==0&&pair<params.pairs)
    {
        overlapOut[pair]=ratio(float(countTree[0]),float(params.height*params.width));
        parallaxOut[pair]=ratio(parallaxTree[0],float(max(countTree[0],1u)));
    }
}
[shader("compute")]
[numthreads(256,1,1)]
void clearRaster(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.pairs*params.targetHeight*params.targetWidth)return;
    minDepthBits[i]=0x7f800000;maxConfidenceBits[i]=0;supportBits[i]=0;
}
uint rasterDestination(uint i)
{
    uint plane=params.height*params.width,pair=i/plane,q=i%plane;
    float u=projected_xyOut[(pair*2)*plane+q],v=projected_xyOut[(pair*2+1)*plane+q];
    uint x=uint(clamp(nearestEven(u),0,int(params.targetWidth-1)));
    uint y=uint(clamp(nearestEven(v),0,int(params.targetHeight-1)));
    return pair*params.targetHeight*params.targetWidth+y*params.targetWidth+x;
}
[shader("compute")]
[numthreads(256,1,1)]
void rasterMin(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.pairs*params.height*params.width||visibleOut[i]==0)return;
    uint destination=rasterDestination(i),ignored;
    // Visible z is strictly positive; IEEE positive-float bit ordering is uint ordering.
    InterlockedMin(minDepthBits[destination],asuint(projected_depthOut[i]),ignored);
}
[shader("compute")]
[numthreads(256,1,1)]
void rasterSupport(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.pairs*params.height*params.width||visibleOut[i]==0)return;
    uint destination=rasterDestination(i),ignored;
    float minimum=asfloat(minDepthBits[destination]),z=projected_depthOut[i];
    if(abs(z-minimum)<=mul(EPS,max(abs(minimum),1.0)))
    {
        float confidence=params.useConfidence!=0?sourceConfidence[i]:1.0;
        InterlockedMax(maxConfidenceBits[destination],confidence==0?0u:asuint(confidence),ignored);
        InterlockedAdd(supportBits[destination],1u,ignored);
    }
}
[shader("compute")]
[numthreads(256,1,1)]
void finalizeRaster(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.pairs*params.targetHeight*params.targetWidth)return;
    float z=asfloat(minDepthBits[i]);bool valid=isfinite(z);
    depthOut[i]=valid?z:0.0;validOut[i]=uint(valid);
    confidenceOut[i]=asfloat(maxConfidenceBits[i]);support_countOut[i]=supportBits[i];
}
[shader("compute")]
[numthreads(256,1,1)]
void sampleTemporal(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.height*params.width;if(i>=params.pairs*plane)return;
    uint pair=i/plane,q=i%plane;
    float2 xy=float2(projected_xyOut[(pair*2)*plane+q],projected_xyOut[(pair*2+1)*plane+q]);
    bool valid=visibleOut[i]!=0;float confidence=float(valid);
    if(params.useTargetDepth!=0)
    {
        bool consistent=depth_consistency_validOut[i]!=0&&occludedOut[i]==0;
        valid=valid&&consistent;
        confidence=mul(mul(confidence,exp(ratio(-depth_consistencyOut[i],params.threshold))),float(consistent));
    }
    if(params.useConfidence!=0)confidence=mul(confidence,bilinear(sourceConfidence,pair*plane,xy));
    aligned_validOut[i]=uint(valid);alignment_confidenceOut[i]=confidence;
    [unroll] for(uint c=0;c<3;c++)aligned_rgbOut[(pair*3+c)*plane+q]=bilinear(sourceRGB,(pair*3+c)*plane,xy);
}
[shader("compute")]
[numthreads(256,1,1)]
void packTemporal(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,count=params.batches*params.height*params.width;
    if(i>=(params.neighbors+1)*count)return;
    uint source=i/count,plane=params.height*params.width,q=i%count,batch=q/plane,pixel=q%plane;
    float depth=0,confidence=0;uint valid=0;
    if(source!=0)
    {
        uint prior=(batch*params.neighbors+source-1)*plane+pixel;
        depth=depthOut[prior];confidence=confidenceOut[prior];valid=validOut[prior];
    }
    packedDepth[i]=depth;packedConfidence[i]=confidence;packedValid[i]=valid;packedTrusted[i]=0;
}
[shader("compute")]
[numthreads(256,1,1)]
void finalizeReference(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.height*params.width,count=params.batches*plane;if(i>=count)return;
    uint batch=i/plane,q=i%plane;float total=0;float3 numerator=float3(0.0);
    for(uint n=0;n<params.neighbors;n++)
    {
        uint pair=batch*params.neighbors+n,j=pair*plane+q;
        float weight=mul(alignment_confidenceOut[j],rasterConfidence[j]);
        weight=mul(mul(weight,float(rasterValid[j]!=0)),float(aligned_validOut[j]!=0));
        total+=weight;
        [unroll] for(uint c=0;c<3;c++)numerator[c]+=mul(aligned_rgbOut[(pair*3+c)*plane+q],weight);
    }
    bool valid=fusedValid[i]!=0&&total>0;
    validOut[i]=uint(valid);depthOut[i]=valid?fusedDepth[i]:0.0;
    confidenceOut[i]=valid?mul(fusedConfidence[i],min(total,1.0)):0.0;
    [unroll] for(uint c=0;c<3;c++)rgbOut[(batch*3+c)*plane+q]=valid?ratio(numerator[c],max(total,EPS)):0.0;
}
