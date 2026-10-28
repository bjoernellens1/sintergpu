// SPDX-License-Identifier: Apache-2.0
// VKL eager float32 profile. Caller owns every buffer and dispatch dependency.
#include "../common/brown_conrady.slang"
#include "../common/float32_divide.slang"
#include "../common/float32_multiply.slang"
struct GridParams { uint batches; uint sourceHeight; uint sourceWidth; uint targetHeight; uint targetWidth; uint distortion; uint debug; };
struct SampleParams { uint batches; uint channels; uint sourceHeight; uint sourceWidth; uint targetHeight; uint targetWidth; uint sampling; };
struct RaysParams { uint batches; uint depthHeight; uint depthWidth; uint depthDistortion; };
struct RegistrationParams { uint batches; uint depthHeight; uint depthWidth; uint targetHeight; uint targetWidth; uint points; uint rgbDistortion; float minDepth; uint hasMaxDepth; float maxDepth; uint debug; };
ConstantBuffer<GridParams> gridParams;
ConstantBuffer<SampleParams> sampleParams;
ConstantBuffer<RaysParams> raysParams;
ConstantBuffer<RegistrationParams> registrationParams;
StructuredBuffer<float> sourceIntrinsics, targetIntrinsics, sourceDistortion;
StructuredBuffer<float> image, sourceUV;
StructuredBuffer<uint> valid;
StructuredBuffer<float> depthIntrinsics, depthDistortion;
StructuredBuffer<float> depth, depthRays, rgbIntrinsics, transformRgbDepth, rgbDistortion;
RWStructuredBuffer<float> sourceUVOut, targetRayOut, displacementOut, imageOut, depthRaysOut;
RWStructuredBuffer<float> projectedZOut, depthOut;
RWStructuredBuffer<uint> validOut, sourceValidOut, destinationIndexOut;
RWStructuredBuffer<uint> minDepthBits, winnerIndex, collisionCount;
// Each int64 host diagnostic is stored as low/high uint32 words. Host readback
// requires little-endian two's-complement int64 and an 8-byte buffer stride.
RWStructuredBuffer<uint2> winnerSourceIndexOut, collisionCountOut;
static const float EPS=1.1920928955078125e-7;
float ratio(float a,float b) { return asfloat(sintergpuF32DivideBits(asuint(a),asuint(b))); }
float mul(float a,float b) { return sintergpuF32MulNoContract(a,b); }
float roundedAdd(float a,float b)
{
    __target_switch
    {
    case spirv:
        return spirv_asm { %rounded:$$float = OpFAdd $a $b; OpDecorate %rounded NoContraction; result:$$float = OpCopyObject %rounded; };
    default:
        precise float rounded=a+b; return rounded;
    }
}
float normalizedRoundtrip(float pixel,uint size)
{
    float normalized=roundedAdd(ratio(mul(2.0,pixel),float(max(size-1,1u))),-1.0);
    return ratio(mul(roundedAdd(normalized,1.0),float(size-1)),2.0);
}
int nearestEven(float value)
{
    float lower=floor(value), fraction=value-lower; int result=int(lower);
    if(fraction>.5||(fraction==.5&&(result&1)!=0))result++;
    return result;
}
[shader("compute")][numthreads(256,1,1)]
void rectifyGrid(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x, plane=gridParams.targetHeight*gridParams.targetWidth;
    if(i>=gridParams.batches*plane)return;
    uint b=i/plane,q=i%plane,x=q%gridParams.targetWidth,y=q/gridParams.targetWidth,k=b*9;
    float2 ray=float2(ratio(float(x)-targetIntrinsics[k+2],targetIntrinsics[k]),
                      ratio(float(y)-targetIntrinsics[k+5],targetIntrinsics[k+4]));
    float2 distorted=gridParams.distortion!=0?brownForward(ray,sourceDistortion,b*5):ray;
    float2 uv=float2(mul(distorted.x,sourceIntrinsics[k])+sourceIntrinsics[k+2],
                     mul(distorted.y,sourceIntrinsics[k+4])+sourceIntrinsics[k+5]);
    float tolerance=16.0*EPS*float(max(gridParams.sourceHeight,gridParams.sourceWidth));
    bool ok=isfinite(uv.x)&&isfinite(uv.y)&&uv.x>=-tolerance&&uv.x<=float(gridParams.sourceWidth-1)+tolerance&&uv.y>=-tolerance&&uv.y<=float(gridParams.sourceHeight-1)+tolerance;
    sourceUVOut[2*i]=uv.x;sourceUVOut[2*i+1]=uv.y;validOut[i]=uint(ok);
    if(gridParams.debug!=0)
    {
        targetRayOut[3*i]=ray.x;targetRayOut[3*i+1]=ray.y;targetRayOut[3*i+2]=1.0;
        float dx=uv.x-float(x),dy=uv.y-float(y);displacementOut[i]=sqrt(mul(dx,dx)+mul(dy,dy));
    }
}
float tap(uint base,int x,int y,float weight,float sum)
{
    if(x<0||y<0||x>=int(sampleParams.sourceWidth)||y>=int(sampleParams.sourceHeight))return sum;
    return fma(image[base+uint(y)*sampleParams.sourceWidth+uint(x)],weight,sum);
}
[shader("compute")][numthreads(256,1,1)]
void rectifyFromGrid(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x, plane=sampleParams.targetHeight*sampleParams.targetWidth;
    if(i>=sampleParams.batches*sampleParams.channels*plane)return;
    uint b=i/(sampleParams.channels*plane), channel=(i/plane)%sampleParams.channels,q=i%plane, grid=b*plane+q;
    if(valid[grid]==0){imageOut[i]=0.0;return;}
    float rawU=sourceUV[2*grid],rawV=sourceUV[2*grid+1];
    float u=clamp(isfinite(rawU)?rawU:0.0,0.0,float(sampleParams.sourceWidth-1));
    float v=clamp(isfinite(rawV)?rawV:0.0,0.0,float(sampleParams.sourceHeight-1));
    float x=normalizedRoundtrip(u,sampleParams.sourceWidth),y=normalizedRoundtrip(v,sampleParams.sourceHeight);
    uint base=(b*sampleParams.channels+channel)*sampleParams.sourceHeight*sampleParams.sourceWidth;
    if(sampleParams.sampling==0){imageOut[i]=image[base+uint(nearestEven(y))*sampleParams.sourceWidth+uint(nearestEven(x))];return;}
    int x0=int(floor(x)),y0=int(floor(y));float dx=x-float(x0),dy=y-float(y0);
    float sum=0.0;
    sum=tap(base,x0,y0,mul(1.0-dx,1.0-dy),sum);
    sum=tap(base,x0+1,y0,mul(dx,1.0-dy),sum);
    sum=tap(base,x0,y0+1,mul(1.0-dx,dy),sum);
    sum=tap(base,x0+1,y0+1,mul(dx,dy),sum);
    imageOut[i]=sum;
}
[shader("compute")][numthreads(256,1,1)]
void depthCameraRays(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=raysParams.depthHeight*raysParams.depthWidth;
    if(i>=raysParams.batches*plane)return;
    uint b=i/plane,q=i%plane,k=b*9;
    float2 xy=float2(ratio(float(q%raysParams.depthWidth)-depthIntrinsics[k+2],depthIntrinsics[k]),
                     ratio(float(q/raysParams.depthWidth)-depthIntrinsics[k+5],depthIntrinsics[k+4]));
    if(raysParams.depthDistortion!=0)xy=brownInverse(xy,depthDistortion,b*5);
    depthRaysOut[2*i]=xy.x;depthRaysOut[2*i+1]=xy.y;
}
[shader("compute")][numthreads(256,1,1)]
void clearRegistration(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,count=registrationParams.batches*registrationParams.targetHeight*registrationParams.targetWidth;
    if(i>=count)return;
    minDepthBits[i]=0x7f800000u;winnerIndex[i]=registrationParams.points;collisionCount[i]=0;
}
[shader("compute")][numthreads(256,1,1)]
void projectRegistration(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=registrationParams.batches*registrationParams.points)return;
    uint b=i/registrationParams.points,k=b*9,t=b*16;
    float d=depth[i];bool dv=isfinite(d)&&d>=registrationParams.minDepth&&(registrationParams.hasMaxDepth==0||d<=registrationParams.maxDepth);
    float safe=dv?d:0.0;
    float3 p=float3(mul(depthRays[2*i],safe),mul(depthRays[2*i+1],safe),safe);
    float3 xyz=float3(((mul(p.x,transformRgbDepth[t])+mul(p.y,transformRgbDepth[t+1]))+mul(p.z,transformRgbDepth[t+2]))+transformRgbDepth[t+3],
                      ((mul(p.x,transformRgbDepth[t+4])+mul(p.y,transformRgbDepth[t+5]))+mul(p.z,transformRgbDepth[t+6]))+transformRgbDepth[t+7],
                      ((mul(p.x,transformRgbDepth[t+8])+mul(p.y,transformRgbDepth[t+9]))+mul(p.z,transformRgbDepth[t+10]))+transformRgbDepth[t+11]);
    bool positive=isfinite(xyz.z)&&xyz.z>EPS;float sz=positive?xyz.z:1.0;
    float2 xy=float2(ratio(xyz.x,sz),ratio(xyz.y,sz));
    if(registrationParams.rgbDistortion!=0)xy=brownForward(xy,rgbDistortion,b*5);
    float2 uv=float2(mul(xy.x,rgbIntrinsics[k])+rgbIntrinsics[k+2],mul(xy.y,rgbIntrinsics[k+4])+rgbIntrinsics[k+5]);
    sourceUVOut[2*i]=uv.x;sourceUVOut[2*i+1]=uv.y;projectedZOut[i]=xyz.z;
    // Bound in float space before integer conversion, including huge finite values.
    bool candidate=dv&&positive&&isfinite(uv.x)&&isfinite(uv.y)&&uv.x>=-.5&&uv.y>=-.5&&uv.x<=float(registrationParams.targetWidth)-.5&&uv.y<=float(registrationParams.targetHeight)-.5;
    int x=0,y=0;if(candidate){x=nearestEven(uv.x);y=nearestEven(uv.y);}
    bool ok=candidate&&x>=0&&y>=0&&uint(x)<registrationParams.targetWidth&&uint(y)<registrationParams.targetHeight;
    sourceValidOut[i]=uint(ok);destinationIndexOut[i]=ok?b*registrationParams.targetHeight*registrationParams.targetWidth+uint(y)*registrationParams.targetWidth+uint(x):0u;
}
[shader("compute")][numthreads(256,1,1)]
void registrationMin(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=registrationParams.batches*registrationParams.points||sourceValidOut[i]==0)return;
    uint ignored;InterlockedMin(minDepthBits[destinationIndexOut[i]],asuint(projectedZOut[i]),ignored);
}
[shader("compute")][numthreads(256,1,1)]
void registrationWinnerCount(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=registrationParams.batches*registrationParams.points||sourceValidOut[i]==0)return;
    uint destination=destinationIndexOut[i],ignored;
    InterlockedAdd(collisionCount[destination],1u,ignored);
    if(asuint(projectedZOut[i])==minDepthBits[destination])InterlockedMin(winnerIndex[destination],i%registrationParams.points,ignored);
}
[shader("compute")][numthreads(256,1,1)]
void finalizeRegistration(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=registrationParams.batches*registrationParams.targetHeight*registrationParams.targetWidth)return;
    float z=asfloat(minDepthBits[i]);bool ok=isfinite(z);depthOut[i]=ok?z:0.0;validOut[i]=uint(ok);
}
[shader("compute")][numthreads(256,1,1)]
void finalizeRegistrationDebug(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=registrationParams.batches*registrationParams.targetHeight*registrationParams.targetWidth)return;
    float z=asfloat(minDepthBits[i]);bool ok=isfinite(z);depthOut[i]=ok?z:0.0;validOut[i]=uint(ok);
    winnerSourceIndexOut[i]=ok?uint2(winnerIndex[i],0u):uint2(0xffffffffu,0xffffffffu);
    collisionCountOut[i]=uint2(collisionCount[i],0u);
}
