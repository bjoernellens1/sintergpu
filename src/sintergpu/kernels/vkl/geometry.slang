// SPDX-License-Identifier: Apache-2.0
// VKL float32 forward geometry. B3HW planar storage; row-major cameras.
#include "../common/float32_divide.slang"
#include "../common/float32_multiply.slang"
struct GeometryParams { uint batches; uint height; uint width; uint useWorld; uint useConfidence; };
ConstantBuffer<GeometryParams> params;
StructuredBuffer<float> depth;
StructuredBuffer<float> confidence;
StructuredBuffer<float> intrinsics;
StructuredBuffer<float> pose;
StructuredBuffer<float> xyz;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> rays_cameraOut;
RWStructuredBuffer<float> xyz_cameraOut;
RWStructuredBuffer<float> normalsOut;
RWStructuredBuffer<float> geometry_confidenceOut;
RWStructuredBuffer<float> xyz_worldOut;

float3 cameraRay(uint b,uint x,uint y)
{
    uint k=b*9;
    return float3(asfloat(sintergpuF32DivideBits(asuint(float(x)-intrinsics[k+2]),asuint(intrinsics[k]))),
                  asfloat(sintergpuF32DivideBits(asuint(float(y)-intrinsics[k+5]),asuint(intrinsics[k+4]))),1.0);
}
float3 cameraPoint(uint b,uint x,uint y)
{
    float z=depth[b*params.height*params.width+y*params.width+x];
    return sintergpuF32ScaleNoContract(cameraRay(b,x,y),isfinite(z)&&z>0 ? z : 0.0);
}
float3 transformPoint(uint b,float3 p)
{
    uint k=b*16;
    return float3(pose[k]*p.x+pose[k+1]*p.y+pose[k+2]*p.z+pose[k+3],
                  pose[k+4]*p.x+pose[k+5]*p.y+pose[k+6]*p.z+pose[k+7],
                  pose[k+8]*p.x+pose[k+9]*p.y+pose[k+10]*p.z+pose[k+11]);
}
[shader("compute")]
[numthreads(256,1,1)]
void depthGeometry(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.height*params.width;
    if(i>=params.batches*plane)return;
    uint b=i/plane,q=i%plane,x=q%params.width,y=q/params.width;
    float z=depth[i];bool valid=isfinite(z)&&z>0;
    float3 ray=cameraRay(b,x,y),point=sintergpuF32ScaleNoContract(ray,valid?z:0.0),normal=0.0;
    bool stencil=false;
    if(x+1<params.width && y+1<params.height)
    {
        float zr=depth[i+1],zd=depth[i+params.width];
        stencil=valid&&isfinite(zr)&&zr>0&&isfinite(zd)&&zd>0
            &&abs(zr-z)<=abs(z)*.25&&abs(zd-z)<=abs(z)*.25;
        if(stencil)
        {
            float3 dx=cameraPoint(b,x+1,y)-point,dy=cameraPoint(b,x,y+1)-point;
            normal=cross(dx,dy);
            normal=normal/max(sqrt(dot(normal,normal)),1e-8);
            if(dot(normal,ray)>0)normal=-normal;
        }
    }
    float3 world=params.useWorld!=0&&valid ? transformPoint(b,point):float3(0.0);
    validOut[i]=uint(valid);
    // A compiler may fold bool-to-float times confidence into a select, losing
    // IEEE 0*NaN/Inf and signed zero. Preserve the donor multiplication by bits.
    float geometryConfidence=float(stencil);
    if(params.useConfidence!=0)
    {
        float c=confidence[i];
        geometryConfidence=stencil?c:(isfinite(c)?asfloat(asuint(c)&0x80000000):asfloat(0x7fc00000));
    }
    geometry_confidenceOut[i]=geometryConfidence;
    [unroll] for(uint c=0;c<3;c++)
    {
        uint j=(b*3+c)*plane+q;
        rays_cameraOut[j]=ray[c];xyz_cameraOut[j]=point[c];
        normalsOut[j]=normal[c];
        if(params.useWorld!=0)xyz_worldOut[j]=world[c];
    }
}
[shader("compute")]
[numthreads(256,1,1)]
void worldTransform(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.height*params.width;
    if(i>=params.batches*plane)return;
    uint b=i/plane,q=i%plane;
    float3 p=float3(xyz[(b*3)*plane+q],xyz[(b*3+1)*plane+q],xyz[(b*3+2)*plane+q]);
    // Standalone transform always applies translation; no implicit validity mask.
    float3 world=transformPoint(b,p);
    [unroll] for(uint c=0;c<3;c++)xyz_worldOut[(b*3+c)*plane+q]=world[c];
}
