// SPDX-License-Identifier: Apache-2.0
// Experimental candidate; original VKL entry ABI retained.
#include "depth_pyramid_core.slang"
struct DepthPyramidParams { uint batches; uint height; uint width; float epsilon; };
ConstantBuffer<DepthPyramidParams> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
[shader("compute")]
[numthreads(256, 1, 1)]
void depthPyramidLevel(uint3 tid : SV_DispatchThreadID)
{
    uint ow=(params.width+1)/2, oh=(params.height+1)/2;
    uint i=tid.x;
    if (i >= params.batches*ow*oh) return;
    uint b=i/(ow*oh), x=(i%ow)*2, y=((i/ow)%oh)*2;
    float d[4], c[4]; uint v[4];
    for(uint dy=0;dy<2;++dy) for(uint dx=0;dx<2;++dx)
    {
        uint k=dy*2+dx;
        d[k]=0; c[k]=0; v[k]=0;
        if (y+dy>=params.height || x+dx>=params.width) continue;
        uint j=b*params.height*params.width+(y+dy)*params.width+x+dx;
        d[k]=depth[j]; v[k]=valid[j];
        if (v[k]!=0 && isfinite(d[k]) && d[k]>0) c[k]=confidence[j];
    }
    DepthPyramidVklResult result=depthPyramidVklBlock(d,v,c,params.epsilon);
    validOut[i]=result.valid;
    depthOut[i]=result.depth;
    confidenceOut[i]=result.confidence;
}
