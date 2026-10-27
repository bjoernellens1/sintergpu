// SPDX-License-Identifier: Apache-2.0
// One validity-aware ceil-half level. Donor masked-INF maximum is preserved.
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
    uint count=0;
    float minimum=asfloat(0x7f800000), maximum=0, sum=0, sumConfidence=0, selectedConfidence=0;
    for(uint dy=0;dy<2;++dy) for(uint dx=0;dx<2;++dx)
    {
        if (y+dy>=params.height || x+dx>=params.width) continue;
        uint j=b*params.height*params.width+(y+dy)*params.width+x+dx;
        float d=depth[j];
        if (valid[j]==0 || !isfinite(d) || d<=0) continue;
        float c=confidence[j];
        if (d<minimum) { minimum=d; selectedConfidence=c; }
        maximum=max(maximum,d); sum+=d; sumConfidence+=c; count++;
    }
    // Missing/invalid samples were masked to INF before donor maximum reduction.
    bool mean=count==4 && maximum-minimum<=params.epsilon*32.0;
    validOut[i]=uint(count>0);
    depthOut[i]=count==0 ? 0 : (mean ? sum/float(count) : minimum);
    confidenceOut[i]=count==0 ? 0 : (mean ? sumConfidence/float(count) : selectedConfidence);
}
