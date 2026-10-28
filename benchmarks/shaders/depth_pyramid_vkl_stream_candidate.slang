// SPDX-License-Identifier: Apache-2.0
// Experimental per-sample reduction for VKL 3384023a311651ea4260cf12ddab1567972f89f3.
// Kept separate until numerical and performance qualification.
struct DepthPyramidParams { uint batches; uint height; uint width; float epsilon; };
ConstantBuffer<DepthPyramidParams> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;

struct DepthPyramidVklState {
    float minimum, maximum, sum, sumConfidence, selectedConfidence;
    uint count;
};

// Called only for a finite positive sample with a nonzero validity mask.
void depthPyramidVklAccumulate(inout DepthPyramidVklState state, float d, float c)
{
    if (d<state.minimum) { state.minimum=d; state.selectedConfidence=c; }
    state.maximum=max(state.maximum,d);
    state.sum+=d;
    state.sumConfidence+=c;
    state.count++;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void depthPyramidLevel(uint3 tid : SV_DispatchThreadID)
{
    uint ow=(params.width+1)/2, oh=(params.height+1)/2;
    uint i=tid.x;
    if (i >= params.batches*ow*oh) return;
    uint b=i/(ow*oh), x=(i%ow)*2, y=((i/ow)%oh)*2;
    DepthPyramidVklState state;
    state.count=0; state.minimum=asfloat(0x7f800000); state.maximum=0;
    state.sum=0; state.sumConfidence=0; state.selectedConfidence=0;
    for(uint dy=0;dy<2;++dy) for(uint dx=0;dx<2;++dx)
    {
        if (y+dy>=params.height || x+dx>=params.width) continue;
        uint j=b*params.height*params.width+(y+dy)*params.width+x+dx;
        float d=depth[j];
        if (valid[j]==0 || !isfinite(d) || d<=0) continue;
        float c=confidence[j];
        depthPyramidVklAccumulate(state,d,c);
    }
    bool mean=state.count==4 && state.maximum-state.minimum<=params.epsilon*32.0;
    validOut[i]=uint(state.count>0);
    depthOut[i]=state.count==0 ? 0 : (mean ? state.sum/float(state.count) : state.minimum);
    confidenceOut[i]=state.count==0 ? 0 : (mean ? state.sumConfidence/float(state.count) : state.selectedConfidence);
}
