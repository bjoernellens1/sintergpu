// SPDX-License-Identifier: Apache-2.0
// Experimental VKL float32 sparse depth validation; see contract document.
struct Params { uint batches; uint points; uint height; uint width; uint minNeighbors; float maxRelativeDispersion; };
ConstantBuffer<Params> params;
StructuredBuffer<float> depth;
StructuredBuffer<int2> pixels;
StructuredBuffer<uint> pixelsValid;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> dispersionOut;
RWStructuredBuffer<float> edgeRiskOut;
RWStructuredBuffer<uint> neighborCountOut;
RWStructuredBuffer<float> confidenceOut;

[shader("compute")][numthreads(64,1,1)]
void sparseDepthValidate(uint3 tid : SV_DispatchThreadID) {
    uint i=tid.x;
    if (i>=params.batches*params.points) return;
    uint b=i/params.points;
    int2 p=pixels[i];
    bool inBounds=p.x>=0 && p.y>=0 && p.x<int(params.width) && p.y<int(params.height);
    float values[9];
    for (uint k=0;k<9;++k) values[k]=0.0;
    uint count=0;float center=0.0;
    if (inBounds) {
        uint area=params.height*params.width;
        center=depth[b*area+uint(p.y)*params.width+uint(p.x)];
        for (int dy=-1;dy<=1;++dy) {
            for (int dx=-1;dx<=1;++dx) {
                int x=p.x+dx,y=p.y+dy;
                if (x<0 || y<0 || x>=int(params.width) || y>=int(params.height)) continue;
                float z=depth[b*area+uint(y)*params.width+uint(x)];
                if (!isfinite(z) || z<=0.0) continue;
                uint at=count;
                while(at>0 && values[at-1]>z) {values[at]=values[at-1];--at;}
                values[at]=z; ++count;
            }
        }
    }
    float dispersion=0.0, edge=0.0;
    bool finiteMetrics=false;
    if (count>0) {
        float median=values[(count-1)/2];
        float deviations[9];
        for (uint k=0;k<9;++k) deviations[k]=0.0;
        for(uint k=0;k<count;++k) {
            float value=abs(values[k]-median);uint at=k;
            while(at>0 && deviations[at-1]>value) {deviations[at]=deviations[at-1];--at;}
            deviations[at]=value;
        }
        float denominator=max(median,1.1920928955078125e-7);
        dispersion=deviations[(count-1)/2]/denominator;
        edge=(values[count-1]-values[0])/denominator;
        finiteMetrics=isfinite(dispersion) && isfinite(edge);
    }
    bool usable=pixelsValid[i]!=0 && inBounds && isfinite(center) && center>0.0 &&
        count>=params.minNeighbors && finiteMetrics &&
        dispersion<=params.maxRelativeDispersion && edge<=params.maxRelativeDispersion;
    depthOut[i]=usable?center:0.0; validOut[i]=uint(usable);
    // torch.nan_to_num converts positive infinity to largest finite float.
    dispersionOut[i]=isinf(dispersion)?3.4028234663852886e38:dispersion;
    edgeRiskOut[i]=isinf(edge)?3.4028234663852886e38:edge;
    neighborCountOut[i]=count;
    confidenceOut[i]=usable ? (float(count)/9.0)*(1.0-clamp(max(dispersion,edge),0.0,1.0)) : 0.0;
}
