// SPDX-License-Identifier: Apache-2.0
// Experimental VKL float32 forward profile; see docs/dense-depth-regularize-experimental.md.
struct Params {
    uint batches; uint height; uint width; uint channels;
    uint minSupport; float maxRgbDifference; float maxRelativeDeviation;
};
ConstantBuffer<Params> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
StructuredBuffer<float> rgb;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
RWStructuredBuffer<uint> directValidOut;
RWStructuredBuffer<uint> supportOut;

float gray(uint batch, uint offset, uint area) {
    uint i = batch * params.channels * area + offset;
    if (params.channels == 1) return rgb[i];
    return (rgb[i] * 0.299 + rgb[i+area] * 0.587) + rgb[i+2*area] * 0.114;
}

[shader("compute")][numthreads(64,1,1)]
void regularizeDenseDepth(uint3 tid : SV_DispatchThreadID) {
    uint area=params.height*params.width;
    uint i=tid.x;
    if (i>=params.batches*area) return;
    uint batch=i/area, local=i%area, y=local/params.width, x=local%params.width;
    float centerGray=gray(batch,local,area);
    float sorted[9];
    uint n=0; float confidenceSum=0.0;
    for (int dy=-1;dy<=1;++dy) {
        for (int dx=-1;dx<=1;++dx) {
            int xx=int(x)+dx, yy=int(y)+dy;
            if (xx<0 || yy<0 || xx>=int(params.width) || yy>=int(params.height)) continue;
            uint off=uint(yy)*params.width+uint(xx), j=batch*area+off;
            float z=depth[j];
            if (valid[j]==0 || !isfinite(z) || z<=0.0 || !(abs(gray(batch,off,area)-centerGray)<=params.maxRgbDifference)) continue;
            uint at=n;
            while (at>0 && sorted[at-1]>z) { sorted[at]=sorted[at-1]; --at; }
            sorted[at]=z; ++n;
            confidenceSum+=confidence[j];
        }
    }
    float median=depth[i];
    if (n>0) median=sorted[(n-1)/2];
    float z=depth[i];
    bool direct=valid[i]!=0 && isfinite(z) && z>0.0 && n>=params.minSupport &&
                abs(z-median)<=params.maxRelativeDeviation*max(z,1.1920928955078125e-7);
    bool outputValid=n>=params.minSupport && isfinite(median) && median>0.0;
    depthOut[i]=outputValid ? median : 0.0;
    validOut[i]=uint(outputValid);
    confidenceOut[i]=outputValid ? (confidenceSum / float(max(n,1))) * (float(n)/9.0) : 0.0;
    directValidOut[i]=uint(direct);
    supportOut[i]=n;
}
