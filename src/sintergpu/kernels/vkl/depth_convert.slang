// SPDX-License-Identifier: Apache-2.0
// Numeric conversion profile; metric status and validation remain host policy.
struct DepthConvertParams { uint count; uint representation; float scale; float focalBaseline; };
ConstantBuffer<DepthConvertParams> params;
StructuredBuffer<float> values;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<float> inverseOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
[shader("compute")]
[numthreads(256, 1, 1)]
void depthConvert(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= params.count) return;
    float v = values[i];
    float d = params.representation == 0 ? v*params.scale :
        (params.representation == 1 ? 1.0/(v*params.scale) : params.focalBaseline/v);
    bool usable = valid[i] != 0 && isfinite(v) && v > 0 && isfinite(d) && d > 0;
    depthOut[i] = usable ? d : 0;
    inverseOut[i] = usable ? 1.0/d : 0;
    validOut[i] = uint(usable);
    confidenceOut[i] = usable ? confidence[i] : 0;
}
