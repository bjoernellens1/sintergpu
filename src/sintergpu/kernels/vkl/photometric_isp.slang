// SPDX-License-Identifier: Apache-2.0
// VKL forward-only RGB profile. No gradients or image-wide evidence reductions.
struct PhotometricParams
{
    uint batches; uint height; uint width;
    uint useExposure; uint useGain; uint useVignette; uint useColor; uint useGamma;
    float epsilon;
};
ConstantBuffer<PhotometricParams> params;
StructuredBuffer<float> rgbLinear; // [B,3,H,W] planar
StructuredBuffer<float> exposureEV; // [B]
StructuredBuffer<float> rgbGain; // [B,3]
StructuredBuffer<float> vignette; // [B,3,3] attenuation coefficients
StructuredBuffer<float> colorMatrix; // [B,3,3], row major
StructuredBuffer<float> responseGamma; // [B]
RWStructuredBuffer<float> rgbObserved; // [B,3,H,W]

[shader("compute")]
[numthreads(256, 1, 1)]
void photometricISP(uint3 tid : SV_DispatchThreadID)
{
    uint pixels = params.height * params.width;
    uint i = tid.x;
    if (i >= params.batches*pixels) return;
    uint b = i / pixels, pixel = i % pixels;
    uint y = pixel / params.width, x = pixel % params.width;
    // linspace(-1,1,1) returns -1, matching the frozen reference.
    float nx = params.width > 1 ? -1.0 + 2.0*float(x)/float(params.width-1) : -1.0;
    float ny = params.height > 1 ? -1.0 + 2.0*float(y)/float(params.height-1) : -1.0;
    float r = nx*nx + ny*ny;
    float3 raw;
    float exposure = params.useExposure != 0 ? exp2(exposureEV[b]) : 1.0;
    for (uint c=0; c<3; ++c)
    {
        float value = rgbLinear[(b*3+c)*pixels+pixel] * exposure;
        if (params.useGain != 0) value *= rgbGain[b*3+c];
        if (params.useVignette != 0)
        {
            uint k = b*9+c*3;
            value *= exp(-(vignette[k]*r + vignette[k+1]*r*r + vignette[k+2]*r*r*r));
        }
        raw[c] = value;
    }
    float3 colored = raw;
    if (params.useColor != 0)
        for (uint c=0; c<3; ++c)
            colored[c] = colorMatrix[b*9+c*3]*raw.x + colorMatrix[b*9+c*3+1]*raw.y + colorMatrix[b*9+c*3+2]*raw.z;
    for (uint c=0; c<3; ++c)
    {
        float value = max(colored[c], 0.0);
        if (params.useGamma != 0)
        {
            float gamma = responseGamma[b];
            value = pow(value+params.epsilon, gamma) - pow(params.epsilon, gamma);
        }
        rgbObserved[(b*3+c)*pixels+pixel] = clamp(value, 0.0, 1.0);
    }
}
