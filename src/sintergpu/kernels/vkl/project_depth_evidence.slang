// SPDX-License-Identifier: Apache-2.0
// VKL 3384023a: contiguous float32 profile, explicit uint32 logical mask ABI.
// Caller owns buffers/device/command encoder and validates cameras/thresholds.
struct ProjectionParams
{
    uint batches; uint points; uint height; uint width;
    float supportThreshold; float conflictThreshold; float epsilon;
};
ConstantBuffer<ProjectionParams> params;
StructuredBuffer<float> xyz; // [B,N,3], scalar interleaved
StructuredBuffer<uint> visible;
StructuredBuffer<uint> provisional;
StructuredBuffer<uint> depthCandidate;
StructuredBuffer<float> worldToCamera; // [B,4,4] row major
StructuredBuffer<float> intrinsics; // [B,3,3] row major
StructuredBuffer<float> sensorDepth; // [B,1,H,W]
RWStructuredBuffer<uint> eligibleOut;
RWStructuredBuffer<uint> projectedOut;
RWStructuredBuffer<uint> sensorOut;
RWStructuredBuffer<uint> supportOut;
RWStructuredBuffer<uint> conflictOut;
RWStructuredBuffer<float> residualOut;

float roundEven(float value)
{
    float lower = floor(value);
    float fraction = value - lower;
    return lower + ((fraction > 0.5 || (fraction == 0.5 && fmod(abs(lower), 2.0) != 0.0)) ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void projectDepthEvidence(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= params.batches * params.points) return;
    uint batch = i / params.points;
    uint m = batch * 16;
    uint k = batch * 9;
    float3 p = float3(xyz[i*3], xyz[i*3+1], xyz[i*3+2]);
    bool candidate = depthCandidate[i] != 0;
    bool eligible = visible[i] != 0 && (provisional[i] != 0 || candidate) && all(isfinite(p));
    // Precise keeps separate float32 multiply/add operations as in the oracle.
    precise float x = p.x*worldToCamera[m] + p.y*worldToCamera[m+1] + p.z*worldToCamera[m+2] + worldToCamera[m+3];
    precise float y = p.x*worldToCamera[m+4] + p.y*worldToCamera[m+5] + p.z*worldToCamera[m+6] + worldToCamera[m+7];
    precise float z = p.x*worldToCamera[m+8] + p.y*worldToCamera[m+9] + p.z*worldToCamera[m+10] + worldToCamera[m+11];
    bool positive = isfinite(z) && z > params.epsilon;
    float safeZ = positive ? z : 1.0;
    precise float u = x / safeZ * intrinsics[k] + intrinsics[k+2];
    precise float v = y / safeZ * intrinsics[k+4] + intrinsics[k+5];
    // Float range test before integer conversion avoids undefined overflow for
    // finite-but-huge projections; safe for dimensions <= 2^24.
    float ru = roundEven(isfinite(u) ? u : 0.0);
    float rv = roundEven(isfinite(v) ? v : 0.0);
    bool projected = eligible && positive && isfinite(u) && isfinite(v)
        && ru >= 0.0 && ru < float(params.width) && rv >= 0.0 && rv < float(params.height);
    bool sensorValid = false;
    float lookup = 0.0;
    if (projected)
    {
        lookup = sensorDepth[batch*params.height*params.width + uint(rv)*params.width + uint(ru)];
        sensorValid = isfinite(lookup) && lookup > 0.0;
    }
    float residual = sensorValid ? abs(z-lookup) : 0.0;
    eligibleOut[i] = uint(eligible);
    projectedOut[i] = uint(projected);
    sensorOut[i] = uint(sensorValid);
    supportOut[i] = uint(sensorValid && residual < params.supportThreshold);
    conflictOut[i] = uint(sensorValid && candidate && z < lookup-params.conflictThreshold);
    residualOut[i] = residual;
}
