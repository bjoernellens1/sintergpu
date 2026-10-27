// Workgroup sums retain all tail lanes until the final barrier.
struct ReductionParams { uint count; };
StructuredBuffer<float> residual;
StructuredBuffer<float> jacobian;
StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> outputTerms;
ConstantBuffer<ReductionParams> params;
groupshared float terms[43][64];

float jac(uint i, uint c) {
    return jacobian[c < 4 ? i * 4 + c : 4 * params.count + i * 2 + c - 4];
}
void reduceShared(uint lane, uint group) {
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride)
            for (uint t = 0; t < 43; ++t)
                terms[t][lane] += terms[t][lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        for (uint t = 0; t < 43; ++t)
            outputTerms[group * 43 + t] = terms[t][0];
}
void buildTerms(uint i, uint lane, bool useWave) {
    for (uint t = 0; t < 43; ++t) {
        float value = 0.0;
        if (i < params.count) {
            if (t < 36) value = jac(i, t / 6) * jac(i, t % 6);
            else if (t < 42) value = jac(i, t - 36) * residual[i];
            else value = residual[i] * residual[i];
        }
        if (useWave) {
            float sum = WaveActiveSum(value);
            value = WaveIsFirstLane() ? sum : 0.0;
        }
        terms[t][lane] = value;
    }
}
[shader("compute")]
[numthreads(64,1,1)]
void reduceResidual(uint3 tid : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID,
                    uint3 group : SV_GroupID) {
    buildTerms(tid.x, lane.x, false);
    reduceShared(lane.x, group.x);
}
[shader("compute")]
[numthreads(64,1,1)]
void reduceResidualSubgroup(uint3 tid : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID,
                           uint3 group : SV_GroupID) {
    buildTerms(tid.x, lane.x, true);
    reduceShared(lane.x, group.x);
}
[shader("compute")]
[numthreads(64,1,1)]
void reducePartials(uint3 tid : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID,
                    uint3 group : SV_GroupID) {
    for (uint t = 0; t < 43; ++t)
        terms[t][lane.x] = tid.x < params.count ? inputTerms[tid.x * 43 + t] : 0.0;
    reduceShared(lane.x, group.x);
}

struct SolveParams { float damping; };
ConstantBuffer<SolveParams> solveParams;
RWStructuredBuffer<float> delta;
RWStructuredBuffer<uint> status;
[shader("compute")]
[numthreads(1,1,1)]
void solveSystem() {
    float L[36];
    float y[6];
    float x[6];
    for (uint i=0; i<6; ++i) { y[i]=0.0; x[i]=0.0; }
    bool valid = true;
    for (uint i=0; i<36; ++i) L[i]=0.0;
    for (uint i=0; i<6; ++i) delta[i]=0.0;
    for (uint i=0; i<6; ++i) {
        for (uint j=0; j<=i; ++j) {
            float v = inputTerms[i*6+j] + (i==j ? solveParams.damping : 0.0);
            for (uint k=0; k<j; ++k) v -= L[i*6+k]*L[j*6+k];
            if (i==j) {
                if (!(v>0.0) || !isfinite(v)) valid=false;
                L[i*6+j]=valid ? sqrt(v) : 1.0;
            } else {
                L[i*6+j]=v/L[j*6+j];
                if (!isfinite(L[i*6+j])) valid=false;
            }
        }
    }
    for (uint i=0; i<6; ++i) {
        float v=-inputTerms[36+i];
        for (uint k=0; k<i; ++k) v-=L[i*6+k]*y[k];
        y[i]=v/L[i*6+i];
        if (!isfinite(y[i])) valid=false;
    }
    for (int i=5; i>=0; --i) {
        float v=y[i];
        for (uint k=uint(i)+1; k<6; ++k) v-=L[k*6+uint(i)]*x[k];
        x[i]=v/L[uint(i)*6+uint(i)];
        if (!isfinite(x[i])) valid=false;
    }
    status[0]=valid ? 0u : 1u;
    for (uint i=0; i<6; ++i) delta[i]=valid ? x[i] : 0.0;
}
