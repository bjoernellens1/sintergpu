// SPDX-License-Identifier: Apache-2.0
// Candidate G21: Caspar/SLAM left-looking damped Cholesky, original loop order.
// See provenance/candidate-families.json. No VKL condition/eigenvalue policy.
#ifndef SINTERGPU_CANDIDATE_SMALL_CHOLESKY
#define SINTERGPU_CANDIDATE_SMALL_CHOLESKY
// Capacity specializes storage stride; dimension may be compile-time or runtime.
// Terms are row-major H followed by b. Solves (H + damping I)x = -b.
// Caller supplies input storage/offset and initial observation-policy validity.
// Output is exactly zero on failure. No allocation, dispatch or synchronization.
public bool sinterSmallCholesky<let Capacity : int>(StructuredBuffer<float> input,
    uint offset, uint dimension, float damping, bool initialValid,
    out float x[Capacity]) {
    float L[Capacity * Capacity], y[Capacity];
    for (uint i = 0; i < Capacity; ++i) { y[i] = 0.0; x[i] = 0.0; }
    if (dimension == 0 || dimension > Capacity) return false;
    for (uint i = 0; i < Capacity * Capacity; ++i) L[i] = 0.0;
    bool valid = initialValid;
    for (uint i = 0; i < dimension; ++i) {
        for (uint j = 0; j <= i; ++j) {
            float value = input[offset + i * dimension + j] + (i == j ? damping : 0.0);
            for (uint k = 0; k < j; ++k) value -= L[i * Capacity + k] * L[j * Capacity + k];
            if (i == j) {
                if (!(value > 0.0) || !isfinite(value)) valid = false;
                L[i * Capacity + j] = valid ? sqrt(value) : 1.0;
            } else {
                L[i * Capacity + j] = value / L[j * Capacity + j];
                if (!isfinite(L[i * Capacity + j])) valid = false;
            }
        }
    }
    for (uint i = 0; i < dimension; ++i) {
        float value = -input[offset + dimension * dimension + i];
        for (uint k = 0; k < i; ++k) value -= L[i * Capacity + k] * y[k];
        y[i] = value / L[i * Capacity + i];
        if (!isfinite(y[i])) valid = false;
    }
    for (int i = int(dimension) - 1; i >= 0; --i) {
        float value = y[i];
        for (uint k = uint(i) + 1; k < dimension; ++k) value -= L[k * Capacity + uint(i)] * x[k];
        x[i] = value / L[uint(i) * Capacity + uint(i)];
        if (!isfinite(x[i])) valid = false;
    }
    if (!valid) for (uint i = 0; i < dimension; ++i) x[i] = 0.0;
    return valid;
}
#endif

StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> delta;
RWStructuredBuffer<uint> status;
struct Config {uint terms;uint activeFrames;uint dofs;float damping;};ConstantBuffer<Config> cfg;

[shader("compute")][numthreads(1,1,1)]
void main() {
 float x[18];bool valid=sinterSmallCholesky<18>(inputTerms,0,cfg.dofs,cfg.damping,inputTerms[cfg.terms-1]>=float(6*(cfg.activeFrames-1)),x);
 status[0]=valid?0u:1u;for(uint i=0;i<cfg.dofs;++i)delta[i]=x[i];
}
