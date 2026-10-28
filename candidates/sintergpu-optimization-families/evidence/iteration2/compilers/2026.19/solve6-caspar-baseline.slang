
StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> delta;
RWStructuredBuffer<uint> status;
struct SolveParams {float damping;};ConstantBuffer<SolveParams> solveParams;
[shader("compute")][numthreads(1,1,1)]
void main() {
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