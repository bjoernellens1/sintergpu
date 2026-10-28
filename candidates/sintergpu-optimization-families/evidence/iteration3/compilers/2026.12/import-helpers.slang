// SPDX-License-Identifier: Apache-2.0
// Compile with -I ../kernels. The host owns descriptors, buffers and completion.
import optimization.point_plane;
import optimization.small_cholesky;
import optimization.normal_terms;
StructuredBuffer<float> reducedTerms;
RWStructuredBuffer<float> solution;
RWStructuredBuffer<uint> status;
[shader("compute")][numthreads(1,1,1)]
void main() {
    float x[6];
    bool valid=sinterSmallCholesky<6>(reducedTerms,0,6,0.01,true,x);
    for(uint i=0;i<6;++i)solution[i]=x[i];
    status[0]=valid?0:1;
}
