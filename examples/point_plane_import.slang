// SPDX-License-Identifier: Apache-2.0
// Import example, not the ABI-qualified generated adapter. Host owns all resources.
import optimization.point_plane;
StructuredBuffer<float4> points;
StructuredBuffer<float4> targets;
StructuredBuffer<float4> normals;
RWStructuredBuffer<float> residual;
RWStructuredBuffer<float4> rotationJacobian;
cbuffer Params { uint count; }
[shader("compute")][numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=count)return;
    float3 p=points[tid.x].xyz,q=targets[tid.x].xyz,n=normals[tid.x].xyz;
    residual[tid.x]=sinterPointPlaneResidual<true>(p,q,n);
    rotationJacobian[tid.x]=float4(sinterPointPlaneRotation(p,n),0.0);
    // Translation Jacobian is n. Caller may store it in its own chosen layout.
}
