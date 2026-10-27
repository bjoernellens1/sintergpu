#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
if(id.y!=0||id.x>=(uint)p[0].x)return;float4 v=input0[id.x];float3 q=transformPoint(v.xyz);
output0[id.x]=v.w>0&&all(isfinite(q))?float4(q,1):float4(0,0,0,0);
}
