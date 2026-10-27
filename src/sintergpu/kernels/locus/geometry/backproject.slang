#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
float z=input0[index2(id.x,id.y,(uint)p[0].z)].x*p[2].x;bool valid=validDepth(z)&&p[7].x==1;
float3 v=float3((float2(id.xy)-p[1].zw)/p[1].xy*z,z);
valid=valid&&all(isfinite(v));
output0[index2(id.x,id.y,(uint)p[0].w)]=valid?float4(v,1):float4(0,0,0,0);
}
