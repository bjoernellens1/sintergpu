#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
// Crop pixel coordinates p1.xy + output center mapped through p1.zw extent.
float2 uv=p[1].xy+(float2(id.xy)+.5)/float2(w,h)*p[1].zw;
float4 value=sampleNearest(input0,uv,(uint)p[2].x,(uint)p[2].y,(uint)p[0].z);
float3 v=(value.xyz-p[3].xyz)/p[4].xyz;
output0[index2(id.x,id.y,(uint)p[0].w)]=all(isfinite(v))?float4(v,value.w):float4(0,0,0,0);
}
