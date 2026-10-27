#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
float4 m=input0[index2(id.x,id.y,(uint)p[0].z)];uint ignored;
// Saturated fixed-point metric totals. Host caps pixels to 2^20; no float atomics.
InterlockedAdd(state[0],(uint)(clamp(isfinite(m.x)?m.x:0,0,4)*256+.5),ignored);
InterlockedAdd(state[1],(uint)(clamp(isfinite(m.y)?m.y:0,0,4)*256+.5),ignored);
InterlockedAdd(state[2],m.z>0?1:0,ignored);InterlockedAdd(state[3],m.w>0?1:0,ignored);
}
