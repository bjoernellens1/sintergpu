#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
if(id.y!=0||id.x>=(uint)p[0].x)return;uint key=state[2*id.x];
if(key==0){output0[id.x]=0;return;}key-=1;
float3 q=float3(key&1023,(key>>10)&1023,(key>>20)&1023)-512;
output0[id.x]=float4((q+.5)*p[1].x,1);
}
