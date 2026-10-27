#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
if(id.y!=0||id.x>=(uint)p[0].x)return;float4 v=input0[id.x];if(v.w<=0||!all(isfinite(v.xyz)))return;
float3 cell=floor(v.xyz/p[1].x);uint cap=(uint)p[1].y,probes=min((uint)p[1].z,cap);uint ignored;
if(any(cell<-512)||any(cell>511)){InterlockedAdd(state[2*cap+1],1,ignored);return;}
uint3 q=(uint3)(cell+512);uint key=q.x|(q.y<<10)|(q.z<<20);key+=1;
uint hash=key*2654435761u;
for(uint j=0;j<probes;j++){uint slot=(hash+j)%cap;uint old;
InterlockedCompareExchange(state[2*slot],0,key,old);
if(old==0||old==key){InterlockedAdd(state[2*slot+1],1,ignored);InterlockedAdd(state[2*cap],1,ignored);return;}
InterlockedAdd(state[2*cap+2],1,ignored);}
InterlockedAdd(state[2*cap+1],1,ignored);
}
