#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
uint slot=index2(id.x,id.y,(uint)p[0].z);uint count=state[2*slot+1];float t=saturate(float(count)/p[1].x);
output0[index2(id.x,id.y,(uint)p[0].w)]=count>0?float4(1-t,t,.1,1):float4(0,0,0,1);
}
