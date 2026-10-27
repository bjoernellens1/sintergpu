#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
float z=asfloat(state[index2(id.x,id.y,(uint)p[0].z)]);bool valid=validDepth(z);
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(valid?z:0,valid?1:0,0,0);
}
