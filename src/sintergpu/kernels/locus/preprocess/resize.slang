#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
float2 source=(float2(id.xy)+.5)*p[1].xy/float2(w,h);
output0[index2(id.x,id.y,(uint)p[0].w)]=sampleNearest(input0,source,(uint)p[1].x,(uint)p[1].y,(uint)p[1].z);
}
