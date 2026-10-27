#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y; if(id.x>=w||id.y>=h)return;
uint yStride=(uint)p[0].z,uvStride=(uint)p[1].x;
// Input floats carry byte values. p1.y=1 limited range; p1.z=1 BT.709, else BT.601.
float y=input0[index2(id.x,id.y,yStride)].x;
float u=input1[index2(id.x/2,id.y/2,uvStride)].x-128;
float v=input2[index2(id.x/2,id.y/2,uvStride)].x-128;
if(p[1].y==1){ y=(y-16)*(255.0/219.0);u*=255.0/224.0;v*=255.0/224.0; }
float3 rgb=p[1].z==1 ? float3(y+1.5748*v,y-.187324*u-.468124*v,y+1.8556*u) : float3(y+1.402*v,y-.344136*u-.714136*v,y+1.772*u);
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(saturate(rgb/255.0),1);
}
