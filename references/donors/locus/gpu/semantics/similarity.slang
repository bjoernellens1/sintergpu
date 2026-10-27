#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
uint features=(uint)p[1].x,pitch=(uint)p[1].y;float dotValue=0,a2=0,b2=0;bool valid=true;
uint base=index2(id.x,id.y,(uint)p[0].z)*pitch;
for(uint k=0;k<features;k++){float a=input0[base+k].x,b=input1[k].x;valid=valid&&isfinite(a)&&isfinite(b);dotValue+=a*b;a2+=a*a;b2+=b*b;}
float value=valid&&a2>0&&b2>0?dotValue/sqrt(a2*b2):0;
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(isfinite(value)?value:0,0,0,1);
}
