#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
if(p[7].x!=1){output0[index2(id.x,id.y,(uint)p[0].w)]=0;return;}
float2 q=(float2(id.xy)-p[1].zw)/p[1].xy;float r2=dot(q,q);
float radial=1+p[2].x*r2+p[2].y*r2*r2+p[2].z*r2*r2*r2;
float2 distorted=q*radial+float2(2*p[3].x*q.x*q.y+p[3].y*(r2+2*q.x*q.x),p[3].x*(r2+2*q.y*q.y)+2*p[3].y*q.x*q.y);
float2 uv=distorted*p[1].xy+p[1].zw;
output0[index2(id.x,id.y,(uint)p[0].w)]=sampleNearest(input0,uv+.5,w,h,(uint)p[0].z);
}
