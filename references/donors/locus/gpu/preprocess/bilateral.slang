#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
float z=input0[index2(id.x,id.y,(uint)p[0].z)].x;float sum=0,weight=0;
if(validDepth(z))for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
int2 q=int2(id.xy)+int2(dx,dy);if(any(q<0)||q.x>=w||q.y>=h)continue;
float n=input0[index2((uint)q.x,(uint)q.y,(uint)p[0].z)].x;if(!validDepth(n))continue;
float delta=n-z;float a=exp(-float(dx*dx+dy*dy)/(2*p[1].x*p[1].x)-delta*delta/(2*p[1].y*p[1].y));sum+=a*n;weight+=a;}
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(weight>0?sum/weight:0,weight>0?1:0,0,0);
}
