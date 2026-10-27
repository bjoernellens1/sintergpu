#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
// Inclusive threshold. p1.y=1 performs bounded 3x3 dilation, 2 erosion, 0 threshold only.
bool result=p[1].y==2;int radius=p[1].y==0?0:1;
for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++){
int2 q=int2(id.xy)+int2(dx,dy);bool hit=false;
if(all(q>=0)&&q.x<w&&q.y<h){float v=input0[index2((uint)q.x,(uint)q.y,(uint)p[0].z)].x;hit=isfinite(v)&&v>=p[1].x;}
result=p[1].y==2?result&&hit:result||hit;}
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(result?1:0,0,0,1);
}
