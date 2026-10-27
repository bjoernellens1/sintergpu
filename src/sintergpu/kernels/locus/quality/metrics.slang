#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h)return;
uint stride=(uint)p[0].z;float3 center=input0[index2(id.x,id.y,stride)].xyz;
if(!all(isfinite(center))){output0[index2(id.x,id.y,(uint)p[0].w)]=float4(0,0,0,validDepth(input1[index2(id.x,id.y,(uint)p[1].z)].x)?1:0);return;}
float c=dot(center,float3(.2126,.7152,.0722));
float l=dot(input0[index2(id.x>0?id.x-1:0,id.y,stride)].xyz,float3(.2126,.7152,.0722));
float r=dot(input0[index2(min(id.x+1,w-1),id.y,stride)].xyz,float3(.2126,.7152,.0722));
float u=dot(input0[index2(id.x,id.y>0?id.y-1:0,stride)].xyz,float3(.2126,.7152,.0722));
float d=dot(input0[index2(id.x,min(id.y+1,h-1),stride)].xyz,float3(.2126,.7152,.0722));
l=isfinite(l)?l:c;r=isfinite(r)?r:c;u=isfinite(u)?u:c;d=isfinite(d)?d:c;
output0[index2(id.x,id.y,(uint)p[0].w)]=float4(abs(r-l)+abs(d-u),abs(l+r+u+d-4*c),any(center<=p[1].x)||any(center>=p[1].y)?1:0,validDepth(input1[index2(id.x,id.y,(uint)p[1].z)].x)?1:0);
}
