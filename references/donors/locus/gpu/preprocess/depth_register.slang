#include "../common/abi.slang"
[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint w=(uint)p[0].x,h=(uint)p[0].y;if(id.x>=w||id.y>=h||p[7].x!=1)return;
float z=input0[index2(id.x,id.y,(uint)p[0].z)].x*p[1].w;if(!validDepth(z))return;
float3 point=float3((float2(id.xy)-p[2].zw)/p[2].xy*z,z);point=transformPoint(point);if(!validDepth(point.z))return;
float2 uv=point.xy/point.z*p[3].xy+p[3].zw;
if(!all(isfinite(uv))||any(uv<-.5)||uv.x>=p[1].x-.5||uv.y>=p[1].y-.5)return;
uint2 q=(uint2)floor(uv+.5);uint ignored;
// Initialize state to float(+infinity) bits; positive depth bit order is numeric order.
InterlockedMin(state[index2(q.x,q.y,(uint)p[1].z)],asuint(point.z),ignored);
}
