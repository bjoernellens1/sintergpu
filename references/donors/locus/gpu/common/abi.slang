// ABI 1: all image/tensor strides are float4 elements, never bytes.
[[vk::binding(0, 0)]] StructuredBuffer<float4> input0;
[[vk::binding(1, 0)]] StructuredBuffer<float4> input1;
[[vk::binding(2, 0)]] StructuredBuffer<float4> input2;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> output0;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> state;
[[vk::binding(5, 0)]] cbuffer Parameters { float4 p[8]; }
uint index2(uint x,uint y,uint stride) { return y*stride+x; }
bool validDepth(float z) { return isfinite(z) && z>0; }
float4 sampleNearest(StructuredBuffer<float4> image,float2 xy,uint w,uint h,uint stride) {
    if (!all(isfinite(xy)) || any(xy<0) || xy.x>=w || xy.y>=h) return float4(0,0,0,0);
    return image[index2((uint)xy.x,(uint)xy.y,stride)];
}
float3 transformPoint(float3 v) { return float3(dot(p[4].xyz,v)+p[4].w,dot(p[5].xyz,v)+p[5].w,dot(p[6].xyz,v)+p[6].w); }
