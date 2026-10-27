// SPDX-License-Identifier: Apache-2.0
// Portable FP32 camera profile derived from ITU-R matrix weights and the
// retained EliGSiR-sim color contract (see provenance/color-ports.json).
// Caller owns all resources, dispatch, barriers and 4-byte-padded allocations.
// Byte atomics preserve unrelated packed RGB/pitch bytes without zero-init.

struct CameraParams
{
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint sourcePitch;
    uint uvPitch;
    uint outputPitch;
    uint outputUVPitch;
    uint matrix709;
    uint limited;
};
ConstantBuffer<CameraParams> Params;
ByteAddressBuffer sourceBytes;
ByteAddressBuffer uvBytes;
StructuredBuffer<float2> mapXY;
RWByteAddressBuffer outputBytes;
RWByteAddressBuffer outputUV;

uint loadByte(ByteAddressBuffer buffer, uint address)
{
    return (buffer.Load(address & ~3u) >> ((address & 3u) * 8u)) & 255u;
}

void writeByte(RWByteAddressBuffer buffer, uint address, uint value)
{
    uint shift = (address & 3u) * 8u;
    uint mask = 255u << shift;
    uint previous;
    // Each logical output byte has exactly one writer. Atomic operations on
    // disjoint bit masks commute, including when adjacent rows share a word.
    buffer.InterlockedAnd(address & ~3u, ~mask, previous);
    buffer.InterlockedOr(address & ~3u, (value & 255u) << shift, previous);
}

uint quantize(float value)
{
    precise float bounded = clamp(value, 0.0f, 255.0f);
    uint lower = uint(floor(bounded));
    precise float fraction = bounded - float(lower);
    return lower + uint(fraction > 0.5f || (fraction == 0.5f && (lower & 1u) != 0u));
}

uint3 quantizeRGB(float3 value)
{
    return uint3(quantize(value.x), quantize(value.y), quantize(value.z));
}

float2 weights()
{
    return Params.matrix709 != 0 ? float2(0.2126f,0.0722f) : float2(0.299f,0.114f);
}

uint3 decodePixel(uint yValue, uint uValue, uint vValue)
{
    float2 k = weights();
    precise float kg = (1.0f-k.x)-k.y;
    precise float luma = float(yValue);
    precise float u = float(uValue)-128.0f;
    precise float v = float(vValue)-128.0f;
    if(Params.limited != 0)
    {
        luma = (luma-16.0f)*(255.0f/219.0f);
        u = u*(255.0f/224.0f);
        v = v*(255.0f/224.0f);
    }
    precise float r = luma+(2.0f*(1.0f-k.x))*v;
    precise float g = (luma-((2.0f*k.y*(1.0f-k.y))/kg)*u)-((2.0f*k.x*(1.0f-k.x))/kg)*v;
    precise float b = luma+(2.0f*(1.0f-k.y))*u;
    return quantizeRGB(float3(r,g,b));
}

uint3 readYUYV(uint2 point)
{
    uint pair = point.y*Params.sourcePitch + (point.x/2u)*4u;
    return decodePixel(loadByte(sourceBytes,pair+(point.x & 1u)*2u),
                       loadByte(sourceBytes,pair+1u),loadByte(sourceBytes,pair+3u));
}

uint3 readRGB(uint2 point)
{
    uint address = point.y*Params.sourcePitch+point.x*3u;
    return uint3(loadByte(sourceBytes,address),loadByte(sourceBytes,address+1u),loadByte(sourceBytes,address+2u));
}

void storeRGB(uint2 point, uint3 value)
{
    uint address = point.y*Params.outputPitch+point.x*3u;
    writeByte(outputBytes,address,value.x);
    writeByte(outputBytes,address+1u,value.y);
    writeByte(outputBytes,address+2u,value.z);
}

uint3 bilinear(uint3 a, uint3 b, uint3 c, uint3 d, float2 fraction)
{
    precise float3 top = float3(a)+(float3(b)-float3(a))*fraction.x;
    precise float3 bottom = float3(c)+(float3(d)-float3(c))*fraction.x;
    precise float3 result = top+(bottom-top)*fraction.y;
    return quantizeRGB(result);
}

bool sampleCoordinates(float2 coordinate, out uint2 lo, out uint2 hi, out float2 fraction)
{
    lo=uint2(0); hi=uint2(0); fraction=float2(0);
    if(!all(isfinite(coordinate))) return false;
    precise float2 clamped = clamp(coordinate,float2(0),float2(Params.srcWidth-1u,Params.srcHeight-1u));
    lo=uint2(floor(clamped));
    hi=min(lo+uint2(1),uint2(Params.srcWidth-1u,Params.srcHeight-1u));
    fraction=clamped-float2(lo);
    return true;
}

[shader("compute")]
[numthreads(16,16,1)]
void nv12Decode(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.srcWidth || id.y>=Params.srcHeight) return;
    uint chroma=(id.y/2u)*Params.uvPitch+(id.x/2u)*2u;
    uint3 rgb=decodePixel(loadByte(sourceBytes,id.y*Params.sourcePitch+id.x),
                          loadByte(uvBytes,chroma),loadByte(uvBytes,chroma+1u));
    storeRGB(id.xy,rgb);
}

[shader("compute")]
[numthreads(16,16,1)]
void yuyvDecode(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.srcWidth || id.y>=Params.srcHeight) return;
    storeRGB(id.xy,readYUYV(id.xy));
}

[shader("compute")]
[numthreads(16,16,1)]
void nv12Encode(uint3 id:SV_DispatchThreadID)
{
    // One thread owns the four luma bytes and centered chroma pair of a 2x2 block.
    if(id.x>=Params.srcWidth/2u || id.y>=Params.srcHeight/2u) return;
    float2 k=weights();
    precise float kg=(1.0f-k.x)-k.y;
    precise float cbSum=0.0f;
    precise float crSum=0.0f;
    for(uint dy=0u;dy<2u;++dy)
    for(uint dx=0u;dx<2u;++dx)
    {
        uint2 point=id.xy*2u+uint2(dx,dy);
        float3 rgb=float3(readRGB(point));
        precise float y=(k.x*rgb.x+kg*rgb.y)+k.y*rgb.z;
        precise float cb=(rgb.z-y)/(2.0f*(1.0f-k.y));
        precise float cr=(rgb.x-y)/(2.0f*(1.0f-k.x));
        cbSum=cbSum+cb; crSum=crSum+cr;
        precise float yy=Params.limited != 0 ? y*(219.0f/255.0f)+16.0f : y;
        writeByte(outputBytes,point.y*Params.outputPitch+point.x,quantize(yy));
    }
    precise float scale=Params.limited != 0 ? 224.0f/255.0f : 1.0f;
    precise float u=(cbSum*0.25f)*scale+128.0f;
    precise float v=(crSum*0.25f)*scale+128.0f;
    uint address=id.y*Params.outputUVPitch+id.x*2u;
    writeByte(outputUV,address,quantize(u));
    writeByte(outputUV,address+1u,quantize(v));
}

[shader("compute")]
[numthreads(16,16,1)]
void remapBilinear(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.dstWidth || id.y>=Params.dstHeight) return;
    float2 coordinate=mapXY[id.y*Params.dstWidth+id.x];
    uint2 lo,hi; float2 fraction;
    uint3 value=uint3(0);
    if(sampleCoordinates(coordinate,lo,hi,fraction))
        value=bilinear(readRGB(lo),readRGB(uint2(hi.x,lo.y)),readRGB(uint2(lo.x,hi.y)),readRGB(hi),fraction);
    storeRGB(id.xy,value);
}

[shader("compute")]
[numthreads(16,16,1)]
void resizeBilinear(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.dstWidth || id.y>=Params.dstHeight) return;
    precise float2 coordinate=(float2(id.xy)+0.5f)*float2(Params.srcWidth,Params.srcHeight)/float2(Params.dstWidth,Params.dstHeight)-0.5f;
    uint2 lo,hi; float2 fraction;
    uint3 value=uint3(0);
    if(sampleCoordinates(coordinate,lo,hi,fraction))
        value=bilinear(readRGB(lo),readRGB(uint2(hi.x,lo.y)),readRGB(uint2(lo.x,hi.y)),readRGB(hi),fraction);
    storeRGB(id.xy,value);
}

[shader("compute")]
[numthreads(16,16,1)]
void fusedYUYVRemap(uint3 id:SV_DispatchThreadID)
{
    if(id.x>=Params.dstWidth || id.y>=Params.dstHeight) return;
    float2 coordinate=mapXY[id.y*Params.dstWidth+id.x];
    uint2 lo,hi; float2 fraction;
    uint3 value=uint3(0);
    if(sampleCoordinates(coordinate,lo,hi,fraction))
    {
        // Decoded RGB MUST be quantized to uint8 at each corner before filtering.
        // Shared helpers make this identical to yuyvDecode -> remapBilinear.
        value=bilinear(readYUYV(lo),readYUYV(uint2(hi.x,lo.y)),readYUYV(uint2(lo.x,hi.y)),readYUYV(hi),fraction);
    }
    storeRGB(id.xy,value);
}
