// SPDX-License-Identifier: Apache-2.0
#include "../common/brief_pattern.slang"
#include "../common/float32_bits.slang"
struct BriefParams { uint batches,channels,height,width,points,bits,defaultPairs,usePixelsValid,useRgbValid; };
ConstantBuffer<BriefParams> Params;
StructuredBuffer<float> rgb;
StructuredBuffer<int64_t> pixels;
StructuredBuffer<int64_t> pairs;
StructuredBuffer<uint> pixelsValid;
StructuredBuffer<uint> rgbValid;
RWStructuredBuffer<uint> descriptorOut;
RWStructuredBuffer<uint> briefValidOut;
groupshared uint supported[256];

bool colorAt(uint batch, int64_t x, int64_t y, out uint gray)
{
    gray=0;
    if(x<0 || y<0 || x>=int64_t(Params.width) || y>=int64_t(Params.height)) return false;
    uint area=Params.height*Params.width, at=batch*area+uint(y)*Params.width+uint(x);
    if(Params.useRgbValid!=0) { if(rgbValid[at]==0) return false; }
    uint base=batch*Params.channels*area+uint(y)*Params.width+uint(x);
    uint r=asuint(rgb[base]);
    if((r&0x7f800000u)==0x7f800000u) return false;
    if(Params.channels==1) { gray=r; return true; }
    uint g=asuint(rgb[base+area]), b=asuint(rgb[base+2*area]);
    if((g&0x7f800000u)==0x7f800000u || (b&0x7f800000u)==0x7f800000u) return false;
    uint wr=sintergpuF32MulBits(r,asuint(0.299f)),wg=sintergpuF32MulBits(g,asuint(0.587f)),wb=sintergpuF32MulBits(b,asuint(0.114f));
    gray=sintergpuF32AddBits(sintergpuF32AddBits(wr,wg),wb);
    return true;
}

[shader("compute")]
[numthreads(256,1,1)]
void briefDescribe(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint point=group.x;
    if(point>=Params.batches*Params.points) return;
    uint batch=point/Params.points;
    int64_t x=pixels[point*2],y=pixels[point*2+1]; uint ignored;
    bool center=colorAt(batch,x,y,ignored);
    if(Params.usePixelsValid!=0) center=center && pixelsValid[point]!=0;
    uint valid=uint(center);
    for(uint bit=lane;bit<Params.bits;bit+=256)
    {
        uint comparison=0;
        if(center)
        {
            VklBriefPair p;
            if(Params.defaultPairs!=0) p=vklBriefPair(bit);
            else { p.ax=pairs[bit*4];p.ay=pairs[bit*4+1];p.bx=pairs[bit*4+2];p.by=pairs[bit*4+3]; }
            // Check offsets before adding: valid centers make these bounds safe
            // even for INT64_MIN/MAX custom offsets (no wrapped address casts).
            int64_t lx=-x,hx=int64_t(Params.width)-1-x,ly=-y,hy=int64_t(Params.height)-1-y;
            bool inside=p.ax>=lx && p.ax<=hx && p.bx>=lx && p.bx<=hx && p.ay>=ly && p.ay<=hy && p.by>=ly && p.by<=hy;
            bool ok=false;
            if(inside) { uint a,b; bool av=colorAt(batch,x+p.ax,y+p.ay,a);bool bv=colorAt(batch,x+p.bx,y+p.by,b);ok=av && bv;comparison=uint(sintergpuF32LessBits(a,b)); }
            valid&=uint(ok);
        }
        descriptorOut[point*Params.bits+bit]=comparison;
    }
    supported[lane]=valid;GroupMemoryBarrierWithGroupSync();
    for(uint stride=128;stride!=0;stride/=2) { if(lane<stride) supported[lane]&=supported[lane+stride];GroupMemoryBarrierWithGroupSync(); }
    if(lane==0) briefValidOut[point]=supported[0];
    if(supported[0]==0) for(uint bit=lane;bit<Params.bits;bit+=256) descriptorOut[point*Params.bits+bit]=0;
}
