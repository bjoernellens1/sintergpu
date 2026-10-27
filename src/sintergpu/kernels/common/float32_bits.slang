// SPDX-License-Identifier: Apache-2.0
#pragma once
// Resource-independent IEEE binary32 round-to-nearest/even arithmetic. Integer
// implementation preserves subnormals on devices whose native FP32 path flushes.
uint64_t sintergpuF32ShiftJam(uint64_t v,uint shift,out bool lost)
{
    if(shift==0) { lost=false;return v; }
    if(shift>=64) { lost=v!=0;return uint64_t(lost); }
    lost=(v&((uint64_t(1)<<shift)-1))!=0;
    return (v>>shift)|uint64_t(lost);
}
uint64_t sintergpuF32RoundRight(uint64_t v,uint shift)
{
    if(shift==0) return v;
    if(shift>=64) return 0;
    uint64_t q=v>>shift,rest=v&((uint64_t(1)<<shift)-1),half=uint64_t(1)<<(shift-1);
    return q+uint64_t(rest>half || (rest==half && (q&1)!=0));
}
uint sintergpuF32MulBits(uint a,uint b)
{
    uint sign=(a^b)&0x80000000u,aa=a&0x7fffffffu,bb=b&0x7fffffffu;
    uint ae=aa>>23,be=bb>>23;
    if(ae==255 || be==255)
    {
        if(aa>0x7f800000u || bb>0x7f800000u || aa==0 || bb==0) return 0x7fc00000u;
        return sign|0x7f800000u;
    }
    if(aa==0 || bb==0) return sign;
    uint64_t am=(aa&0x7fffffu)|(ae!=0?0x800000u:0u),bm=(bb&0x7fffffu)|(be!=0?0x800000u:0u);
    uint64_t product=am*bm;uint hi=uint(product>>32),lo=uint(product);
    int top=hi!=0?32+int(firstbithigh(hi)):int(firstbithigh(lo));
    int sum=int(max(ae,1u))+int(max(be,1u)),exponent=sum-173+top;
    if(exponent>=255) return sign|0x7f800000u;
    if(exponent<=0) return sign|uint(sintergpuF32RoundRight(product,uint(151-sum)));
    uint mantissa=uint(sintergpuF32RoundRight(product,uint(top-23)));
    if(mantissa>=0x1000000u) { mantissa>>=1;exponent++; }
    if(exponent>=255) return sign|0x7f800000u;
    return sign|(uint(exponent)<<23)|(mantissa&0x7fffffu);
}
uint sintergpuF32AddBits(uint a,uint b)
{
    uint aa=a&0x7fffffffu,bb=b&0x7fffffffu;
    if(aa==0 && bb==0) return (a&b)&0x80000000u;
    if(aa==0) return b;if(bb==0) return a;
    if(aa>=0x7f800000u || bb>=0x7f800000u)
    {
        if(aa>0x7f800000u || bb>0x7f800000u || (aa==bb && ((a^b)&0x80000000u)!=0)) return 0x7fc00000u;
        return aa==0x7f800000u?a:b;
    }
    if(aa<bb) { uint t=a;a=b;b=t;t=aa;aa=bb;bb=t; }
    bool subtract=((a^b)&0x80000000u)!=0;
    if(subtract && aa==bb) return 0;
    uint ae=aa>>23,be=bb>>23,exponent=max(ae,1u);
    uint64_t am=uint64_t((aa&0x7fffffu)|(ae!=0?0x800000u:0u))<<3;
    uint64_t bm=uint64_t((bb&0x7fffffu)|(be!=0?0x800000u:0u))<<3;
    bool lost;bm=sintergpuF32ShiftJam(bm,exponent-max(be,1u),lost);
    uint64_t sum=subtract?am-bm:am+bm;
    if(!subtract && sum>=(uint64_t(1)<<27)) { bool ignored;sum=sintergpuF32ShiftJam(sum,1,ignored);exponent++; }
    if(subtract) for(uint i=0;i<24 && sum<(uint64_t(1)<<26) && exponent>1;i++) { sum=(sum<<1)|uint64_t(lost);exponent--; }
    uint q=uint(sum>>3),rest=uint(sum&7);q+=uint(rest>4 || (rest==4 && (q&1)!=0));
    if(q>=0x1000000u) { q>>=1;exponent++; }
    uint sign=a&0x80000000u;
    if(exponent>=255) return sign|0x7f800000u;
    if(exponent==1 && q<0x800000u) return sign|q;
    return sign|(exponent<<23)|(q&0x7fffffu);
}
bool sintergpuF32LessBits(uint a,uint b)
{
    if((a&0x7fffffffu)>0x7f800000u || (b&0x7fffffffu)>0x7f800000u) return false;
    if(((a|b)&0x7fffffffu)==0) return false;
    bool an=(a&0x80000000u)!=0,bn=(b&0x80000000u)!=0;
    if(an!=bn) return an;
    return an?(a&0x7fffffffu)>(b&0x7fffffffu):a<b;
}
