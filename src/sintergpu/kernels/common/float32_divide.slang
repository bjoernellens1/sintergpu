// SPDX-License-Identifier: Apache-2.0
#pragma once
// Resource-independent IEEE binary32 division, round to nearest/ties to even.
// Only scalar32 integer arithmetic; preserves subnormals without native FP modes.
// Input NaN payloads are unspecified. The result of invalid arithmetic is NaN.
uint sintergpuF32DivideBits(uint a,uint b)
{
    uint sign=(a^b)&0x80000000u,aa=a&0x7fffffffu,bb=b&0x7fffffffu;
    if(aa>0x7f800000u || bb>0x7f800000u || (aa==0 && bb==0) || (aa==0x7f800000u && bb==0x7f800000u))return 0x7fc00000u;
    if(bb==0 || aa==0x7f800000u)return sign|0x7f800000u;
    if(aa==0 || bb==0x7f800000u)return sign;
    uint ae=aa>>23,be=bb>>23;
    uint am=(aa&0x7fffffu)|(ae!=0?0x800000u:0u),bm=(bb&0x7fffffu)|(be!=0?0x800000u:0u);
    int sa=23-int(firstbithigh(am)),sb=23-int(firstbithigh(bm));
    am<<=uint(sa);bm<<=uint(sb);
    int exponent=int(max(ae,1u))-int(max(be,1u))-sa+sb+127;
    if(am<bm){am<<=1;exponent--;}
    if(exponent>=255)return sign|0x7f800000u;
    if(exponent<-23)return sign;
    // Exactly half the smallest subnormal rounds to even zero.
    if(exponent==-23)return sign|uint(am>bm);
    uint top=exponent>0?23u:uint(exponent+22),q=1u<<top,remainder=am-bm;
    // Normalized quotient lies in [1,2); remainder stays <24 bits. Each step
    // produces one quotient bit; every invocation takes at most23 steps.
    for(int bit=int(top)-1;bit>=0;bit--)
    {
        remainder<<=1;
        if(remainder>=bm){remainder-=bm;q|=1u<<uint(bit);}
    }
    uint twice=remainder<<1;
    q+=uint(twice>bm || (twice==bm && (q&1u)!=0));
    if(exponent<=0)return sign|q;
    if(q>=0x1000000u){q>>=1;exponent++;}
    if(exponent>=255)return sign|0x7f800000u;
    return sign|(uint(exponent)<<23)|(q&0x7fffffu);
}
