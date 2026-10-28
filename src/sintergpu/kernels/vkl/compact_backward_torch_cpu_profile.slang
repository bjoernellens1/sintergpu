// SPDX-License-Identifier: Apache-2.0
// Experimental Torch CPU-observed logical gradient bit profile only.
// Not a claim of Torch GPU or framework parity. See profile tests.
// Caller validates forward slots (unique/in-range per batch), all bounds and
// storage ownership, and supplies barriers between initialization/passes.
struct ProfileParams {
    uint batches, items, capacity, bytesPerItem;
    uint logicalBytes, gradBase, componentBytes;
};
ConstantBuffer<ProfileParams> params;
StructuredBuffer<uint> slotSource;
StructuredBuffer<uint> gradWords;
RWStructuredBuffer<uint> inverse;
RWStructuredBuffer<uint> outputWords;
uint loadByte(uint p) { return (gradWords[p>>2u]>>((p&3u)*8u))&255u; }

// IEEE +0 + component: normalize negative zero and quiet signalling NaNs.
// Works with unaligned input and uses no floating point arithmetic.
uint profileByte(uint componentStart,uint index) {
    uint c=params.componentBytes; uint out=loadByte(componentStart+index);
    uint high=0,low=0;
    for(uint i=0;i<c;++i) {
        uint v=loadByte(componentStart+i);
        if(i<4u) low|=v<<(8u*i); else high|=v<<(8u*(i-4u));
    }
    bool negativeZero=false, nan=false;
    if(c==2u) { negativeZero=low==0x8000u; nan=(low&0x7c00u)==0x7c00u && (low&0x03ffu)!=0u; }
    else if(c==4u) { negativeZero=low==0x80000000u; nan=(low&0x7f800000u)==0x7f800000u && (low&0x007fffffu)!=0u; }
    else { negativeZero=high==0x80000000u && low==0u; nan=(high&0x7ff00000u)==0x7ff00000u && ((high&0x000fffffu)!=0u || low!=0u); }
    if(negativeZero && index==c-1u) out=0u;
    if(nan && index==(c==2u?1u:(c==4u?2u:6u))) out|=(c==2u?2u:(c==4u?64u:8u));
    return out;
}
[shader("compute")][numthreads(256,1,1)]
void backwardProfileInit(uint3 tid:SV_DispatchThreadID) {
    if(tid.x<params.batches*params.items) inverse[tid.x]=0xffffffffu;
}
[shader("compute")][numthreads(256,1,1)]
void backwardProfileInverse(uint3 tid:SV_DispatchThreadID) {
    uint slot=tid.x;
    if(slot>=params.batches*params.capacity) return;
    uint source=slotSource[slot];
    if(source!=0xffffffffu && source<params.items) inverse[(slot/params.capacity)*params.items+source]=slot;
}
[shader("compute")][numthreads(256,1,1)]
void backwardProfileGather(uint3 tid:SV_DispatchThreadID) {
    uint word=tid.x;
    if(word>=(params.logicalBytes+3u)/4u) return;
    uint packed=0;
    for(uint lane=0;lane<4u;++lane) {
        uint pos=word*4u+lane;
        if(pos>=params.logicalBytes) continue;
        uint slot=inverse[pos/params.bytesPerItem];
        if(slot==0xffffffffu) continue;
        uint inItem=pos%params.bytesPerItem;
        uint inComponent=inItem%params.componentBytes;
        uint componentStart=params.gradBase+slot*params.bytesPerItem+inItem-inComponent;
        packed|=profileByte(componentStart,inComponent)<<(lane*8u);
    }
    outputWords[word]=packed;
}
