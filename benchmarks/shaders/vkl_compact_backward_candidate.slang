// SPDX-License-Identifier: Apache-2.0
// Experimental logical dense backward, not a GPU-qualified implementation.
// Caller guarantees frozen forward slotSource: per batch each non-sentinel
// source is < items and occurs at most once. Explicit barriers are required.
struct BackwardParams {
    uint batches, items, capacity, bytesPerItem;
    uint logicalBytes, gradBase;
};
ConstantBuffer<BackwardParams> params;
StructuredBuffer<uint> slotSource;
StructuredBuffer<uint> gradWords;
RWStructuredBuffer<uint> inverse;
RWStructuredBuffer<uint> outputWords;

[shader("compute")][numthreads(256,1,1)]
void backwardInit(uint3 tid: SV_DispatchThreadID) {
    uint i=tid.x;
    if(i<params.batches*params.items) inverse[i]=0xffffffffu;
}

[shader("compute")][numthreads(256,1,1)]
void backwardInverse(uint3 tid: SV_DispatchThreadID) {
    uint slot=tid.x;
    if(slot>=params.batches*params.capacity) return;
    uint source=slotSource[slot];
    if(source==0xffffffffu) return;
    // Caller supplies valid unique sources from the validated forward pass.
    if(source<params.items) inverse[(slot/params.capacity)*params.items+source]=slot;
}

[shader("compute")][numthreads(256,1,1)]
void backwardGather(uint3 tid: SV_DispatchThreadID) {
    uint word=tid.x;
    if(word >= (params.logicalBytes+3u)/4u) return;
    uint packed=0;
    for(uint lane=0;lane<4;++lane) {
        uint byteIndex=word*4u+lane;
        if(byteIndex>=params.logicalBytes) continue;
        uint item=byteIndex/params.bytesPerItem;
        uint slot=inverse[item];
        if(slot==0xffffffffu) continue;
        uint src=params.gradBase+slot*params.bytesPerItem+byteIndex%params.bytesPerItem;
        uint byteValue=(gradWords[src>>2u]>>((src&3u)*8u))&255u;
        packed|=byteValue<<(lane*8u);
    }
    outputWords[word]=packed;
}
