// SPDX-License-Identifier: Apache-2.0
// Experimental scalar32 ABI for VKL compact.py revision
// 3384023a311651ea4260cf12ddab1567972f89f3. Not catalogued or qualified.
// Host validates all offsets/ranges/shape products, padded word buffers and
// disjoint writable ranges. shapeStrides contains uint2(extent, byteStride)
// for each of rank dimensions, beginning with batch and source item.
struct CompactParams {
    uint batches, items, capacity, rank;
    uint itemBytes, payloadItems, logicalValueBytes, actionBytes;
    uint valueBase, maskBase, maskBatchStride, maskItemStride;
    uint actionBase, actionBatchStride, actionItemStride;
};
ConstantBuffer<CompactParams> params;
StructuredBuffer<uint> maskWords;
StructuredBuffer<uint> valueWords;
StructuredBuffer<uint> actionWords;
StructuredBuffer<uint2> shapeStrides;
StructuredBuffer<uint> prefix;
RWStructuredBuffer<uint> flags;
RWStructuredBuffer<uint> segmentIds;
RWStructuredBuffer<uint> slotToSource;
RWStructuredBuffer<uint> indicesOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<uint> actionsOut;
RWStructuredBuffer<uint> countsOut;
RWStructuredBuffer<uint> valuesOut;

uint maskByte(uint offset) { return (maskWords[offset>>2] >> ((offset&3)*8)) & 255u; }
uint valueByte(uint offset) { return (valueWords[offset>>2] >> ((offset&3)*8)) & 255u; }
uint actionByte(uint offset) { return (actionWords[offset>>2] >> ((offset&3)*8)) & 255u; }

[shader("compute")][numthreads(256,1,1)]
void compactInit(uint3 tid : SV_DispatchThreadID) {
    uint i=tid.x;
    uint slots=params.batches*params.capacity;
    if(i<slots) {
        slotToSource[i]=0xffffffffu;
        indicesOut[2*i]=0; indicesOut[2*i+1]=0;
        validOut[i]=0;
        uint words=params.actionBytes==8 ? 2 : 1;
        actionsOut[i*words]=0;
        if(words==2) actionsOut[i*words+1]=0;
    }
    if(i<params.batches) { countsOut[2*i]=0; countsOut[2*i+1]=0; }
}

[shader("compute")][numthreads(256,1,1)]
void compactFlags(uint3 tid : SV_DispatchThreadID) {
    uint i=tid.x, count=params.batches*params.items;
    if(i>=count) return;
    uint b=i/params.items, n=i%params.items;
    flags[i]=uint(maskByte(params.maskBase+b*params.maskBatchStride+n*params.maskItemStride)!=0);
    segmentIds[i]=b;
}

[shader("compute")][numthreads(256,1,1)]
void compactScatterMetadata(uint3 tid : SV_DispatchThreadID) {
    uint i=tid.x, count=params.batches*params.items;
    if(i>=count) return;
    uint b=i/params.items, n=i%params.items;
    if(maskByte(params.maskBase+b*params.maskBatchStride+n*params.maskItemStride)==0) return;
    uint rank=prefix[i];
    if(rank==0 || rank>params.capacity) return;
    uint slot=b*params.capacity+rank-1;
    slotToSource[slot]=n; indicesOut[2*slot]=n; indicesOut[2*slot+1]=0; validOut[slot]=1;
    if(params.actionBytes!=0) {
        uint offset=params.actionBase+b*params.actionBatchStride+n*params.actionItemStride;
        uint low=0;
        for(uint k=0;k<4;++k) low|=actionByte(offset+k)<<(8*k);
        uint words=params.actionBytes==8 ? 2 : 1;
        actionsOut[slot*words]=low;
        if(words==2) {
            uint high=0;
            for(uint k=0;k<4;++k) high|=actionByte(offset+4+k)<<(8*k);
            actionsOut[slot*words+1]=high;
        }
    }
}

[shader("compute")][numthreads(256,1,1)]
void compactFinalizeCounts(uint3 tid : SV_DispatchThreadID) {
    uint b=tid.x;
    if(b>=params.batches) return;
    uint count=params.items==0 ? 0 : min(prefix[(b+1)*params.items-1],params.capacity);
    countsOut[2*b]=count; countsOut[2*b+1]=0;
}

[shader("compute")][numthreads(256,1,1)]
void compactGatherWords(uint3 tid : SV_DispatchThreadID) {
    uint wordIndex=tid.x;
    uint words=(params.logicalValueBytes+3u)/4u;
    if(wordIndex>=words) return;
    uint packed=0;
    uint bytesPerSlot=params.payloadItems*params.itemBytes;
    for(uint lane=0;lane<4;++lane) {
        uint outputByte=wordIndex*4+lane;
        if(outputByte>=params.logicalValueBytes) continue;
        uint slot=outputByte/bytesPerSlot;
        uint inputIndex=slotToSource[slot];
        if(inputIndex==0xffffffffu) continue;
        uint b=slot/params.capacity;
        uint inSlot=outputByte%bytesPerSlot;
        uint element=inSlot/params.itemBytes;
        uint offset=params.valueBase+b*shapeStrides[0].y+inputIndex*shapeStrides[1].y+inSlot%params.itemBytes;
        for(uint dim=params.rank;dim>2;--dim) {
            uint2 descriptor=shapeStrides[dim-1];
            uint coordinate=element%descriptor.x;
            element/=descriptor.x;
            offset+=coordinate*descriptor.y;
        }
        packed|=valueByte(offset)<<(lane*8);
    }
    valuesOut[wordIndex]=packed;
}
