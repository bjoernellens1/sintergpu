#define SINTERGPU_SCAN_UINT 1
// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Ruilong Li, UC Berkeley. Copyright (c) 2025 AMD-AIOSS.
// Segmented scan semantics ported from nerfacc; see provenance/primitives.json.
// Original helper lineage: PyTorch ScanUtils.cuh at 06a64f7eaa47ce430a3fa61016010075b59b18a7.
// Parallel hierarchical adaptation uses caller-owned scratch and explicit passes.
#if SINTERGPU_SCAN_UINT
#define ScanScalar uint
#else
#define ScanScalar float
#endif
StructuredBuffer<ScanScalar> values;
StructuredBuffer<uint> segmentIds;
RWStructuredBuffer<ScanScalar> output;
RWStructuredBuffer<ScanScalar> blockValues;
RWStructuredBuffer<uint> blockIds;
StructuredBuffer<ScanScalar> scannedBlocks;
StructuredBuffer<ScanScalar> inclusiveValues;
uint count;
uint product;
groupshared ScanScalar localValues[256];
groupshared uint localIds[256];
ScanScalar identity() { return product != 0 ? ScanScalar(1) : ScanScalar(0); }
ScanScalar combine(ScanScalar a, ScanScalar b) { return product != 0 ? a*b : a+b; }

// Segment IDs must be nondecreasing: each segment is a contiguous run, IDs never reused.
[shader("compute")][numthreads(256,1,1)]
void scanBlocks(uint3 group:SV_GroupID, uint lane:SV_GroupIndex) {
    uint index=group.x*256+lane;
    bool active=index<count;
    localValues[lane]=active ? values[index] : identity();
    localIds[lane]=active ? segmentIds[index] : 0;
    GroupMemoryBarrierWithGroupSync();
    for(uint offset=1;offset<256;offset<<=1) {
        ScanScalar value=localValues[lane];
        if(active && lane>=offset && localIds[lane]==localIds[lane-offset])
            value=combine(localValues[lane-offset],value);
        // Complete every read before any lane overwrites the preceding stage.
        GroupMemoryBarrierWithGroupSync();
        localValues[lane]=value;
        GroupMemoryBarrierWithGroupSync();
    }
    if(active) {
        output[index]=localValues[lane];
        if(lane==255 || index+1==count) {
            blockValues[group.x]=localValues[lane];
            blockIds[group.x]=localIds[lane];
        }
    }
}

// Run after the recursively scanned block totals and before lower carry passes.
[shader("compute")][numthreads(256,1,1)]
void scanCarry(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    if(i>=count) return;
    uint block=i/256;
    if(block>0 && segmentIds[i]==blockIds[block-1])
        output[i]=combine(scannedBlocks[block-1],output[i]);
}

// Separate source/destination avoids cross-thread read/write races and division by zero.
[shader("compute")][numthreads(256,1,1)]
void scanExclusive(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    if(i>=count) return;
    output[i]=(i==0 || segmentIds[i]!=segmentIds[i-1]) ? identity() : inclusiveValues[i-1];
}
