// SPDX-License-Identifier: Apache-2.0
// Stable uint32 payload compaction. Prefixes come from scan.slang with SINTERGPU_SCAN_UINT=1.
StructuredBuffer<uint> predicate;
StructuredBuffer<uint> payload;
StructuredBuffer<uint> prefix;
RWStructuredBuffer<uint> flags;
RWStructuredBuffer<uint> segmentIds;
RWStructuredBuffer<uint> compacted;
RWStructuredBuffer<uint> selectedCount;
uint count;
[shader("compute")][numthreads(256,1,1)]
void compactPrepare(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    // Dispatch at least one group even for empty input: initialize the device count.
    if(i==0) selectedCount[0]=0;
    if(i>=count) return;
    flags[i]=uint(predicate[i]!=0);
    segmentIds[i]=0;
}
[shader("compute")][numthreads(256,1,1)]
void compactScatter(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    if(i>=count) return;
    if(predicate[i]!=0) compacted[prefix[i]-1]=payload[i];
    if(i+1==count) selectedCount[0]=prefix[i];
}
