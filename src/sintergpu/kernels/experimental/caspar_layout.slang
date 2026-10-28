// SPDX-License-Identifier: Apache-2.0
// Experimental Caspar chunk layout conversion; no hardware qualification.
// Frozen donor cab53390aa111cd5526cb30c9aef484e73521186 layout.py.
// Host validates descriptors, padded ranges and disjoint writable resources.
struct LayoutParams {
    uint n, components, chunkCount, sourceBase;
    uint rowStride, componentStride, outputCount;
};
ConstantBuffer<LayoutParams> params;
StructuredBuffer<uint> sourceWords;
// Seven uints per chunk: start, width, live width, four component IDs.
// Followed by three uints per logical component: chunk start, width, lane.
StructuredBuffer<uint> layoutWords;
RWStructuredBuffer<uint> outputWords;

[shader("compute")][numthreads(256,1,1)]
void packAosoa(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    if(i>=params.outputCount) return;
    // Find the last chunk whose start is <= i. At most 13 iterations for
    // the checked 4096-chunk host-plan bound. No dispatch occurs when n=0.
    uint lo=0u, hi=params.chunkCount;
    while(lo<hi) {
        uint mid=lo+(hi-lo)/2u;
        if(layoutWords[mid*7u]<=i) lo=mid+1u; else hi=mid;
    }
    uint base=(lo-1u)*7u, start=layoutWords[base], width=layoutWords[base+1u];
    uint local=i-start, lane=local%width, value=0u;
    if(lane<layoutWords[base+2u]) {
        uint row=local/width, component=layoutWords[base+3u+lane];
        value=sourceWords[params.sourceBase+row*params.rowStride+component*params.componentStride];
    }
    outputWords[i]=value;
}

[shader("compute")][numthreads(256,1,1)]
void unpackAosoa(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;
    if(i>=params.outputCount) return;
    uint row=i/params.components, component=i%params.components;
    uint base=params.chunkCount*7u+component*3u;
    uint start=layoutWords[base], width=layoutWords[base+1u], lane=layoutWords[base+2u];
    uint value=sourceWords[params.sourceBase+start+row*width+lane];
    outputWords[i]=value;
}
