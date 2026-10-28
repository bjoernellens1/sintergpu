// SPDX-License-Identifier: Apache-2.0
// Experimental, serial reductions; the caller must enforce docs/depth-evidence-map.md.
import experimental.depth_evidence_map;
struct DepthEvidenceParams {
    uint batches; uint height; uint width; uint gridHeight;
    uint gridWidth; uint useConfidence; float nearBound; float farBound;
};
ConstantBuffer<DepthEvidenceParams> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
RWStructuredBuffer<float> globalOut;
RWStructuredBuffer<float> gridOut;

DepthEvidenceMap pixel(uint b, uint y, uint x) {
    uint i = (b * params.height + y) * params.width + x;
    bool hasRight = x + 1 < params.width;
    bool hasDown = y + 1 < params.height;
    float right = hasRight ? depth[i + 1] : 0.0;
    float down = hasDown ? depth[i + params.width] : 0.0;
    bool rightValid = hasRight ? valid[i + 1] != 0 : false;
    bool downValid = hasDown ? valid[i + params.width] != 0 : false;
    float c = params.useConfidence != 0 ? confidence[i] : 1.0;
    return depthEvidenceMap(depth[i], valid[i] != 0, c, right, rightValid,
                            down, downValid, params.nearBound, params.farBound);
}
uint ceilRatio(uint numerator, uint denominator) {
    return numerator / denominator + uint(numerator % denominator != 0);
}

// Eight output floats per batch: valid, hole, near, far, variance, edge, confidence, stable.
[shader("compute")][numthreads(64, 1, 1)]
void depthEvidenceGlobal(uint3 tid : SV_DispatchThreadID) {
    uint b = tid.x;
    if (b >= params.batches) return;
    float support=0, nearCount=0, farCount=0, edges=0, conf=0, stable=0, pairs=0, squared=0;
    for (uint y=0; y<params.height; ++y) {
        for (uint x=0; x<params.width; ++x) {
            DepthEvidenceMap p = pixel(b,y,x);
            support+=p.support; nearCount+=p.nearCount; farCount+=p.farCount;
            edges+=p.edge; conf+=p.confidence; stable+=p.stable;
            pairs+=p.pairCount; squared+=p.pairSquaredDifference;
        }
    }
    float safe = max(support, 1.0);
    float vf = support / float(params.height * params.width);
    uint o = b * 8;
    globalOut[o]=vf; globalOut[o+1]=1.0-vf; globalOut[o+2]=nearCount/safe;
    globalOut[o+3]=farCount/safe; globalOut[o+4]=squared/(4.0*max(pairs,1.0));
    globalOut[o+5]=edges/safe; globalOut[o+6]=conf/safe; globalOut[o+7]=stable/safe;
}

// Eight floats per bin in NHWC order: support/area, near, far, depth, depth^2, edge, confidence, stable.
[shader("compute")][numthreads(64, 1, 1)]
void depthEvidenceGrid(uint3 tid : SV_DispatchThreadID) {
    uint gridSize = params.gridHeight * params.gridWidth;
    uint index = tid.x;
    if (index >= params.batches * gridSize) return;
    uint b = index / gridSize;
    uint bin = index % gridSize;
    uint gy = bin / params.gridWidth;
    uint gx = bin % params.gridWidth;
    uint y0=ceilRatio(gy*params.height,params.gridHeight);
    uint y1=ceilRatio((gy+1)*params.height,params.gridHeight);
    uint x0=ceilRatio(gx*params.width,params.gridWidth);
    uint x1=ceilRatio((gx+1)*params.width,params.gridWidth);
    float support=0, nearCount=0, farCount=0, sum=0, sum2=0, edges=0, conf=0, stable=0;
    for (uint y=y0; y<y1; ++y) {
        for (uint x=x0; x<x1; ++x) {
            DepthEvidenceMap p = pixel(b,y,x);
            support+=p.support; nearCount+=p.nearCount; farCount+=p.farCount;
            sum+=p.depth; sum2+=p.depthSquared; edges+=p.edge;
            conf+=p.confidence; stable+=p.stable;
        }
    }
    float safe=max(support,1.0);
    uint o=index*8;
    gridOut[o]=support/float((y1-y0)*(x1-x0)); gridOut[o+1]=nearCount/safe;
    gridOut[o+2]=farCount/safe; gridOut[o+3]=sum/safe; gridOut[o+4]=sum2/safe;
    gridOut[o+5]=edges/safe; gridOut[o+6]=conf/safe; gridOut[o+7]=stable/safe;
}
