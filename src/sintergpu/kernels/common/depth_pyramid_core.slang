// SPDX-License-Identifier: Apache-2.0
// Experimental, resource-independent 2x2 reductions. Inputs are in row-major
// order. VKL callers set validity to zero for out-of-bounds samples.
// VKL: bjoernellens1/vkl 3384023a311651ea4260cf12ddab1567972f89f3.
// SLAM: bjoernellens1/slang-rgbd-slam e35e52b35f692b8337696ea3c9d6721d1cc2bc6b.
#ifndef SINTERGPU_DEPTH_PYRAMID_CORE_SLANG
#define SINTERGPU_DEPTH_PYRAMID_CORE_SLANG

struct DepthPyramidVklResult { float depth; uint valid; float confidence; };

DepthPyramidVklResult depthPyramidVklBlock(float depth[4], uint valid[4], float confidence[4], float epsilon)
{
    uint count=0;
    float minimum=asfloat(0x7f800000), maximum=0, sum=0, sumConfidence=0, selectedConfidence=0;
    for (uint k=0; k<4; ++k)
    {
        float d=depth[k];
        if (valid[k]==0 || !isfinite(d) || d<=0) continue;
        float c=confidence[k];
        if (d<minimum) { minimum=d; selectedConfidence=c; }
        maximum=max(maximum,d); sum+=d; sumConfidence+=c; count++;
    }
    bool mean=count==4 && maximum-minimum<=epsilon*32.0;
    DepthPyramidVklResult result;
    result.valid=uint(count>0);
    result.depth=count==0 ? 0 : (mean ? sum/float(count) : minimum);
    result.confidence=count==0 ? 0 : (mean ? sumConfidence/float(count) : selectedConfidence);
    return result;
}

float depthPyramidSlamBlock(float depth[4], float depthJump)
{
    float sum=0, minimum=1e20, maximum=0; uint count=0;
    for (uint k=0; k<4; k++) if (depth[k]>0)
    {
        sum+=depth[k]; count++; minimum=min(minimum,depth[k]); maximum=max(maximum,depth[k]);
    }
    return (count>0 && maximum-minimum<=depthJump) ? sum/count : 0;
}
#endif
