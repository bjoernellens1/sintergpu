// SPDX-License-Identifier: Apache-2.0
// Frozen VKL temporal_best_observation forward numerical profile, row-major BxN.
struct TemporalBestParams { uint batches; uint neighbors; uint useQuality; uint useConfidence; };
ConstantBuffer<TemporalBestParams> params;
StructuredBuffer<uint> equivalence;
StructuredBuffer<float> target_quality;
StructuredBuffer<float> neighbor_quality;
StructuredBuffer<float> pairwise_quality;
StructuredBuffer<float> pairwise_confidence;
RWStructuredBuffer<int> best_indexOut;
RWStructuredBuffer<float> best_qualityOut;
RWStructuredBuffer<float> replaceabilityOut;
RWStructuredBuffer<float> replacement_confidenceOut;
[shader("compute")]
[numthreads(256,1,1)]
void temporalBest(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.batches)return;
    uint bestIndex=0;float best=asfloat(0xff800000);
    for(uint n=0;n<params.neighbors;n++)
    {
        uint j=i*params.neighbors+n;
        float q=equivalence[j]!=0?(params.useQuality!=0?pairwise_quality[j]:neighbor_quality[j]):asfloat(0xff800000);
        // Torch max selects the first NaN, otherwise the first maximum.
        if(!isnan(best)&&(isnan(q)||q>best)){best=q;bestIndex=n;}
    }
    float target=target_quality[i];bool targetWins=!isfinite(best)||target>=best;
    float difference=best-target;
    float advantage=targetWins?0.0:(isnan(difference)?asfloat(0x7fc00000):max(difference,0.0));
    uint j=i*params.neighbors+bestIndex;
    float confidence=params.useConfidence!=0?pairwise_confidence[j]:float(equivalence[j]!=0);
    float replacement=confidence*advantage;
    // Preserve the donor's IEEE product even if the compiler folds 0*Inf.
    if(!targetWins&&(isnan(advantage)||(isinf(advantage)&&confidence==0.0)))replacement=asfloat(0x7fc00000);
    best_indexOut[i]=targetWins?0:int(bestIndex+1);
    best_qualityOut[i]=targetWins?target:best;
    replaceabilityOut[i]=advantage;
    replacement_confidenceOut[i]=targetWins?0.0:replacement;
}
