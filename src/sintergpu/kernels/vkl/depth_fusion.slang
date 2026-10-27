// SPDX-License-Identifier: Apache-2.0
// Source-major [S,B,1,H,W], contiguous float32; typed uint32 logical masks.
// No image-wide loops or hidden state: one independent pixel per invocation.
struct FusionParams { uint count; uint sources; uint relative; uint minSupport; uint minTemporalSupport; float threshold; };
ConstantBuffer<FusionParams> params;
StructuredBuffer<float> depths;
StructuredBuffer<uint> valids;
StructuredBuffer<float> confidences;
StructuredBuffer<uint> visibility;
StructuredBuffer<uint> consistency;
StructuredBuffer<uint> trusted;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
RWStructuredBuffer<int> winning_sourceOut;
RWStructuredBuffer<int> support_countOut;
RWStructuredBuffer<int> temporal_support_countOut;
RWStructuredBuffer<float> depth_varianceOut;
RWStructuredBuffer<int> conflict_owner_sourceOut;
RWStructuredBuffer<int> dominant_contributorOut;
RWStructuredBuffer<float> temporal_weight_fractionOut;
RWStructuredBuffer<float> source_disagreementOut;
static const float EPS=1.1920928955078125e-7;
// GPU division permits approximation. Preserve exact power-of-two ratios without
// float64: equal significands differ only by exponent. This keeps constant-depth
// fields fixed instead of creating false normals in a following geometry pass.
// Other ratios retain ordinary target float32 division accuracy.
float stableRatio(float a,float b)
{
    uint ua=asuint(a),ub=asuint(b);
    uint ea=(ua>>23)&255,eb=(ub>>23)&255;
    int exponent=int(ea)-int(eb)+127;
    if((ua&0x80000000)==0&&(ub&0x80000000)==0&&ea>0&&ea<255&&eb>0&&eb<255
       &&(ua&0x7fffff)==(ub&0x7fffff)&&exponent>0&&exponent<255)
        return asfloat(uint(exponent)<<23);
    return a/b;
}
bool usable(uint s,uint i)
{
    uint j=s*params.count+i;float d=depths[j];
    return valids[j]!=0&&isfinite(d)&&d>0&&confidences[j]>0;
}
bool temporalUsable(uint s,uint i)
{
    uint j=s*params.count+i;
    return usable(s,i)&&(s==0||(visibility[j]!=0&&consistency[j]!=0));
}
[shader("compute")]
[numthreads(256,1,1)]
void fuseDepth(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.count)return;
    float total=0,numerator=0,bestScore=-1;int winner=0;uint support=0;
    for(uint s=0;s<params.sources;s++)
    {
        uint j=s*params.count+i;bool use=usable(s,i);
        float w=use?confidences[j]:0.0,d=use?depths[j]:0.0;
        total+=w;numerator+=w*d;support+=uint(use);
        float score=params.relative!=0?w*.5:w;
        if(score>bestScore){bestScore=score;winner=int(s);}
    }
    float mean=stableRatio(numerator,max(total,EPS)),agreeW=0,agreeD=0;uint agrees=0;
    for(uint s=0;s<params.sources;s++)
    {
        uint j=s*params.count+i;
        bool agree=usable(s,i)&&abs(depths[j]-mean)<=params.threshold*max(mean,EPS);
        float w=agree?confidences[j]:0.0,d=agree?depths[j]:0.0;
        agrees+=uint(agree);agreeW+=w;agreeD+=w*d;
    }
    bool valid=support>0,allConsistent=agrees>0&&agrees==support;
    depthOut[i]=!valid?0.0:(allConsistent?stableRatio(agreeD,max(agreeW,EPS)):depths[uint(winner)*params.count+i]);
    confidenceOut[i]=!valid?0.0:(allConsistent?min(agreeW,1.0):bestScore);
    validOut[i]=uint(valid);winning_sourceOut[i]=valid?winner:-1;
}
[shader("compute")]
[numthreads(256,1,1)]
void temporalDepthFuse(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.count)return;
    bool anyTrusted=false;float bestTrustScore=-1;uint trustWinner=0;
    for(uint s=0;s<params.sources;s++)
    {
        uint j=s*params.count+i;bool use=temporalUsable(s,i),trust=trusted[j]!=0;
        anyTrusted=anyTrusted||(use&&trust);
        float score=!use?0.0:(trust?1.0+confidences[j]:confidences[j]*.5);
        if(score>bestTrustScore){bestTrustScore=score;trustWinner=s;}
    }
    float bestCluster=-1;uint consensusWinner=0;
    if(!anyTrusted)
    {
        for(uint s=0;s<params.sources;s++)
        {
            float score=0,reference=depths[s*params.count+i];
            if(temporalUsable(s,i))
            {
                for(uint t=0;t<params.sources;t++)
                {
                    uint j=t*params.count+i;
                    if(temporalUsable(t,i)&&abs(depths[j]-reference)<=params.threshold*max(reference,EPS))score+=confidences[j];
                }
            }
            if(score>bestCluster){bestCluster=score;consensusWinner=s;}
        }
    }
    uint owner=anyTrusted?trustWinner:consensusWinner;
    float reference=depths[owner*params.count+i],weightSum=0,temporalW=0,numerator=0,bestWeight=-1,disagreement=0;
    int support=0,temporalSupport=0,dominant=0;bool trustedSupport=false;
    for(uint s=0;s<params.sources;s++)
    {
        uint j=s*params.count+i;bool use=temporalUsable(s,i);
        bool agree=use&&abs(depths[j]-reference)<=params.threshold*max(reference,EPS);
        float weight=agree?confidences[j]:0.0,safeDepth=agree?depths[j]:0.0;
        support+=int(agree);temporalSupport+=int(agree&&s>0);
        trustedSupport=trustedSupport||(agree&&trusted[j]!=0);
        weightSum+=weight;temporalW+=s>0?weight:0.0;numerator+=weight*safeDepth;
        if(weight>bestWeight){bestWeight=weight;dominant=int(s);}
        if(use)disagreement=max(disagreement,abs(depths[j]-reference));
    }
    float fused=stableRatio(numerator,max(weightSum,EPS)),variance=0;
    for(uint s=0;s<params.sources;s++)
    {
        uint j=s*params.count+i;
        bool agree=temporalUsable(s,i)&&abs(depths[j]-reference)<=params.threshold*max(reference,EPS);
        float residual=agree?depths[j]-fused:0.0,weight=agree?confidences[j]:0.0;
        variance+=weight*residual*residual;
    }
    bool valid=trustedSupport||(uint(support)>=params.minSupport&&uint(temporalSupport)>=params.minTemporalSupport);
    depthOut[i]=valid?fused:0.0;validOut[i]=uint(valid);confidenceOut[i]=valid?min(weightSum,1.0):0.0;
    support_countOut[i]=support;temporal_support_countOut[i]=temporalSupport;
    depth_varianceOut[i]=valid?stableRatio(variance,max(weightSum,EPS)):0.0;
    winning_sourceOut[i]=valid?dominant:-1;dominant_contributorOut[i]=valid?dominant:-1;
    conflict_owner_sourceOut[i]=valid?int(owner):-1;
    temporal_weight_fractionOut[i]=valid?stableRatio(temporalW,max(weightSum,EPS)):0.0;
    source_disagreementOut[i]=valid?disagreement:0.0;
}
