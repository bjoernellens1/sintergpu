// SPDX-License-Identifier: Apache-2.0
// Float32 B1HW depth/confidence, BCHW RGB, scalar32 uint logical/provenance.
struct ResamplingParams {
    uint batches; uint height; uint width; uint outputHeight; uint outputWidth;
    uint channels; uint firstStep; uint useDirect; uint useConfidence; uint minSupport;
    float scaleX; float scaleY; float maxRGBDifference; float maxRelativeSpread;
};
ConstantBuffer<ResamplingParams> params;
StructuredBuffer<float> depth;
StructuredBuffer<uint> valid;
StructuredBuffer<float> confidence;
StructuredBuffer<uint> direct_valid;
StructuredBuffer<float> rgb;
RWStructuredBuffer<float> depthOut;
RWStructuredBuffer<uint> validOut;
RWStructuredBuffer<float> confidenceOut;
RWStructuredBuffer<uint> direct_validOut;
RWStructuredBuffer<uint> propagated_validOut;
RWStructuredBuffer<uint> provenanceOut;
RWStructuredBuffer<int> supportOut;
bool usableDepth(uint i) { return valid[i]!=0&&isfinite(depth[i])&&depth[i]>0; }
// Exact uint32 limbs avoid requiring shaderFloat64 or shaderInt64. Inputs to the
// product are <=24 and <=25 bits, so its <=49-bit result fits two limbs.
uint2 multiplySignificands(uint a,uint b)
{
    uint lowProduct=(a&65535)*(b&65535);
    uint cross=(a>>16)*(b&65535)+(a&65535)*(b>>16);
    uint low=lowProduct+(cross<<16);
    uint high=(a>>16)*(b>>16)+(cross>>16)+uint(low<lowProduct);
    return uint2(low,high);
}
uint2 normalizeLimbs(uint2 value,uint highest)
{
    uint shift=63-highest;
    if(shift==0)return value;
    if(shift>=32)return uint2(0,value.x<<(shift-32));
    return uint2(value.x<<shift,(value.y<<shift)|(value.x>>(32-shift)));
}
// Compare roundToNearestEven(numerator/denominator) <= threshold, as float32
// Torch does. A raw cross product would compare the unrounded real quotient.
// Instead compare against the exact midpoint between threshold and its next
// float; ties choose threshold only when its significand is even.
// The caller supplies nonnegative numerator, positive finite denominator and a
// nonnegative threshold. No approximate floating-point divide decides the mask.
bool roundedRatioWithin(float numerator,float denominator,float threshold)
{
    if(isnan(threshold)||threshold<0)return false;
    if(isinf(threshold))return true;
    if(numerator==0)return true;
    if(!isfinite(numerator))return false;
    uint a=asuint(numerator),b=asuint(denominator),t=asuint(threshold);
    uint ae=(a>>23)&255,be=(b>>23)&255,te=(t>>23)&255;
    uint am=(a&0x7fffff)|(ae!=0?0x800000:0);
    uint bm=(b&0x7fffff)|(be!=0?0x800000:0);
    uint tm=(t&0x7fffff)|(te!=0?0x800000:0);
    int ax=ae!=0?int(ae)-150:-149;
    int bx=be!=0?int(be)-150:-149;
    int midpointExponent=te!=0?int(te)-151:-150;
    uint2 product=multiplySignificands(bm,2*tm+1);
    uint aTop=uint(firstbithigh(am));
    uint pTop=product.y!=0?32+uint(firstbithigh(product.y)):uint(firstbithigh(product.x));
    int aExponent=ax+int(aTop),pExponent=bx+midpointExponent+int(pTop);
    if(aExponent!=pExponent)return aExponent<pExponent;
    uint2 left=normalizeLimbs(uint2(am,0),aTop),right=normalizeLimbs(product,pTop);
    if(left.y!=right.y)return left.y<right.y;
    if(left.x!=right.x)return left.x<right.x;
    return (t&1)==0;
}

[shader("compute")]
[numthreads(256,1,1)]
void initializeDepth(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.batches*params.height*params.width)return;
    bool use=usableDepth(i);
    depthOut[i]=use?depth[i]:0.0;validOut[i]=uint(use);
    confidenceOut[i]=use?(params.useConfidence!=0?confidence[i]:1.0):0.0;
}
[shader("compute")]
[numthreads(256,1,1)]
void depthUpsample(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.outputHeight*params.outputWidth;
    if(i>=params.batches*plane)return;
    uint b=i/plane,x=i%params.outputWidth,y=(i/params.outputWidth)%params.outputHeight;
    // Host supplies rounded float32 scale; no approximate GPU reciprocal.
    uint sx=min(uint(floor(float(x)*params.scaleX)),params.width-1);
    uint sy=min(uint(floor(float(y)*params.scaleY)),params.height-1);
    uint j=b*params.height*params.width+sy*params.width+sx;bool use=usableDepth(j);
    depthOut[i]=use?depth[j]:0.0;validOut[i]=uint(use);confidenceOut[i]=use?confidence[j]:0.0;
    direct_validOut[i]=0;propagated_validOut[i]=uint(use);provenanceOut[i]=use?2:0;
}
[shader("compute")]
[numthreads(256,1,1)]
void labelDirectDepth(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x;if(i>=params.batches*params.height*params.width)return;
    bool use=valid[i]!=0;direct_validOut[i]=uint(use);propagated_validOut[i]=0;provenanceOut[i]=uint(use);
}
[shader("compute")]
[numthreads(256,1,1)]
void densifyDepth(uint3 tid : SV_DispatchThreadID)
{
    uint i=tid.x,plane=params.height*params.width;if(i>=params.batches*plane)return;
    uint b=i/plane;int x=int(i%params.width),y=int((i/params.width)%params.height);
    bool centerValid=usableDepth(i);
    bool direct=centerValid;
    if(params.firstStep!=0){if(params.useDirect!=0)direct=centerValid&&(direct_valid[i]!=0);}
    else direct=direct_valid[i]!=0;
    float values[9];for(uint k=0;k<9;k++)values[k]=asfloat(0x7f800000);
    uint support=0;float minimum=asfloat(0x7f800000),maximum=0,sumConfidence=0;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)
    {
        int yy=y+dy,xx=x+dx;
        if(yy<0||xx<0||yy>=int(params.height)||xx>=int(params.width))continue;
        uint j=b*plane+uint(yy)*params.width+uint(xx);
        if(!usableDepth(j))continue;
        bool colorAgree=true;
        for(uint c=0;c<params.channels;c++)
        {
            uint base=(b*params.channels+c)*plane;
            float difference=abs(rgb[base+uint(yy)*params.width+uint(xx)]-rgb[base+uint(y)*params.width+uint(x)]);
            colorAgree=colorAgree&&(difference<=params.maxRGBDifference);
        }
        if(!colorAgree)continue;
        float d=depth[j];values[support]=d;support++;minimum=min(minimum,d);maximum=max(maximum,d);sumConfidence+=confidence[j];
    }
    bool fill=false;float selected=0,fillConfidence=0;
    if(!centerValid&&support>=params.minSupport)
    {
        // Nine-element local sort; infinities represent absent neighbors.
        for(uint pass=0;pass<8;pass++)for(uint k=0;k<8-pass;k++)
            if(values[k]>values[k+1]){float tmp=values[k];values[k]=values[k+1];values[k+1]=tmp;}
        selected=values[(support-1)/2];
        fill=roundedRatioWithin(maximum-minimum,max(selected,1.1920928955078125e-7),params.maxRelativeSpread);
        fillConfidence=(sumConfidence/float(support))*(float(support)/9.0);
    }
    bool outValid=centerValid||fill,propagated=outValid&&!direct;
    depthOut[i]=centerValid?depth[i]:(fill?selected:0.0);
    validOut[i]=uint(outValid);confidenceOut[i]=centerValid?confidence[i]:(fill?fillConfidence:0.0);
    direct_validOut[i]=uint(direct);propagated_validOut[i]=uint(propagated);provenanceOut[i]=uint(direct)+2*uint(propagated);supportOut[i]=int(support);
}
