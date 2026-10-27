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
        float spread=(maximum-minimum)/max(selected,1.1920928955078125e-7);
        fill=spread<=params.maxRelativeSpread;
        fillConfidence=(sumConfidence/float(support))*(float(support)/9.0);
    }
    bool outValid=centerValid||fill,propagated=outValid&&!direct;
    depthOut[i]=centerValid?depth[i]:(fill?selected:0.0);
    validOut[i]=uint(outValid);confidenceOut[i]=centerValid?confidence[i]:(fill?fillConfidence:0.0);
    direct_validOut[i]=uint(direct);propagated_validOut[i]=uint(propagated);provenanceOut[i]=uint(direct)+2*uint(propagated);supportOut[i]=int(support);
}
