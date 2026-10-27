#define MATCH_GROUP_SIZE 64
float caspar_copysign(float mag, float sign_value)
{
    uint m = asuint(abs(mag));
    uint s = asuint(sign_value) & 0x80000000u;
    return asfloat(m | s);
}

float caspar_ldexp(float x, int exponent)
{
    return x * exp2((float)exponent);
}

float caspar_hypot(float a, float b)
{
    return length(float2(a, b));
}

float caspar_rhypot(float a, float b)
{
    return rsqrt(a * a + b * b);
}

float caspar_norm3(float a, float b, float c)
{
    return length(float3(a, b, c));
}

float caspar_rnorm3(float a, float b, float c)
{
    return rsqrt(a * a + b * b + c * c);
}

float caspar_norm4(float a, float b, float c, float d)
{
    return length(float4(a, b, c, d));
}

float caspar_rnorm4(float a, float b, float c, float d)
{
    return rsqrt(a * a + b * b + c * c + d * d);
}

// SymForce/Caspar DABSEG kernel: backproject
void caspar_backproject(float u, float v, float z, float fx, float fy, float cx, float cy, out float out_0, out float out_1, out float out_2)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    r0 = z;
    r1 = u;
    r2 = cx;
    r3 = -1.00000000000000000e+00;
    r2 = fma(r2, r3, r1);
    r2 = r0 * r2;
    r1 = fx;
    r1 = 1.0 / r1;
    r2 = r2 * r1;
    out_0 = r2;
    r2 = v;
    r1 = cy;
    r3 = fma(r1, r3, r2);
    r3 = r0 * r3;
    r1 = fy;
    r1 = 1.0 / r1;
    r3 = r3 * r1;
    out_1 = r3;
    out_2 = r0;
}

// SymForce/Caspar DABSEG kernel: transform
void caspar_transform(float x, float y, float z, float r00, float r01, float r02, float r10, float r11, float r12, float r20, float r21, float r22, float tx, float ty, float tz, out float out_0, out float out_1, out float out_2)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f;
    r0 = tx;
    r1 = z;
    r2 = r02;
    r2 = fma(r1, r2, r0);
    r0 = y;
    r3 = r01;
    r4 = x;
    r5 = r00;
    r2 = fma(r0, r3, r2);
    r2 = fma(r4, r5, r2);
    out_0 = r2;
    r2 = ty;
    r5 = r12;
    r5 = fma(r1, r5, r2);
    r2 = r11;
    r3 = r10;
    r5 = fma(r0, r2, r5);
    r5 = fma(r4, r3, r5);
    out_1 = r5;
    r5 = tz;
    r3 = r22;
    r3 = fma(r1, r3, r5);
    r1 = r21;
    r5 = r20;
    r3 = fma(r0, r1, r3);
    r3 = fma(r4, r5, r3);
    out_2 = r3;
}

// SymForce/Caspar DABSEG kernel: project
void caspar_project(float x, float y, float z, float fx, float fy, float cx, float cy, out float out_0, out float out_1)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f;
    r0 = cx;
    r1 = x;
    r2 = fx;
    r3 = r1 * r2;
    r4 = z;
    r4 = 1.0 / r4;
    r3 = fma(r4, r3, r0);
    out_0 = r3;
    r3 = cy;
    r0 = y;
    r5 = fy;
    r6 = r0 * r5;
    r6 = fma(r4, r6, r3);
    out_1 = r6;
}

// SymForce/Caspar DABSEG kernel: tsdf_fuse
void caspar_tsdf_fuse(float oldDistance, float oldWeight, float observation, float observationWeight, float maxWeight, out float out_0, out float out_1)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    r0 = observation;
    r1 = observationWeight;
    r2 = oldDistance;
    r3 = oldWeight;
    r2 = fma(r2, r3, r0 * r1);
    r0 = 9.99999999999999980e-13;
    r1 = r3 + r1;
    r0 = max(r0, r1);
    r0 = 1.0 / r0;
    r0 = r2 * r0;
    out_0 = r0;
    r0 = maxWeight;
    r1 = min(r0, r1);
    out_1 = r1;
}

// SymForce/Caspar DABSEG kernel: se3_coefficients
void caspar_se3_coefficients(float thetaSquared, out float out_0, out float out_1, out float out_2)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f, r9 = 0.0f;
    r0 = thetaSquared;
    r1 = 1.00000000000000002e-02;
    r2 = max(r0, r1);
    r3 = sqrt(r2);
    r4 = sin(r3);
    r5 = rsqrt(r2);
    r5 = r4 * r5;
    r6 = 1.00000000000000000e+00;
    r7 = -1.66666666666666657e-01;
    r7 = fma(r0, r7, r6);
    r8 = 8.33333333333333322e-03;
    r9 = r0 * r0;
    r7 = fma(r8, r9, r7);
    r5 = r0 < r1 ? r7 : r5;
    out_0 = r5;
    r5 = -1.00000000000000000e+00;
    r7 = cos(r3);
    r7 = fma(r5, r7, r6);
    r6 = 1.0 / r2;
    r6 = r7 * r6;
    r7 = 5.00000000000000000e-01;
    r8 = -4.16666666666666644e-02;
    r8 = fma(r0, r8, r7);
    r7 = 1.38888888888888873e-03;
    r8 = fma(r7, r9, r8);
    r6 = r0 < r1 ? r8 : r6;
    out_1 = r6;
    r5 = fma(r4, r5, r3);
    r4 = r2 * r2;
    r4 = r2 * r4;
    r4 = rsqrt(r4);
    r4 = r5 * r4;
    r5 = 1.66666666666666657e-01;
    r2 = -8.33333333333333322e-03;
    r2 = fma(r0, r2, r5);
    r5 = 1.98412698412698413e-04;
    r2 = fma(r5, r9, r2);
    r4 = r0 < r1 ? r2 : r4;
    out_2 = r4;
}

// SymForce/Caspar DABSEG kernel: twist_point
void caspar_twist_point(float wx, float wy, float wz, float vx, float vy, float vz, float a, float b, float c, float px, float py, float pz, float translationFlag, out float out_0, out float out_1, out float out_2)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f, r9 = 0.0f, r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f, r14 = 0.0f, r15 = 0.0f, r16 = 0.0f, r17 = 0.0f, r18 = 0.0f, r19 = 0.0f, r20 = 0.0f, r21 = 0.0f, r22 = 0.0f, r23 = 0.0f;
    r0 = px;
    r1 = a;
    r2 = wy;
    r3 = pz;
    r4 = py;
    r5 = wz;
    r6 = -1.00000000000000000e+00;
    r7 = r5 * r6;
    r8 = fma(r4, r7, r2 * r3);
    r9 = fma(r1, r8, r0);
    r10 = b;
    r11 = r2 * r0;
    r12 = wx;
    r11 = fma(r12, r4, r6 * r11);
    r13 = r12 * r3;
    r13 = fma(r5, r0, r6 * r13);
    r14 = fma(r13, r7, r2 * r11);
    r15 = translationFlag;
    r16 = vx;
    r17 = vz;
    r18 = vy;
    r19 = fma(r18, r7, r2 * r17);
    r20 = fma(r10, r19, r16);
    r21 = c;
    r22 = r2 * r16;
    r22 = fma(r12, r18, r6 * r22);
    r23 = r12 * r17;
    r23 = fma(r6, r23, r5 * r16);
    r7 = fma(r23, r7, r2 * r22);
    r20 = fma(r21, r7, r20);
    r9 = fma(r10, r14, r9);
    r9 = fma(r15, r20, r9);
    out_0 = r9;
    r4 = fma(r1, r13, r4);
    r9 = r12 * r6;
    r9 = fma(r11, r9, r5 * r8);
    r18 = fma(r10, r23, r18);
    r20 = r12 * r6;
    r20 = fma(r22, r20, r5 * r19);
    r18 = fma(r21, r20, r18);
    r4 = fma(r10, r9, r4);
    r4 = fma(r15, r18, r4);
    out_1 = r4;
    r11 = fma(r1, r11, r3);
    r1 = r2 * r6;
    r13 = fma(r12, r13, r8 * r1);
    r22 = fma(r10, r22, r17);
    r1 = r2 * r6;
    r23 = fma(r12, r23, r19 * r1);
    r22 = fma(r21, r23, r22);
    r11 = fma(r10, r13, r11);
    r11 = fma(r15, r22, r11);
    out_2 = r11;
}

// Original permissive FAST-9/oriented-BRIEF implementation (not OpenCV ORB).
struct FeatureParams {
 uint width,height,fullWidth,fullHeight;float fx,fy,cx,cy;uint level;float scale;
 uint gridX,gridY,quota,maxFeatures;float fastThreshold,ratio;uint maxHamming;
};
ConstantBuffer<FeatureParams> cfg;
StructuredBuffer<float> rgb;
StructuredBuffer<float> grayInput;
RWStructuredBuffer<float> grayOutput;
RWStructuredBuffer<float> fastScores;
RWStructuredBuffer<float> harrisScores;
StructuredBuffer<float> depth;
StructuredBuffer<float4> pattern;
RWStructuredBuffer<uint> featureCount;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<float4> landmarks;
RWStructuredBuffer<uint> descriptors;
StructuredBuffer<uint> sourceDescriptors,targetDescriptors;
StructuredBuffer<uint> sourceCount,targetCount;
RWStructuredBuffer<uint> forwardMatches,reverseMatches;
RWStructuredBuffer<uint> matches,matchCount;
static const int2 circle[16] = {int2(0,-3),int2(1,-3),int2(2,-2),int2(3,-1),int2(3,0),int2(3,1),int2(2,2),int2(1,3),int2(0,3),int2(-1,3),int2(-2,2),int2(-3,1),int2(-3,0),int2(-3,-1),int2(-2,-2),int2(-1,-3)};
float sampleGray(int2 p) {p=clamp(p,int2(0),int2(cfg.width-1,cfg.height-1));return grayInput[uint(p.y)*cfg.width+uint(p.x)];}
[shader("compute")][numthreads(16,16,1)]
void rgbToGray(uint3 tid:SV_DispatchThreadID) {
 if(tid.x>=cfg.width||tid.y>=cfg.height)return;uint i=tid.y*cfg.width+tid.x;
 grayOutput[i]=.299*rgb[i*3]+.587*rgb[i*3+1]+.114*rgb[i*3+2];
}
[shader("compute")][numthreads(16,16,1)]
void downsampleGray(uint3 tid:SV_DispatchThreadID) {
 if(tid.x>=cfg.width||tid.y>=cfg.height)return;
 uint w=cfg.width*2,h=cfg.height*2; // Inputs require dimensions divisible by pyramid scale.
 float sum=0;static const float k[5]={1,4,6,4,1};
 for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++) {
  int2 p=clamp(int2(tid.xy)*2+int2(x,y),int2(0),int2(w-1,h-1));
  sum+=k[x+2]*k[y+2]*grayInput[uint(p.y)*w+uint(p.x)];
 }
 grayOutput[tid.y*cfg.width+tid.x]=sum/256;
}
[shader("compute")][numthreads(1,1,1)]
void resetFeatures() {featureCount[0]=0;}
[shader("compute")][numthreads(1,1,1)]
void resetMatchCount() {matchCount[0]=0;}
[shader("compute")][numthreads(16,16,1)]
void fastDetect(uint3 tid:SV_DispatchThreadID) {
 if(tid.x>=cfg.width||tid.y>=cfg.height)return;uint i=tid.y*cfg.width+tid.x;
 fastScores[i]=0;if(tid.x<19||tid.y<19||tid.x+19>=cfg.width||tid.y+19>=cfg.height)return;
 float center=grayInput[i];float delta[16];
 for(uint k=0;k<16;k++)delta[k]=sampleGray(int2(tid.xy)+circle[k])-center;
 float score=0;
 for(uint start=0;start<16;start++) {
  float bright=255,dark=255;
  for(uint n=0;n<9;n++) {float d=delta[(start+n)&15];bright=min(bright,d);dark=min(dark,-d);}
  score=max(score,max(bright,dark));
 }
 fastScores[i]=score>cfg.fastThreshold?score:0;
}
[shader("compute")][numthreads(16,16,1)]
void nonmaxHarris(uint3 tid:SV_DispatchThreadID) {
 if(tid.x>=cfg.width||tid.y>=cfg.height)return;uint i=tid.y*cfg.width+tid.x;
 harrisScores[i]=0;float score=fastScores[i];if(score<=0)return;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
  uint j=uint(int(tid.y)+y)*cfg.width+uint(int(tid.x)+x);
  if(fastScores[j]>score||(fastScores[j]==score&&j<i))return;
 }
 float xx=0,xy=0,yy=0;
 for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++) {
  int2 p=int2(tid.xy)+int2(x,y);
  float dx=(sampleGray(p+int2(1,0))-sampleGray(p-int2(1,0)))*.5;
  float dy=(sampleGray(p+int2(0,1))-sampleGray(p-int2(0,1)))*.5;
  xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;
 }
 harrisScores[i]=max(0.,xx*yy-xy*xy-.04*(xx+yy)*(xx+yy));
}
groupshared float cellBest[64];
groupshared uint cellIndex[64];
groupshared uint selectedIndex[4];
groupshared uint compactBase;
groupshared uint selectedCount;
[shader("compute")][numthreads(64,1,1)]
void selectCells(uint3 gid:SV_GroupID,uint lane:SV_GroupIndex) {
 uint x0=gid.x*cfg.width/cfg.gridX,x1=(gid.x+1)*cfg.width/cfg.gridX;
 uint y0=gid.y*cfg.height/cfg.gridY,y1=(gid.y+1)*cfg.height/cfg.gridY;
 uint width=x1-x0,area=width*(y1-y0);
 float localScore[4]={0,0,0,0};uint localIndex[4]={0xffffffff,0xffffffff,0xffffffff,0xffffffff};
 for(uint k=lane;k<area;k+=64) {
  uint x=x0+k%width,y=y0+k/width,i=y*cfg.width+x;float score=harrisScores[i];if(score<=0)continue;
  float2 original=(float2(x,y)+.5)*cfg.scale-.5;
  uint2 pixel=uint2(floor(original+.5));float z=depth[pixel.y*cfg.fullWidth+pixel.x];if(!isfinite(z)||z<=0)continue;
  for(uint rank=0;rank<4;rank++) {
   if(score>localScore[rank]||(score==localScore[rank]&&i<localIndex[rank])) {
    for(int shift=3;shift>int(rank);shift--){localScore[shift]=localScore[shift-1];localIndex[shift]=localIndex[shift-1];}
    localScore[rank]=score;localIndex[rank]=i;break;
   }
  }
 }
 if(lane==0)selectedCount=0;
 GroupMemoryBarrierWithGroupSync();
 for(uint rank=0;rank<cfg.quota;rank++) {
  cellBest[lane]=localScore[0];cellIndex[lane]=localIndex[0];GroupMemoryBarrierWithGroupSync();
  for(uint stride=32;stride>0;stride>>=1) {
   if(lane<stride && (cellBest[lane+stride]>cellBest[lane]||(cellBest[lane+stride]==cellBest[lane]&&cellIndex[lane+stride]<cellIndex[lane]))) {cellBest[lane]=cellBest[lane+stride];cellIndex[lane]=cellIndex[lane+stride];}
   GroupMemoryBarrierWithGroupSync();
  }
  if(lane==0 && cellBest[0]>0) {selectedIndex[selectedCount]=cellIndex[0];selectedCount++;}
  uint winner=cellIndex[0];GroupMemoryBarrierWithGroupSync();
  if(winner==localIndex[0]) {
   for(uint shift=0;shift<3;shift++){localScore[shift]=localScore[shift+1];localIndex[shift]=localIndex[shift+1];}
   localScore[3]=0;localIndex[3]=0xffffffff;
  }
  GroupMemoryBarrierWithGroupSync();
 }
 if(lane==0)InterlockedAdd(featureCount[0],selectedCount,compactBase);
 GroupMemoryBarrierWithGroupSync();
 if(lane<selectedCount && compactBase+lane<cfg.maxFeatures) {
  uint i=selectedIndex[lane];float2 pixel=(float2(i%cfg.width,i/cfg.width)+.5)*cfg.scale-.5;
  uint2 p=uint2(floor(pixel+.5));float z=depth[p.y*cfg.fullWidth+p.x];float a,b,c;
  caspar_backproject(pixel.x,pixel.y,z,cfg.fx,cfg.fy,cfg.cx,cfg.cy,a,b,c);
  positions[compactBase+lane]=float4(pixel,z,float(cfg.level));landmarks[compactBase+lane]=float4(a,b,c,1);
 }
}
[shader("compute")][numthreads(64,1,1)]
void describe(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=featureCount[0]||i>=cfg.maxFeatures)return;
 float4 feature=positions[i];if(uint(feature.w)!=cfg.level)return;
 int2 center=int2(floor((feature.xy+.5)/cfg.scale-.5+.5));
 float mx=0,my=0;
 for(int y=-15;y<=15;y++)for(int x=-15;x<=15;x++)if(x*x+y*y<=225){float value=sampleGray(center+int2(x,y));mx+=float(x)*value;my+=float(y)*value;}
 float angle=atan2(my,mx),s=sin(angle),c=cos(angle);
 for(uint word=0;word<8;word++) {
  uint bits=0;
  for(uint bit=0;bit<32;bit++) {
   float4 offsets=pattern[word*32+bit];
   int2 a=int2(floor(float2(c*offsets.x-s*offsets.y,s*offsets.x+c*offsets.y)+.5));
   int2 b=int2(floor(float2(c*offsets.z-s*offsets.w,s*offsets.z+c*offsets.w)+.5));
   if(sampleGray(center+a)<sampleGray(center+b))bits|=1u<<bit;
  }
  descriptors[i*8+word]=bits;
 }
}
#ifndef MATCH_GROUP_SIZE
#define MATCH_GROUP_SIZE 64
#endif
groupshared uint targetTile[MATCH_GROUP_SIZE*8];
[shader("compute")][numthreads(MATCH_GROUP_SIZE,1,1)]
void matchForward(uint3 tid:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
 uint source=tid.x,n=min(sourceCount[0],cfg.maxFeatures),m=min(targetCount[0],cfg.maxFeatures);
 uint sourceWords[8];for(uint k=0;k<8;k++)sourceWords[k]=source<n?sourceDescriptors[source*8+k]:0;
 uint best=257,second=257,index=0xffffffff;
 for(uint begin=0;begin<m;begin+=MATCH_GROUP_SIZE) {
  for(uint k=0;k<8;k++)targetTile[lane*8+k]=begin+lane<m?targetDescriptors[(begin+lane)*8+k]:0;
  GroupMemoryBarrierWithGroupSync();
  if(source<n)for(uint j=0;j<min(uint(MATCH_GROUP_SIZE),m-begin);j++) {
   uint distance=0;for(uint k=0;k<8;k++)distance+=countbits(sourceWords[k]^targetTile[j*8+k]);
   if(distance<best){second=best;best=distance;index=begin+j;}else if(distance<second)second=distance;
  }
  GroupMemoryBarrierWithGroupSync();
 }
 if(source<cfg.maxFeatures)forwardMatches[source]=source<n && m>=2 && best<=cfg.maxHamming && float(best)<cfg.ratio*float(second)?index:0xffffffff;
}
[shader("compute")][numthreads(MATCH_GROUP_SIZE,1,1)]
void matchReverse(uint3 tid:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
 uint target=tid.x,n=min(targetCount[0],cfg.maxFeatures),m=min(sourceCount[0],cfg.maxFeatures);
 uint words[8];for(uint k=0;k<8;k++)words[k]=target<n?targetDescriptors[target*8+k]:0;
 uint best=257,index=0xffffffff;
 for(uint begin=0;begin<m;begin+=MATCH_GROUP_SIZE) {
  for(uint k=0;k<8;k++)targetTile[lane*8+k]=begin+lane<m?sourceDescriptors[(begin+lane)*8+k]:0;
  GroupMemoryBarrierWithGroupSync();
  if(target<n)for(uint j=0;j<min(uint(MATCH_GROUP_SIZE),m-begin);j++) {
   uint distance=0;for(uint k=0;k<8;k++)distance+=countbits(words[k]^targetTile[j*8+k]);
   if(distance<best){best=distance;index=begin+j;}
  }
  GroupMemoryBarrierWithGroupSync();
 }
 if(target<cfg.maxFeatures)reverseMatches[target]=index;
}
[shader("compute")][numthreads(64,1,1)]
void filterMatches(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=cfg.maxFeatures)return;uint target=forwardMatches[i];
 bool valid=i<sourceCount[0]&&target!=0xffffffff;
 if(valid)valid=reverseMatches[target]==i;
 matches[i]=valid?target:0xffffffff;
 if(valid){uint ignored;InterlockedAdd(matchCount[0],1,ignored);}
}

