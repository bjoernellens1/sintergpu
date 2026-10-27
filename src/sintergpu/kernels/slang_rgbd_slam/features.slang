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
