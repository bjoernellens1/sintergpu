#ifndef DENSE_GROUP_SIZE
#define DENSE_GROUP_SIZE 64
#endif
#ifndef DENSE_USE_WAVE
#define DENSE_USE_WAVE 1
#endif
#ifndef DENSE_MIN_WAVE_SIZE
#define DENSE_MIN_WAVE_SIZE 1
#endif
struct PhotoParams {float weight;float huber;};
ConstantBuffer<PhotoParams> photoParams;
StructuredBuffer<float> sourceIntensity;
StructuredBuffer<float> referenceIntensity;
StructuredBuffer<float> rgbImage;
RWStructuredBuffer<float> intensityOutput;
RWStructuredBuffer<float> outputTerms;
#if DENSE_USE_WAVE
groupshared float denseTerms[43][(DENSE_GROUP_SIZE+DENSE_MIN_WAVE_SIZE-1)/DENSE_MIN_WAVE_SIZE];
#else
groupshared float denseTerms[43][DENSE_GROUP_SIZE];
#endif
#if !DENSE_USE_WAVE
groupshared uint denseValid[DENSE_GROUP_SIZE];
#endif
[shader("compute")][numthreads(256,1,1)]
void rgbIntensity(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=mapParams.width*mapParams.height)return;
 intensityOutput[i]=dot(float3(rgbImage[i*3],rgbImage[i*3+1],rgbImage[i*3+2]),float3(.299,.587,.114))/255.;
}
[shader("compute")][numthreads(16,16,1)]
void downsampleIntensity(uint3 tid:SV_DispatchThreadID) {
 uint x=tid.x,y=tid.y;if(x>=mapParams.width||y>=mapParams.height)return;uint w=mapParams.width*2;
 float a=sourceIntensity[y*2*w+x*2],b=sourceIntensity[y*2*w+x*2+1],c=sourceIntensity[(y*2+1)*w+x*2],d=sourceIntensity[(y*2+1)*w+x*2+1];
 intensityOutput[y*mapParams.width+x]=min(min(a,b),min(c,d))<0?-1:(a+b+c+d)*.25;
}
float intensityAt(float2 uv) {
 int2 p=int2(floor(uv));float2 f=uv-float2(p);uint i=uint(p.y)*trackParams.width+uint(p.x);
 float a=referenceIntensity[i],b=referenceIntensity[i+1],c=referenceIntensity[i+trackParams.width],d=referenceIntensity[i+trackParams.width+1];
 if(min(min(a,b),min(c,d))<0)return -1;
 return lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y);
}
[shader("compute")][numthreads(DENSE_GROUP_SIZE,1,1)]
void fusedAssociate(uint3 tid:SV_DispatchThreadID,uint3 lane:SV_GroupThreadID,uint3 group:SV_GroupID) {
 uint i=tid.x;bool valid=false;float r=0,pr=0;float j[6],pj[6];
 for(uint c=0;c<6;c++){j[c]=0;pj[c]=0;}
 if(i<trackParams.count && trackingStatus[1]==0) {
  float4 source=sourceVertices[i],sn=sourceNormals[i];float3 p=rotatePose(source.xyz)+float3(pose[3],pose[7],pose[11]);
  if(source.w>0 && sn.w>0 && p.z>0) {
   float u,v;caspar_project(p.x,p.y,p.z,trackParams.fx,trackParams.fy,trackParams.cx,trackParams.cy,u,v);float2 uv=float2(u,v);int2 pixel=int2(round(uv));
   if(all(pixel>=0)&&pixel.x<int(trackParams.width)&&pixel.y<int(trackParams.height)) {
    uint k=uint(pixel.y)*trackParams.width+uint(pixel.x);float4 q=referenceVertices[k],n=referenceNormals[k];
    valid=q.w>0&&n.w>0&&length(p-q.xyz)<trackParams.distance&&dot(rotatePose(sn.xyz),n.xyz)>trackParams.normalCos;
    if(valid) {
     caspar_point_plane(p.x,p.y,p.z,q.x,q.y,q.z,n.x,n.y,n.z,r,j[0],j[1],j[2],j[3],j[4],j[5]);
     float weight=sqrt(min(1.,trackParams.huber/max(abs(r),1e-20)));r*=weight;for(uint c=0;c<6;c++)j[c]*=weight;
     if(photoParams.weight>0 && all(uv>=float2(2)) && u<float(trackParams.width)-3 && v<float(trackParams.height)-3) {
      float ti=intensityAt(uv),xp=intensityAt(uv+float2(1,0)),xm=intensityAt(uv-float2(1,0)),yp=intensityAt(uv+float2(0,1)),ym=intensityAt(uv-float2(0,1));
      if(ti>=0&&xp>=0&&xm>=0&&yp>=0&&ym>=0) {
       float gx=(xp-xm)*.5,gy=(yp-ym)*.5;
       caspar_photo_factor(p.x,p.y,p.z,trackParams.fx,trackParams.fy,gx,gy,ti,sourceIntensity[i],pr,pj[0],pj[1],pj[2],pj[3],pj[4],pj[5]);
       float pw=photoParams.weight*sqrt(min(1.,photoParams.huber/max(abs(pr),1e-20)));pr*=pw;for(uint c=0;c<6;c++)pj[c]*=pw;
      }
     }
    }
   }
  }
 }
#if DENSE_USE_WAVE
 uint count=WaveActiveCountBits(valid);if(WaveIsFirstLane())InterlockedAdd(trackingStatus[0],count);
#else
 denseValid[lane.x]=valid?1:0;
#endif
 for(uint t=0;t<43;t++) {
  float value=t<36?j[t/6]*j[t%6]+pj[t/6]*pj[t%6]:(t<42?j[t-36]*r+pj[t-36]*pr:r*r+pr*pr);
#if DENSE_USE_WAVE
  float sum=WaveActiveSum(value);if(WaveIsFirstLane())denseTerms[t][lane.x/WaveGetLaneCount()]=sum;
#else
  denseTerms[t][lane.x]=value;
#endif
 }
 GroupMemoryBarrierWithGroupSync();
#if DENSE_USE_WAVE
 uint waves=(DENSE_GROUP_SIZE+WaveGetLaneCount()-1)/WaveGetLaneCount();
 for(uint t=lane.x;t<43;t+=DENSE_GROUP_SIZE){float sum=0;for(uint w=0;w<waves;w++)sum+=denseTerms[t][w];outputTerms[group.x*43+t]=sum;}
#else
 for(uint stride=DENSE_GROUP_SIZE/2;stride>0;stride>>=1){if(lane.x<stride){for(uint t=0;t<43;t++)denseTerms[t][lane.x]+=denseTerms[t][lane.x+stride];denseValid[lane.x]+=denseValid[lane.x+stride];}GroupMemoryBarrierWithGroupSync();}
 if(lane.x==0){InterlockedAdd(trackingStatus[0],denseValid[0]);for(uint t=0;t<43;t++)outputTerms[group.x*43+t]=denseTerms[t][0];}
#endif
}
