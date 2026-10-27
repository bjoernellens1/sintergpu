// Dispatch/control is handwritten; calibrated residuals and all derivative
// arithmetic below execute actual SymForce/Caspar DABSEG-generated functions.
struct ReprojectionParams { uint count; float fx; float fy; float cx; float cy; float huber; uint minMatches; };
ConstantBuffer<ReprojectionParams> reprojectionParams;
StructuredBuffer<float> worldPoints;
StructuredBuffer<float> observations;
StructuredBuffer<uint> validMask;
RWStructuredBuffer<float> pose;
RWStructuredBuffer<float> outputTerms;
groupshared float terms[43][64];
RWStructuredBuffer<uint> stats;
RWStructuredBuffer<float> cost;
StructuredBuffer<float> delta;
StructuredBuffer<uint> solveStatus;
StructuredBuffer<float> normalTerms;

[shader("compute")][numthreads(1,1,1)]
void resetReprojection() { stats[0]=0;stats[1]=0;cost[0]=0; }
[shader("compute")][numthreads(1,1,1)]
void resetReprojectionCount() {stats[0]=0;}

[shader("compute")][numthreads(64,1,1)]
void buildReprojection(uint3 tid:SV_DispatchThreadID,uint3 lane:SV_GroupThreadID,uint3 group:SV_GroupID) {
 uint i=tid.x;
 float r[2];float j[12];float l[6];
 r[0]=0;r[1]=0;[unroll]for(uint k=0;k<12;k++)j[k]=0;
 bool valid=false;
 if(i<reprojectionParams.count)valid=validMask[i]!=0 && stats[1]==0;
 if(valid) {
  float3 world=float3(worldPoints[i*3],worldPoints[i*3+1],worldPoints[i*3+2]);
  float2 measured=float2(observations[i*2],observations[i*2+1]);
  valid=all(isfinite(world))&&all(isfinite(measured));
  if(valid) {
   float x,y,z;
   caspar_transform(world.x,world.y,world.z,pose[0],pose[1],pose[2],pose[4],pose[5],pose[6],pose[8],pose[9],pose[10],pose[3],pose[7],pose[11],x,y,z);
   valid=z>1e-6&&all(isfinite(float3(x,y,z)));
   if(valid) {
    caspar_reproject(x,y,z,reprojectionParams.fx,reprojectionParams.fy,reprojectionParams.cx,reprojectionParams.cy,measured.x,measured.y,
       pose[0],pose[1],pose[2],pose[4],pose[5],pose[6],pose[8],pose[9],pose[10],
       r[0],r[1],j[0],j[1],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11],l[0],l[1],l[2],l[3],l[4],l[5]);
    valid=isfinite(r[0])&&isfinite(r[1]);
    [unroll]for(uint k=0;k<12;k++)valid=valid&&isfinite(j[k]);
    if(valid){float norm=length(float2(r[0],r[1]));float w=sqrt(min(1.,reprojectionParams.huber/max(norm,1e-12)));r[0]*=w;r[1]*=w;[unroll]for(uint k=0;k<12;k++)j[k]*=w;}
   }
  }
 }
 if(!valid){r[0]=0;r[1]=0;[unroll]for(uint k=0;k<12;k++)j[k]=0;}
 uint count=WaveActiveCountBits(valid);if(WaveIsFirstLane())InterlockedAdd(stats[0],count);
 [unroll]for(uint t=0;t<43;t++) {
  float value;
  if(t<36)value=j[t/6]*j[t%6]+j[6+t/6]*j[6+t%6];
  else if(t<42)value=j[t-36]*r[0]+j[t-30]*r[1];
  else value=r[0]*r[0]+r[1]*r[1];
  float sum=WaveActiveSum(value);terms[t][lane.x]=WaveIsFirstLane()?sum:0;
 }
 GroupMemoryBarrierWithGroupSync();
 for(uint stride=32;stride>0;stride>>=1) {
  if(lane.x<stride)[unroll]for(uint t=0;t<43;t++)terms[t][lane.x]+=terms[t][lane.x+stride];
  GroupMemoryBarrierWithGroupSync();
 }
 if(lane.x==0)[unroll]for(uint t=0;t<43;t++)outputTerms[group.x*43+t]=terms[t][0];
}

[shader("compute")][numthreads(1,1,1)]
void updateReprojection() {
 cost[0]=normalTerms[42];
 if(stats[1]!=0)return;
 if(stats[0]<reprojectionParams.minMatches||solveStatus[0]!=0){stats[1]=1;return;}
 float3 omega=float3(delta[0],delta[1],delta[2]);float3 velocity=float3(delta[3],delta[4],delta[5]);
 float a,b,c;caspar_se3_coefficients(dot(omega,omega),a,b,c);
 float updated[12];bool finitePose=true;
 [unroll]for(uint col=0;col<4;col++) {
  float x,y,z;caspar_twist_point(omega.x,omega.y,omega.z,velocity.x,velocity.y,velocity.z,a,b,c,pose[col],pose[4+col],pose[8+col],col==3?1.:0.,x,y,z);
  updated[col]=x;updated[4+col]=y;updated[8+col]=z;finitePose=finitePose&&all(isfinite(float3(x,y,z)));
 }
 if(!finitePose){stats[1]=1;return;}
 [unroll]for(uint k=0;k<12;k++)pose[k]=updated[k];
}

[shader("compute")][numthreads(1,1,1)]
void evaluateReprojection() {
 cost[0]=normalTerms[42];
 if(stats[0]<reprojectionParams.minMatches||!isfinite(cost[0]))stats[1]=1;
}
