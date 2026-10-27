// Coupled RGB-D bundle adjustment: anchored poses and per-source inverse depth.
struct DenseParams {
 uint width,height;float fx,fy,cx,cy;uint frames,activeFrames,dofs,terms,count;
 float depthWeight,geoWeight,photoWeight,damping,distance,depthJump,minFactorRatio;
};
ConstantBuffer<DenseParams> cfg;
StructuredBuffer<float> depths,initialDepths,rgb;
RWStructuredBuffer<float> gray;
RWStructuredBuffer<float4> normals;
RWStructuredBuffer<float> poses,candidatePoses,inverseDepth,hpd,hdd,bd;
StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> outputTerms,delta;
RWStructuredBuffer<uint> status;
float3 ray(uint x,uint y) {float a,b,c;caspar_backproject(float(x),float(y),1,cfg.fx,cfg.fy,cfg.cx,cfg.cy,a,b,c);return float3(a,b,c);}
float3 cameraVertex(uint frame,uint x,uint y) {return ray(x,y)*depths[frame*cfg.width*cfg.height+y*cfg.width+x];}
float3 toWorld(uint frame,float3 p) {
 uint b=frame*16;float x,y,z;
 caspar_transform(p.x,p.y,p.z,poses[b],poses[b+1],poses[b+2],poses[b+4],poses[b+5],poses[b+6],poses[b+8],poses[b+9],poses[b+10],poses[b+3],poses[b+7],poses[b+11],x,y,z);
 return float3(x,y,z);
}
float3 toCamera(uint frame,float3 p) {
 uint b=frame*16;p-=float3(poses[b+3],poses[b+7],poses[b+11]);float x,y,z;
 caspar_transform(p.x,p.y,p.z,poses[b],poses[b+4],poses[b+8],poses[b+1],poses[b+5],poses[b+9],poses[b+2],poses[b+6],poses[b+10],0,0,0,x,y,z);
 return float3(x,y,z);
}
[shader("compute")][numthreads(64,1,1)]
void initializeDense(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=cfg.count)return;uint image=cfg.width*cfg.height;
 uint frame=i/image,pixel=i%image,x=pixel%cfg.width,y=pixel/cfg.width;
 float z=depths[i],seed=initialDepths[i],inverse=seed>0?1/seed:0;inverseDepth[i]=frame<cfg.activeFrames && z>0 && isfinite(1/z) && isfinite(inverse)?inverse:0;
 gray[i]=.299*rgb[i*3]+.587*rgb[i*3+1]+.114*rgb[i*3+2];normals[i]=0;
 if(frame>=cfg.activeFrames || z<=0 || x==0 || y==0 || x+1>=cfg.width || y+1>=cfg.height)return;
 float za=depths[i-1],zb=depths[i+1],zc=depths[i-cfg.width],zd=depths[i+cfg.width];
 if(min(min(za,zb),min(zc,zd))<=0 || abs(za-zb)>=cfg.depthJump || abs(zc-zd)>=cfg.depthJump)return;
 float3 n=cross(cameraVertex(frame,x+1,y)-cameraVertex(frame,x-1,y),
                cameraVertex(frame,x,y+1)-cameraVertex(frame,x,y-1));
 float lengthN=length(n);if(lengthN>1e-10)normals[i]=float4(n/lengthN,1);
}
// Cache at most six scalar factors and eighteen pose/depth cross terms per
// source pixel. No lane holds a dense 18x18 matrix. Every lane reaches barriers.
groupshared float waveTerms[64];
float worldJacComponent(uint source,uint target,uint coordinate,float value) {
 uint frame=coordinate/6+1;
 return frame==source?value:(frame==target?-value:0);
}
[shader("compute")][numthreads(64,1,1)]
void linearizeSchur(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint lane:SV_GroupIndex) {
 uint i=tid.x,image=cfg.width*cfg.height,source=i/image;
 float J[36],R[6],D[6],crossTerms[18];uint targets[6];uint factorCount=0;
 for(uint k=0;k<36;k++)J[k]=0;
 for(uint k=0;k<6;k++){R[k]=0;D[k]=0;targets[k]=0;}
 for(uint k=0;k<18;k++)crossTerms[k]=0;
 float depthH=1,depthB=0,cost=0;
 bool valid=i<cfg.count && source<cfg.activeFrames;
 if(valid)valid=inverseDepth[i]>0 && isfinite(inverseDepth[i]) && depths[i]>0;
 if(valid) {
  uint pixel=i%image,x=pixel%cfg.width,y=pixel/cfg.width;
  float rho=inverseDepth[i],measured=1/depths[i],prior=rho-measured;
  depthH=cfg.depthWeight*cfg.depthWeight+cfg.damping;
  depthB=cfg.depthWeight*cfg.depthWeight*prior;
  cost=cfg.depthWeight*cfg.depthWeight*prior*prior;
  float3 sourceRay=ray(x,y),world=toWorld(source,sourceRay/rho);
  for(uint target=0;target<cfg.activeFrames;target++) {
   if(target==source)continue;
   float3 p=toCamera(target,world);if(p.z<=1e-6||!all(isfinite(p)))continue;
   float u,v;caspar_project(p.x,p.y,p.z,cfg.fx,cfg.fy,cfg.cx,cfg.cy,u,v);float2 uv=float2(u,v);
   if(!all(isfinite(uv)))continue;
   if(any(uv<0)||uv.x>=float(cfg.width-1)||uv.y>=float(cfg.height-1))continue;
   uint2 nearest=uint2(floor(uv+.5));uint k=target*image+nearest.y*cfg.width+nearest.x;
   float z=depths[k];float4 normal=normals[k];if(z<=0||normal.w<=0)continue;
   float3 q=ray(nearest.x,nearest.y)*z;if(length(p-q)>=cfg.distance)continue;
   uint2 base=uint2(floor(uv));float2 f=uv-float2(base);uint offset=target*image+base.y*cfg.width+base.x;
   float a=gray[offset],b=gray[offset+1],c=gray[offset+cfg.width],d=gray[offset+cfg.width+1];
   float intensity=lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y);
   float gx=lerp(b-a,d-c,f.y),gy=lerp(c-a,d-b,f.x);
   float output[16];uint t=target*16,s=source*16;
   caspar_dense_factor(world.x,world.y,world.z,p.x,p.y,p.z,
    poses[t],poses[t+4],poses[t+8],poses[t+1],poses[t+5],poses[t+9],poses[t+2],poses[t+6],poses[t+10],
    poses[s],poses[s+1],poses[s+2],poses[s+4],poses[s+5],poses[s+6],poses[s+8],poses[s+9],poses[s+10],
    sourceRay.x,sourceRay.y,sourceRay.z,rho,q.x,q.y,q.z,normal.x,normal.y,normal.z,
    intensity,gray[i],gx,gy,cfg.fx,cfg.fy,
    output[0],output[1],output[2],output[3],output[4],output[5],output[6],output[7],
    output[8],output[9],output[10],output[11],output[12],output[13],output[14],output[15]);
   bool finiteFactor=true;for(uint check=0;check<16;check++)finiteFactor=finiteFactor&&isfinite(output[check]);
   if(!finiteFactor)continue;
   for(uint channel=0;channel<2;channel++) {
    float residual=output[channel],scale=channel==0?cfg.geoWeight:cfg.photoWeight;
    float huber=channel==0?.03:10.;scale*=sqrt(min(1.,huber/max(abs(residual),1e-20)));
    uint factor=factorCount++;R[factor]=residual*scale;D[factor]=output[14+channel]*scale;targets[factor]=target;
    for(uint coordinate=0;coordinate<6;coordinate++)J[factor*6+coordinate]=output[2+channel*6+coordinate]*scale;
    depthH+=D[factor]*D[factor];depthB+=D[factor]*R[factor];cost+=R[factor]*R[factor];
   }
  }
  for(uint coordinate=0;coordinate<cfg.dofs;coordinate++) {
   float value=0;
   for(uint factor=0;factor<factorCount;factor++)value+=worldJacComponent(source,targets[factor],coordinate,J[factor*6+coordinate%6])*D[factor];
   crossTerms[coordinate]=value;hpd[i*cfg.dofs+coordinate]=value;
  }
  hdd[i]=depthH;bd[i]=depthB;
 } else if(i<cfg.count) {
  for(uint coordinate=0;coordinate<cfg.dofs;coordinate++)hpd[i*cfg.dofs+coordinate]=0;
  hdd[i]=1;bd[i]=0;
 }
 for(uint term=0;term<cfg.terms;term++) {
  float value=0;
  if(valid) {
   if(term<cfg.dofs*cfg.dofs) {
    uint row=term/cfg.dofs,column=term%cfg.dofs;
    for(uint factor=0;factor<factorCount;factor++)value+=worldJacComponent(source,targets[factor],row,J[factor*6+row%6])*worldJacComponent(source,targets[factor],column,J[factor*6+column%6]);
    value-=crossTerms[row]*crossTerms[column]/depthH;
   } else if(term<cfg.dofs*cfg.dofs+cfg.dofs) {
    uint row=term-cfg.dofs*cfg.dofs;
    for(uint factor=0;factor<factorCount;factor++)value+=worldJacComponent(source,targets[factor],row,J[factor*6+row%6])*R[factor];
    value-=crossTerms[row]*depthB/depthH;
   } else if(term==cfg.dofs*cfg.dofs+cfg.dofs)value=cost;else value=float(factorCount);
  }
  float sum=WaveActiveSum(value);
  if(WaveIsFirstLane())waveTerms[lane/WaveGetLaneCount()]=sum;
  GroupMemoryBarrierWithGroupSync();
  if(lane==0) {float total=0;for(uint wave=0;wave<(64+WaveGetLaneCount()-1)/WaveGetLaneCount();wave++)total+=waveTerms[wave];outputTerms[gid.x*cfg.terms+term]=total;}
  GroupMemoryBarrierWithGroupSync();
 }
}
[shader("compute")][numthreads(64,1,1)]
void reduceSchur(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint lane:SV_GroupIndex) {
 for(uint term=0;term<cfg.terms;term++) {
  float value=tid.x<cfg.count?inputTerms[tid.x*cfg.terms+term]:0;
  float sum=WaveActiveSum(value);
  if(WaveIsFirstLane())waveTerms[lane/WaveGetLaneCount()]=sum;
  GroupMemoryBarrierWithGroupSync();
  if(lane==0) {float total=0;for(uint wave=0;wave<(64+WaveGetLaneCount()-1)/WaveGetLaneCount();wave++)total+=waveTerms[wave];outputTerms[gid.x*cfg.terms+term]=total;}
  GroupMemoryBarrierWithGroupSync();
 }
}
[shader("compute")][numthreads(1,1,1)]
void solveSchur() {
 float L[324],y[18],x[18];bool valid=inputTerms[cfg.terms-1]>=float(6*(cfg.activeFrames-1));
 for(uint k=0;k<324;k++)L[k]=0;
 for(uint k=0;k<18;k++){y[k]=0;x[k]=0;}
 for(uint i=0;i<cfg.dofs;i++)for(uint j=0;j<=i;j++) {
  float value=inputTerms[i*cfg.dofs+j]+(i==j?cfg.damping:0);
  for(uint k=0;k<j;k++)value-=L[i*18+k]*L[j*18+k];
  if(i==j){if(!(value>0)||!isfinite(value))valid=false;L[i*18+j]=valid?sqrt(value):1;}
  else{L[i*18+j]=value/L[j*18+j];if(!isfinite(L[i*18+j]))valid=false;}
 }
 for(uint i=0;i<cfg.dofs;i++) {
  float value=-inputTerms[cfg.dofs*cfg.dofs+i];
  for(uint k=0;k<i;k++)value-=L[i*18+k]*y[k];
  y[i]=value/L[i*18+i];if(!isfinite(y[i]))valid=false;
 }
 for(int i=int(cfg.dofs)-1;i>=0;i--) {
  float value=y[i];for(uint k=uint(i)+1;k<cfg.dofs;k++)value-=L[k*18+uint(i)]*x[k];
  x[i]=value/L[uint(i)*18+uint(i)];if(!isfinite(x[i]))valid=false;
 }
 status[0]=valid?0:1;for(uint k=0;k<cfg.dofs;k++)delta[k]=valid?x[k]:0;
}
[shader("compute")][numthreads(64,1,1)]
void updateDensePoses(uint3 tid:SV_DispatchThreadID) {
 uint frame=tid.x;if(frame==0||frame>=cfg.activeFrames||status[0]!=0)return;
 uint begin=(frame-1)*6;float3 w=float3(delta[begin],delta[begin+1],delta[begin+2]),v=float3(delta[begin+3],delta[begin+4],delta[begin+5]);
 float a,b,c;caspar_se3_coefficients(dot(w,w),a,b,c);uint offset=frame*16;
 float oldPose[16],updatedPose[16];for(uint k=0;k<16;k++){oldPose[k]=poses[offset+k];updatedPose[k]=oldPose[k];}
 bool finitePose=true;
 for(uint column=0;column<4;column++) {
  float3 p=float3(oldPose[column],oldPose[4+column],oldPose[8+column]);float x,y,z;
  caspar_twist_point(w.x,w.y,w.z,v.x,v.y,v.z,a,b,c,p.x,p.y,p.z,column==3?1.:0.,x,y,z);
  updatedPose[column]=x;updatedPose[4+column]=y;updatedPose[8+column]=z;
  finitePose=finitePose&&isfinite(x)&&isfinite(y)&&isfinite(z);
 }
 if(finitePose)for(uint k=0;k<16;k++)candidatePoses[offset+k]=updatedPose[k];
 else {uint ignored;InterlockedOr(status[0],1,ignored);}

}
[shader("compute")][numthreads(64,1,1)]
void commitDensePoses(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i<16 || i>=cfg.activeFrames*16 || status[0]!=0)return;
 poses[i]=candidatePoses[i];
}
[shader("compute")][numthreads(64,1,1)]
void updateDenseDepth(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=cfg.count||status[0]!=0||inverseDepth[i]<=0)return;
 float value=bd[i];for(uint k=0;k<cfg.dofs;k++)value+=hpd[i*cfg.dofs+k]*delta[k];
 float updated=inverseDepth[i]-value/hdd[i];if(isfinite(updated)){float measured=1/depths[i];inverseDepth[i]=clamp(updated,measured/1.1,measured/.9);}
}

// Acceptance is based on a fresh evaluation after the final pose/depth update.
RWStructuredBuffer<float> costs;
RWStructuredBuffer<uint> transactionStatus;
StructuredBuffer<float> backupPoses,backupDepth;
[shader("compute")][numthreads(1,1,1)]
void saveInitialCost() {costs[0]=inputTerms[cfg.terms-2];costs[1]=costs[0];costs[2]=inputTerms[cfg.terms-1];costs[3]=costs[2];transactionStatus[0]=0;}
[shader("compute")][numthreads(1,1,1)]
void latchDenseFailure() {if(status[0]!=0)transactionStatus[0]=1;}
[shader("compute")][numthreads(1,1,1)]
void finalizeDense() {
 float initial=costs[0],final=inputTerms[cfg.terms-2];costs[1]=final;
 costs[3]=inputTerms[cfg.terms-1];
 bool enough=costs[3]>=float(6*(cfg.activeFrames-1))&&costs[3]>=cfg.minFactorRatio*costs[2];
 bool accepted=transactionStatus[0]==0&&enough&&isfinite(initial)&&isfinite(final)&&final<=initial+abs(initial)*1e-4+1e-7;
 status[0]=accepted?0:(enough?1:2);
}
[shader("compute")][numthreads(64,1,1)]
void rollbackDense(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(status[0]==0)return;
 if(i<cfg.frames*16)poses[i]=backupPoses[i];
 if(i<cfg.count)inverseDepth[i]=backupDepth[i];
}
