
struct GraphParams {uint nodeCount;uint edgeCount;float damping;uint firstEdge;uint initCount;};
ConstantBuffer<GraphParams> cfg;
RWStructuredBuffer<float> poses;
StructuredBuffer<float> backup;
RWStructuredBuffer<float> measurements;
StructuredBuffer<float> edgeInformation;
StructuredBuffer<uint> topology,offsets,adjacency,initMask;
RWStructuredBuffer<float> edgeJacobian,edgeResidual,edgeAction,edgeCosts;
RWStructuredBuffer<uint> edgeValid;
RWStructuredBuffer<float> solution,residualCG,directionCG,preconditioner,appliedCG;
RWStructuredBuffer<float> cgScalars; // rho, alpha, initial rho, initial edge cost
RWStructuredBuffer<uint> cgState; // active PCG iteration
RWStructuredBuffer<uint> stats; // valid edges, persistent failure
RWStructuredBuffer<float> cost; // pre/post cost of last nonlinear iteration

groupshared float shared0[64];
groupshared float shared1[64];
groupshared float alphaShared;
groupshared float betaShared;

float4 matrixQuaternion(float3 a,float3 b,float3 c) {
 float trace=a.x+b.y+c.z;float4 q;
 if(trace>0){float s=2*sqrt(max(trace+1,1e-20));q=float4((c.y-b.z)/s,(a.z-c.x)/s,(b.x-a.y)/s,s*.25);}
 else if(a.x>b.y && a.x>c.z){float s=2*sqrt(max(1+a.x-b.y-c.z,1e-20));q=float4(s*.25,(a.y+b.x)/s,(a.z+c.x)/s,(c.y-b.z)/s);}
 else if(b.y>c.z){float s=2*sqrt(max(1+b.y-a.x-c.z,1e-20));q=float4((a.y+b.x)/s,s*.25,(b.z+c.y)/s,(a.z-c.x)/s);}
 else{float s=2*sqrt(max(1+c.z-a.x-b.y,1e-20));q=float4((a.z+c.x)/s,(b.z+c.y)/s,s*.25,(b.x-a.y)/s);}
 return q*rsqrt(max(dot(q,q),1e-20));
}
float4 poseQuaternion(uint node) {
 uint i=node*16;return matrixQuaternion(float3(poses[i],poses[i+1],poses[i+2]),float3(poses[i+4],poses[i+5],poses[i+6]),float3(poses[i+8],poses[i+9],poses[i+10]));
}
float3 poseTranslation(uint node){uint i=node*16;return float3(poses[i+3],poses[i+7],poses[i+11]);}
float4 measurementQuaternion(uint edge) {
 uint i=edge*16;return matrixQuaternion(float3(measurements[i],measurements[i+1],measurements[i+2]),float3(measurements[i+4],measurements[i+5],measurements[i+6]),float3(measurements[i+8],measurements[i+9],measurements[i+10]));
}
float3 measurementTranslation(uint edge){uint i=edge*16;return float3(measurements[i+3],measurements[i+7],measurements[i+11]);}
bool properRows(float3 a,float3 b,float3 c,float4 tail) {
 return all(isfinite(a))&&all(isfinite(b))&&all(isfinite(c))&&all(isfinite(tail))&&
 abs(dot(a,a)-1)<.002&&abs(dot(b,b)-1)<.002&&abs(dot(c,c)-1)<.002&&
 abs(dot(a,b))<.002&&abs(dot(a,c))<.002&&abs(dot(b,c))<.002&&
 abs(dot(a,cross(b,c))-1)<.002&&all(abs(tail-float4(0,0,0,1))<1e-5);
}
bool properPose(uint n) {
 uint i=n*16;return properRows(float3(poses[i],poses[i+1],poses[i+2]),float3(poses[i+4],poses[i+5],poses[i+6]),float3(poses[i+8],poses[i+9],poses[i+10]),float4(poses[i+12],poses[i+13],poses[i+14],poses[i+15]))&&all(isfinite(poseTranslation(n)));
}
bool properMeasurement(uint e) {
 uint i=e*16;return properRows(float3(measurements[i],measurements[i+1],measurements[i+2]),float3(measurements[i+4],measurements[i+5],measurements[i+6]),float3(measurements[i+8],measurements[i+9],measurements[i+10]),float4(measurements[i+12],measurements[i+13],measurements[i+14],measurements[i+15]))&&all(isfinite(measurementTranslation(e)));
}
void graphFactor(float4 qi,float3 ti,float4 qj,float3 tj,float4 qm,float3 tm,out float r[6],out float Ji[36],out float Jj[36]) {
 caspar_pose_graph_factor(qi.x,qi.y,qi.z,qi.w,ti.x,ti.y,ti.z,qj.x,qj.y,qj.z,qj.w,tj.x,tj.y,tj.z,qm.x,qm.y,qm.z,qm.w,tm.x,tm.y,tm.z,r[0],r[1],r[2],r[3],r[4],r[5],Ji[0],Ji[1],Ji[2],Ji[3],Ji[4],Ji[5],Ji[6],Ji[7],Ji[8],Ji[9],Ji[10],Ji[11],Ji[12],Ji[13],Ji[14],Ji[15],Ji[16],Ji[17],Ji[18],Ji[19],Ji[20],Ji[21],Ji[22],Ji[23],Ji[24],Ji[25],Ji[26],Ji[27],Ji[28],Ji[29],Ji[30],Ji[31],Ji[32],Ji[33],Ji[34],Ji[35],Jj[0],Jj[1],Jj[2],Jj[3],Jj[4],Jj[5],Jj[6],Jj[7],Jj[8],Jj[9],Jj[10],Jj[11],Jj[12],Jj[13],Jj[14],Jj[15],Jj[16],Jj[17],Jj[18],Jj[19],Jj[20],Jj[21],Jj[22],Jj[23],Jj[24],Jj[25],Jj[26],Jj[27],Jj[28],Jj[29],Jj[30],Jj[31],Jj[32],Jj[33],Jj[34],Jj[35]);
}
void edgeFactor(uint e,out float r[6],out float Ji[36],out float Jj[36]) {
 uint i=topology[e*2],j=topology[e*2+1];
 graphFactor(poseQuaternion(i),poseTranslation(i),poseQuaternion(j),poseTranslation(j),measurementQuaternion(e),measurementTranslation(e),r,Ji,Jj);
}
void reducePair(uint lane) {
 GroupMemoryBarrierWithGroupSync();
 for(uint stride=32;stride>0;stride>>=1){if(lane<stride){shared0[lane]+=shared0[lane+stride];shared1[lane]+=shared1[lane+stride];}GroupMemoryBarrierWithGroupSync();}
}

[shader("compute")][numthreads(64,1,1)]
void initializeMeasurements(uint3 tid:SV_DispatchThreadID) {
 uint e=cfg.firstEdge+tid.x;if(tid.x>=cfg.initCount||e>=cfg.edgeCount||initMask[e]==0)return;
 uint ni=topology[e*2],nj=topology[e*2+1];if(ni>=cfg.nodeCount||nj>=cfg.nodeCount)return;
 uint i=ni*16,j=nj*16;
 [unroll]for(uint col=0;col<4;col++) {
  float3 p=float3(poses[j+col],poses[j+4+col],poses[j+8+col]);
  if(col==3)p-=float3(poses[i+3],poses[i+7],poses[i+11]);
  float x,y,z;caspar_transform(p.x,p.y,p.z,poses[i],poses[i+4],poses[i+8],poses[i+1],poses[i+5],poses[i+9],poses[i+2],poses[i+6],poses[i+10],0,0,0,x,y,z);
  measurements[e*16+col]=x;measurements[e*16+4+col]=y;measurements[e*16+8+col]=z;measurements[e*16+12+col]=col==3?1.:0.;
 }
}
[shader("compute")][numthreads(1,1,1)]
void resetGraph(){if(cfg.nodeCount>0){stats[0]=0;stats[1]=0;cost[0]=0;cost[1]=0;}}
[shader("compute")][numthreads(1,1,1)]
void resetEdgeCount(){if(cfg.nodeCount>0)stats[0]=0;}

[shader("compute")][numthreads(64,1,1)]
void linearizeEdges(uint3 tid:SV_DispatchThreadID) {
 uint e=tid.x;if(e>=cfg.edgeCount)return;edgeValid[e]=0;edgeCosts[e]=0;
 if(stats[1]!=0)return;
 float2 info=float2(edgeInformation[e*2],edgeInformation[e*2+1]);
 if(!all(isfinite(info))||any(info<0)){InterlockedOr(stats[1],1);return;}
 if(all(info==0))return;
 uint i=topology[e*2],j=topology[e*2+1];
 if(i>=cfg.nodeCount||j>=cfg.nodeCount||i==j){InterlockedOr(stats[1],1);return;}
 if(!properPose(i)||!properPose(j)||!properMeasurement(e)){InterlockedOr(stats[1],1);return;}
 float r[6],Ji[36],Jj[36];edgeFactor(e,r,Ji,Jj);bool finite=true;float sum=0;
 [unroll]for(uint row=0;row<6;row++) {
  float weight=row<3?info.x:info.y;float value=r[row]*weight;finite=finite&&isfinite(value);edgeResidual[e*6+row]=value;sum+=value*value;
  [unroll]for(uint col=0;col<6;col++) {
   float a=Ji[row*6+col]*weight,b=Jj[row*6+col]*weight;
   finite=finite&&isfinite(a)&&isfinite(b);edgeJacobian[e*72+row*12+col]=a;edgeJacobian[e*72+row*12+6+col]=b;
  }
 }
 if(!finite||!isfinite(sum)){InterlockedOr(stats[1],1);return;}
 edgeCosts[e]=sum;edgeValid[e]=1;InterlockedAdd(stats[0],1);
}

[shader("compute")][numthreads(64,1,1)]
void initializeCG(uint lane:SV_GroupIndex) {
 float rho=0,total=0;
 for(uint index=lane;index<cfg.nodeCount*6;index+=64) {
  uint node=index/6,col=index%6;float diagonal=cfg.damping,gradient=0;
  if(node>0 && stats[1]==0)for(uint k=offsets[node];k<offsets[node+1];k++) {
   uint encoded=adjacency[k],e=encoded/2,side=encoded%2;if(e>=cfg.edgeCount||edgeValid[e]==0)continue;
   [unroll]for(uint row=0;row<6;row++){float j=edgeJacobian[e*72+row*12+side*6+col];gradient+=j*edgeResidual[e*6+row];diagonal+=j*j;}
  }
  if(node==0){diagonal=1;gradient=0;}
  diagonal=max(diagonal,1e-12);float r=-gradient,z=r/diagonal;
  solution[index]=0;residualCG[index]=r;directionCG[index]=z;preconditioner[index]=diagonal;rho+=r*z;
 }
 for(uint e=lane;e<cfg.edgeCount;e+=64)total+=edgeCosts[e];
 shared0[lane]=rho;shared1[lane]=total;reducePair(lane);
 if(lane==0){cgScalars[0]=shared0[0];cgScalars[1]=0;cgScalars[2]=shared0[0];cgScalars[3]=shared1[0];cost[0]=shared1[0];
  bool finite=isfinite(shared0[0])&&isfinite(shared1[0]);if(!finite)stats[1]=1;cgState[0]=stats[1]==0&&shared0[0]>1e-20?1:0;}
}
[shader("compute")][numthreads(64,1,1)]
void operatorEdges(uint3 tid:SV_DispatchThreadID) {
 uint e=tid.x/6,row=tid.x%6;if(e>=cfg.edgeCount)return;float value=0;
 if(cgState[0]!=0&&edgeValid[e]!=0){uint i=topology[e*2],j=topology[e*2+1];[unroll]for(uint col=0;col<6;col++)value+=edgeJacobian[e*72+row*12+col]*directionCG[i*6+col]+edgeJacobian[e*72+row*12+6+col]*directionCG[j*6+col];}
 edgeAction[e*6+row]=value;
}
[shader("compute")][numthreads(64,1,1)]
void operatorNodes(uint3 tid:SV_DispatchThreadID) {
 uint index=tid.x;if(index>=cfg.nodeCount*6)return;uint node=index/6,col=index%6;float value=0;
 if(node>0&&cgState[0]!=0){value=cfg.damping*directionCG[index];for(uint k=offsets[node];k<offsets[node+1];k++){
  uint encoded=adjacency[k],e=encoded/2,side=encoded%2;if(e>=cfg.edgeCount||edgeValid[e]==0)continue;
  [unroll]for(uint row=0;row<6;row++)value+=edgeJacobian[e*72+row*12+side*6+col]*edgeAction[e*6+row];}}
 appliedCG[index]=value;
}
[shader("compute")][numthreads(64,1,1)]
void pcgStep(uint lane:SV_GroupIndex) {
 float dot=0;for(uint i=lane;i<cfg.nodeCount*6;i+=64)dot+=directionCG[i]*appliedCG[i];
 shared0[lane]=dot;shared1[lane]=0;reducePair(lane);
 if(lane==0){alphaShared=0;if(cgState[0]!=0&&stats[1]==0){float denominator=shared0[0];if(denominator>0&&isfinite(denominator)){alphaShared=cgScalars[0]/denominator;if(!isfinite(alphaShared)){stats[1]=1;cgState[0]=0;alphaShared=0;}}else{stats[1]=1;cgState[0]=0;}}}
 GroupMemoryBarrierWithGroupSync();
 float rho=0;
 for(uint i=lane;i<cfg.nodeCount*6;i+=64){solution[i]+=alphaShared*directionCG[i];float r=residualCG[i]-alphaShared*appliedCG[i];residualCG[i]=r;rho+=r*r/preconditioner[i];}
 shared0[lane]=rho;shared1[lane]=0;reducePair(lane);
 if(lane==0){betaShared=0;if(cgState[0]!=0&&stats[1]==0){float next=shared0[0];if(!isfinite(next)){stats[1]=1;cgState[0]=0;}else{betaShared=next/cgScalars[0];cgScalars[0]=next;if(next<=1e-10*max(cgScalars[2],1.)){cgState[0]=0;betaShared=0;}}}cgScalars[1]=alphaShared;}
 GroupMemoryBarrierWithGroupSync();
 for(uint i=lane;i<cfg.nodeCount*6;i+=64)directionCG[i]=cgState[0]!=0?residualCG[i]/preconditioner[i]+betaShared*directionCG[i]:0;
}

[shader("compute")][numthreads(64,1,1)]
void updateNodes(uint3 tid:SV_DispatchThreadID) {
 uint node=tid.x;if(node==0||node>=cfg.nodeCount||stats[1]!=0)return;
 float3 w=float3(solution[node*6],solution[node*6+1],solution[node*6+2]);float3 v=float3(solution[node*6+3],solution[node*6+4],solution[node*6+5]);
 float a,b,c;caspar_se3_coefficients(dot(w,w),a,b,c);float updated[12];bool finite=true;
 [unroll]for(uint col=0;col<4;col++){uint i=node*16;float x,y,z;caspar_twist_point(w.x,w.y,w.z,v.x,v.y,v.z,a,b,c,poses[i+col],poses[i+4+col],poses[i+8+col],col==3?1.:0.,x,y,z);updated[col]=x;updated[4+col]=y;updated[8+col]=z;finite=finite&&all(isfinite(float3(x,y,z)));}
 if(!finite){InterlockedOr(stats[1],1);return;}
 [unroll]for(uint k=0;k<12;k++)poses[node*16+k]=updated[k];
}
[shader("compute")][numthreads(64,1,1)]
void evaluateEdges(uint3 tid:SV_DispatchThreadID) {
 uint e=tid.x;if(e>=cfg.edgeCount)return;edgeCosts[e]=0;if(edgeValid[e]==0||stats[1]!=0)return;
 float r[6],Ji[36],Jj[36];edgeFactor(e,r,Ji,Jj);float sum=0;
 [unroll]for(uint row=0;row<6;row++){float value=r[row]*edgeInformation[e*2+(row<3?0:1)];sum+=value*value;}
 edgeCosts[e]=sum;if(!isfinite(sum))InterlockedOr(stats[1],1);
}
[shader("compute")][numthreads(64,1,1)]
void checkCost(uint lane:SV_GroupIndex) {
 float sum=0;for(uint e=lane;e<cfg.edgeCount;e+=64)sum+=edgeCosts[e];shared0[lane]=sum;shared1[lane]=0;reducePair(lane);
 if(lane==0){cost[1]=shared0[0];if(!isfinite(shared0[0])||shared0[0]>cgScalars[3]*1.001+1e-6)stats[1]=1;}
}
[shader("compute")][numthreads(64,1,1)]
void rollbackNodes(uint3 tid:SV_DispatchThreadID){uint i=tid.x;if(i<cfg.nodeCount*16&&stats[1]!=0)poses[i]=backup[i];}
