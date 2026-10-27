struct LoopParams {uint capacity;uint edgeIndex;uint querySlot;uint nodeIndex;uint historyCount;uint currentFrame;uint anchorSlot;uint ranBA;uint targetSlot;uint queryPhysicalSlot;uint queryFrameId;uint targetFrameId;uint odometryEdges;uint activeEdges;uint proposalEnabled;};
ConstantBuffer<LoopParams> lp;
StructuredBuffer<float> verifiedCameraPose;
StructuredBuffer<float> targetWorldPose;
StructuredBuffer<float> queryOldWorldPose;
RWStructuredBuffer<float> currentWorldPose;
RWStructuredBuffer<uint> trackingStatus;
StructuredBuffer<uint> verifierStats;
StructuredBuffer<uint> archiveFrameIds;
StructuredBuffer<uint> graphStats;
StructuredBuffer<float> graphPoses;
StructuredBuffer<float> graphBackup;
RWStructuredBuffer<float> graphMeasurements;
RWStructuredBuffer<float> graphInformation;
RWStructuredBuffer<uint> loopEvent;
RWStructuredBuffer<uint> graphGuard;
RWStructuredBuffer<float> archivePose;
RWStructuredBuffer<float> savedLoopMeasurement;
void inverseRigid(float m[16],out float result[16]) {
 for(uint i=0;i<16;i++)result[i]=0;result[15]=1;
 for(uint r=0;r<3;r++)for(uint c=0;c<3;c++)result[r*4+c]=m[c*4+r];
 float x,y,z;caspar_transform(-m[3],-m[7],-m[11],m[0],m[4],m[8],m[1],m[5],m[9],m[2],m[6],m[10],0,0,0,x,y,z);
 result[3]=x;result[7]=y;result[11]=z;
}
void multiplyRigid(float a[16],float b[16],out float result[16]) {
 for(uint i=0;i<16;i++)result[i]=0;result[15]=1;
 for(uint col=0;col<4;col++){float flag=col==3?1.:0.,x,y,z;
 caspar_transform(b[col],b[4+col],b[8+col],a[0],a[1],a[2],a[4],a[5],a[6],a[8],a[9],a[10],flag*a[3],flag*a[7],flag*a[11],x,y,z);
 result[col]=x;result[4+col]=y;result[8+col]=z;}
}
[shader("compute")][numthreads(1,1,1)]
void resetLoopEvent(){loopEvent[0]=0;loopEvent[1]=0;graphGuard[0]=0;graphGuard[1]=1;}
[shader("compute")][numthreads(64,1,1)]
void setGraphInformation(uint3 tid:SV_DispatchThreadID){uint edge=tid.x;if(edge>=lp.activeEdges)return;
 float value=edge<lp.odometryEdges?1.:10.;if(lp.proposalEnabled!=0&&edge==lp.edgeIndex)value=0;
 graphInformation[edge*2]=value;graphInformation[edge*2+1]=value;}
[shader("compute")][numthreads(1,1,1)]
void prepareLoopEdge() {
 if(lp.proposalEnabled!=0){graphInformation[lp.edgeIndex*2]=0;graphInformation[lp.edgeIndex*2+1]=0;}
 if(archiveFrameIds[lp.targetSlot]!=lp.targetFrameId||archiveFrameIds[lp.queryPhysicalSlot]!=lp.queryFrameId)return;
 if(verifierStats[1]!=0)return;
 float vc[16],vw[16];for(uint i=0;i<16;i++)vc[i]=verifiedCameraPose[i];inverseRigid(vc,vw);
 if(trackingStatus[1]!=0){for(uint i=0;i<16;i++)currentWorldPose[i]=vw[i];trackingStatus[0]=verifierStats[0];trackingStatus[1]=0;loopEvent[0]=2;return;}
 if(lp.proposalEnabled==0)return;
 float oldCurrent[16],oldInverse[16],oldQuery[16],target[16],targetInverse[16],delta[16],newQuery[16],measured[16];
 for(uint i=0;i<16;i++){oldCurrent[i]=currentWorldPose[i];oldQuery[i]=queryOldWorldPose[i];target[i]=targetWorldPose[i];}
 inverseRigid(oldCurrent,oldInverse);multiplyRigid(vw,oldInverse,delta);multiplyRigid(delta,oldQuery,newQuery);inverseRigid(target,targetInverse);multiplyRigid(targetInverse,newQuery,measured);
 for(uint i=0;i<16;i++)graphMeasurements[lp.edgeIndex*16+i]=measured[i];
 graphInformation[lp.edgeIndex*2]=10;graphInformation[lp.edgeIndex*2+1]=10;loopEvent[0]=1;
}
[shader("compute")][numthreads(1,1,1)]
void graphAcceptance(){graphGuard[0]=0;graphGuard[1]=(loopEvent[0]==1&&graphStats[1]==0&&trackingStatus[1]==0)?0:1;if(graphGuard[1]==0)loopEvent[1]=1;}
[shader("compute")][numthreads(16,1,1)]
void commitGraphNode(uint3 tid:SV_DispatchThreadID){if(graphGuard[1]==0&&tid.x<16)archivePose[tid.x]=graphPoses[lp.nodeIndex*16+tid.x];}
[shader("compute")][numthreads(1,1,1)]
void correctCurrentPose(){if(graphGuard[1]!=0)return;float old[16],next[16],inverseOld[16],delta[16],current[16],result[16];
 for(uint i=0;i<16;i++){old[i]=graphBackup[lp.querySlot*16+i];next[i]=graphPoses[lp.querySlot*16+i];current[i]=currentWorldPose[i];}
 inverseRigid(old,inverseOld);multiplyRigid(next,inverseOld,delta);multiplyRigid(delta,current,result);for(uint i=0;i<16;i++)currentWorldPose[i]=result[i];
}
[shader("compute")][numthreads(16,1,1)]
void retainLoopMeasurement(uint3 tid:SV_DispatchThreadID){if(graphGuard[1]==0&&tid.x<16)savedLoopMeasurement[tid.x]=graphMeasurements[lp.edgeIndex*16+tid.x];}
RWStructuredBuffer<float> historyOldNodes;
RWStructuredBuffer<float> historyNewNodes;
RWStructuredBuffer<float> historyPoses;
RWStructuredBuffer<uint> historyValid;
RWStructuredBuffer<uint> historyAnchors;
RWStructuredBuffer<uint> historyAnchorIds;

StructuredBuffer<uint> baStatus;
RWStructuredBuffer<uint> historyGuard;
StructuredBuffer<float> inputArchivePose;
[shader("compute")][numthreads(16,1,1)]
void snapshotOldNode(uint3 tid:SV_DispatchThreadID){if(tid.x<16)historyOldNodes[lp.nodeIndex*16+tid.x]=inputArchivePose[tid.x];}
[shader("compute")][numthreads(16,1,1)]
void snapshotNewNode(uint3 tid:SV_DispatchThreadID){if(tid.x<16)historyNewNodes[lp.nodeIndex*16+tid.x]=inputArchivePose[tid.x];}
[shader("compute")][numthreads(1,1,1)]
void makeHistoryGuard(){historyGuard[0]=0;historyGuard[1]=(trackingStatus[1]==0&&(loopEvent[1]!=0||(lp.ranBA!=0&&baStatus[0]==0)))?0:1;}
[shader("compute")][numthreads(64,1,1)]
void correctHistory(uint3 tid:SV_DispatchThreadID){uint index=tid.x;if(historyGuard[1]!=0||index>=lp.historyCount||historyValid[index]==0)return;
 uint slot=historyAnchors[index];if(slot>=lp.capacity||historyAnchorIds[index]!=archiveFrameIds[slot])return;
 float old[16],next[16],oldInverse[16],delta[16],frame[16],result[16];for(uint i=0;i<16;i++){old[i]=historyOldNodes[slot*16+i];next[i]=historyNewNodes[slot*16+i];frame[i]=historyPoses[index*16+i];}
 inverseRigid(old,oldInverse);multiplyRigid(next,oldInverse,delta);multiplyRigid(delta,frame,result);for(uint i=0;i<16;i++)historyPoses[index*16+i]=result[i];
}
[shader("compute")][numthreads(16,1,1)]
void storeHistory(uint3 tid:SV_DispatchThreadID){uint i=tid.x;if(trackingStatus[1]!=0){if(i==0)historyValid[lp.currentFrame]=0;return;}
 if(i<16)historyPoses[lp.currentFrame*16+i]=currentWorldPose[i];
 if(i==0){historyValid[lp.currentFrame]=1;historyAnchors[lp.currentFrame]=lp.anchorSlot;historyAnchorIds[lp.currentFrame]=archiveFrameIds[lp.anchorSlot];}
}
