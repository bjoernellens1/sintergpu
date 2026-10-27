struct VerificationParams {uint maxFeatures,hypotheses;float maxDistance;uint minMatches;float maxRMSE;};
ConstantBuffer<VerificationParams> cfg;
StructuredBuffer<float4> sourceLandmarks,targetLandmarks,sourcePositions;
StructuredBuffer<uint> sourceCount,targetCount,matches;
RWStructuredBuffer<uint> pairIndices,pairCount,hypothesisScores,verificationStats;
RWStructuredBuffer<float> hypothesisPoses,hypothesisCosts,bestRigid,candidatePose,outputPose,verificationCost;
StructuredBuffer<float> targetWorldPose;
RWStructuredBuffer<float> worldPoints,observations;
RWStructuredBuffer<uint> validMask;
StructuredBuffer<uint> refinementStats;
StructuredBuffer<float> refinementCost;
uint hashIndex(uint k) {k^=k>>16;k*=0x7feb352d;k^=k>>15;k*=0x846ca68b;return k^(k>>16);}
float3 transformRigid(float3 p,float matrix[16]) {
 float x,y,z;caspar_transform(p.x,p.y,p.z,matrix[0],matrix[1],matrix[2],matrix[4],matrix[5],matrix[6],matrix[8],matrix[9],matrix[10],matrix[3],matrix[7],matrix[11],x,y,z);return float3(x,y,z);
}
[shader("compute")][numthreads(1,1,1)]
void resetVerification() {pairCount[0]=0;verificationStats[0]=0;verificationStats[1]=1;verificationCost[0]=0;}
[shader("compute")][numthreads(64,1,1)]
void compactPairs(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;bool valid=i<min(sourceCount[0],cfg.maxFeatures);
 if(valid){uint j=matches[i];valid=j<min(targetCount[0],cfg.maxFeatures);if(valid){float4 a=sourceLandmarks[i],b=targetLandmarks[j];valid=a.w>0&&b.w>0&&a.z>0&&b.z>0&&all(isfinite(a))&&all(isfinite(b));}}
 uint offset=WavePrefixCountBits(valid),count=WaveActiveCountBits(valid),base=0;
 if(WaveIsFirstLane())InterlockedAdd(pairCount[0],count,base);
 base=WaveReadLaneFirst(base);if(valid && base+offset<cfg.maxFeatures)pairIndices[base+offset]=i;
}
groupshared float model[16];
groupshared uint modelValid;
groupshared uint inliers[64];
groupshared float errors[64];
[shader("compute")][numthreads(64,1,1)]
void scoreHypotheses(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
 uint hypothesis=group.x,n=min(pairCount[0],cfg.maxFeatures);
 if(lane==0) {
  modelValid=0;for(uint k=0;k<16;k++)model[k]=k%5==0?1:0;
  if(n>=3) {
   uint first=hashIndex(hypothesis*3+1)%n,second=hashIndex(hypothesis*3+2)%(n-1);
   if(second>=first)second++;
   uint third=hashIndex(hypothesis*3+3)%(n-2);if(third>=min(first,second))third++;if(third>=max(first,second))third++;
   uint a=pairIndices[first],b=pairIndices[second],c=pairIndices[third];
   float3 s0=sourceLandmarks[a].xyz,s1=sourceLandmarks[b].xyz,s2=sourceLandmarks[c].xyz;
   float3 t0=targetLandmarks[matches[a]].xyz,t1=targetLandmarks[matches[b]].xyz,t2=targetLandmarks[matches[c]].xyz;
   float3 su=s1-s0,sv=cross(su,s2-s0),tu=t1-t0,tv=cross(tu,t2-t0);
   float slu=length(su),slv=length(sv),tlu=length(tu),tlv=length(tv);
   if(min(min(slu,slv),min(tlu,tlv))>1e-5) {
    su/=slu;sv/=slv;tu/=tlu;tv/=tlv;float3 sw=cross(sv,su),tw=cross(tv,tu);
    for(uint row=0;row<3;row++)for(uint col=0;col<3;col++)model[row*4+col]=tu[row]*su[col]+tw[row]*sw[col]+tv[row]*sv[col];
    float x,y,z;caspar_transform(s0.x,s0.y,s0.z,model[0],model[1],model[2],model[4],model[5],model[6],model[8],model[9],model[10],0,0,0,x,y,z);
    float3 translation=t0-float3(x,y,z);model[3]=translation.x;model[7]=translation.y;model[11]=translation.z;
    modelValid=1;for(uint k=0;k<16;k++)if(!isfinite(model[k]))modelValid=0;
   }
  }
 }
 GroupMemoryBarrierWithGroupSync();
 uint count=0;float error=0;
 if(modelValid!=0) {
  float cached[16];for(uint k=0;k<16;k++)cached[k]=model[k];
  for(uint index=lane;index<n;index+=64) {
   uint i=pairIndices[index],j=matches[i];float3 difference=transformRigid(sourceLandmarks[i].xyz,cached)-targetLandmarks[j].xyz;
   float squared=dot(difference,difference);if(isfinite(squared)&&squared<cfg.maxDistance*cfg.maxDistance){count++;error+=squared;}
  }
 }
 inliers[lane]=count;errors[lane]=error;GroupMemoryBarrierWithGroupSync();
 for(uint stride=32;stride>0;stride>>=1){if(lane<stride){inliers[lane]+=inliers[lane+stride];errors[lane]+=errors[lane+stride];}GroupMemoryBarrierWithGroupSync();}
 if(lane==0){hypothesisScores[hypothesis]=inliers[0];hypothesisCosts[hypothesis]=errors[0];for(uint k=0;k<16;k++)hypothesisPoses[hypothesis*16+k]=model[k];}
}
[shader("compute")][numthreads(1,1,1)]
void selectHypothesis() {
 uint best=0,bestCount=0;float bestCost=1e30;
 for(uint h=0;h<cfg.hypotheses;h++){uint count=hypothesisScores[h];float cost=hypothesisCosts[h];if(count>bestCount||(count==bestCount&&cost<bestCost)){best=h;bestCount=count;bestCost=cost;}}
 verificationStats[0]=bestCount;if(bestCount<cfg.minMatches)return;
 float world[16];for(uint k=0;k<16;k++){world[k]=k%5==0?1:0;bestRigid[k]=hypothesisPoses[best*16+k];}
 for(uint col=0;col<4;col++) {
  float x,y,z,flag=col==3?1:0;
  caspar_transform(bestRigid[col],bestRigid[4+col],bestRigid[8+col],targetWorldPose[0],targetWorldPose[1],targetWorldPose[2],targetWorldPose[4],targetWorldPose[5],targetWorldPose[6],targetWorldPose[8],targetWorldPose[9],targetWorldPose[10],targetWorldPose[3]*flag,targetWorldPose[7]*flag,targetWorldPose[11]*flag,x,y,z);
  world[col]=x;world[4+col]=y;world[8+col]=z;
 }
 float inverted[16];for(uint k=0;k<16;k++)inverted[k]=k%5==0?1:0;
 for(uint row=0;row<3;row++)for(uint col=0;col<3;col++)inverted[row*4+col]=world[col*4+row];
 float x,y,z;caspar_transform(world[3],world[7],world[11],inverted[0],inverted[1],inverted[2],inverted[4],inverted[5],inverted[6],inverted[8],inverted[9],inverted[10],0,0,0,x,y,z);
 inverted[3]=-x;inverted[7]=-y;inverted[11]=-z;
 bool valid=true;for(uint k=0;k<16;k++)valid=valid&&isfinite(inverted[k]);if(!valid)return;
 for(uint k=0;k<16;k++)candidatePose[k]=inverted[k];verificationStats[1]=0;
}
[shader("compute")][numthreads(64,1,1)]
void buildRefinement(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=cfg.maxFeatures)return;validMask[i]=0;
 if(verificationStats[1]!=0||i>=min(sourceCount[0],cfg.maxFeatures))return;
 uint j=matches[i];if(j>=min(targetCount[0],cfg.maxFeatures))return;
 float4 a=sourceLandmarks[i],b=targetLandmarks[j],pixel=sourcePositions[i];
 if(a.w<=0||b.w<=0||!all(isfinite(a))||!all(isfinite(b))||!all(isfinite(pixel.xy)))return;
 float rigid[16];for(uint k=0;k<16;k++)rigid[k]=bestRigid[k];
 float3 difference=transformRigid(a.xyz,rigid)-b.xyz;if(dot(difference,difference)>=cfg.maxDistance*cfg.maxDistance)return;
 float x,y,z;caspar_transform(b.x,b.y,b.z,targetWorldPose[0],targetWorldPose[1],targetWorldPose[2],targetWorldPose[4],targetWorldPose[5],targetWorldPose[6],targetWorldPose[8],targetWorldPose[9],targetWorldPose[10],targetWorldPose[3],targetWorldPose[7],targetWorldPose[11],x,y,z);
 if(!all(isfinite(float3(x,y,z))))return;
 worldPoints[i*3]=x;worldPoints[i*3+1]=y;worldPoints[i*3+2]=z;
 observations[i*2]=pixel.x;observations[i*2+1]=pixel.y;validMask[i]=1;
}
[shader("compute")][numthreads(1,1,1)]
void finishVerification() {
 if(verificationStats[1]!=0||refinementStats[1]!=0||refinementStats[0]<cfg.minMatches){verificationStats[1]=1;return;}
 float cost=refinementCost[0];if(!isfinite(cost)||cost>cfg.maxRMSE*cfg.maxRMSE*float(refinementStats[0])){verificationStats[1]=1;return;}
 bool valid=true;for(uint k=0;k<16;k++)valid=valid&&isfinite(candidatePose[k]);if(!valid){verificationStats[1]=1;return;}
 for(uint k=0;k<16;k++)outputPose[k]=candidatePose[k];verificationStats[0]=refinementStats[0];verificationCost[0]=cost;
}
