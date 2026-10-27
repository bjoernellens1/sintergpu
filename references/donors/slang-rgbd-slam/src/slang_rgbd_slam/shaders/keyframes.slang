struct KFParams {uint width;uint height;uint lowWidth;uint lowHeight;uint maxFeatures;uint frameRow;uint latest;uint frameIndex;uint slot;uint capacity;uint minLoopFrames;float fx;float fy;float cx;float cy;float retrievalThreshold;};
ConstantBuffer<KFParams> kf;
StructuredBuffer<float> frameDepth;
StructuredBuffer<float> frameRGB;
StructuredBuffer<float> lowDepth;
StructuredBuffer<float> lowIntensity;
StructuredBuffer<float> worldPose;
StructuredBuffer<uint> trackingStatus;
RWStructuredBuffer<float> savedDepth;
RWStructuredBuffer<float> savedRawDepth;
RWStructuredBuffer<float> savedMeasuredLowDepth;
RWStructuredBuffer<float> savedRGB;
RWStructuredBuffer<float> savedLowDepth;
RWStructuredBuffer<float> savedLowRGB;
RWStructuredBuffer<float> savedPose;
StructuredBuffer<float4> featurePositions;
StructuredBuffer<float4> featureLandmarks;
StructuredBuffer<uint> featureDescriptors;
StructuredBuffer<uint> featureCount;
RWStructuredBuffer<float4> savedPositions;
RWStructuredBuffer<float4> savedLandmarks;
RWStructuredBuffer<uint> savedDescriptors;
RWStructuredBuffer<uint> savedCount;
RWStructuredBuffer<uint> frameIds;
[shader("compute")][numthreads(256,1,1)]
void captureImages(uint3 tid:SV_DispatchThreadID) {
 if(trackingStatus[1]!=0)return;uint i=tid.x;
 if(i<kf.width*kf.height){savedDepth[i]=frameDepth[i];savedRawDepth[i]=frameDepth[i];for(uint c=0;c<3;c++)savedRGB[i*3+c]=frameRGB[i*3+c];}
 if(i<kf.lowWidth*kf.lowHeight){savedLowDepth[i]=lowDepth[i];savedMeasuredLowDepth[i]=lowDepth[i];for(uint c=0;c<3;c++)savedLowRGB[i*3+c]=lowIntensity[i]*255.;}
 if(i<16)savedPose[i]=worldPose[i];if(i==0)frameIds[kf.slot]=kf.frameIndex;
}
[shader("compute")][numthreads(256,1,1)]
void captureFeatures(uint3 tid:SV_DispatchThreadID) {
 if(trackingStatus[1]!=0)return;uint i=tid.x;
 if(i<kf.maxFeatures){savedPositions[i]=featurePositions[i];savedLandmarks[i]=featureLandmarks[i];for(uint c=0;c<8;c++)savedDescriptors[i*8+c]=featureDescriptors[i*8+c];}
 if(i==0)savedCount[0]=featureCount[0];
}
StructuredBuffer<float> baPoses;
StructuredBuffer<float> baInverseDepth;
StructuredBuffer<uint> baStatus;
RWStructuredBuffer<float> currentWorldPose;
RWStructuredBuffer<uint> rebuildGuard;
[shader("compute")][numthreads(1,1,1)]
void makeBAGuard(){rebuildGuard[0]=0;rebuildGuard[1]=(trackingStatus[1]!=0||baStatus[0]!=0)?1:0;}
[shader("compute")][numthreads(256,1,1)]
void commitBA(uint3 tid:SV_DispatchThreadID) {
 if(rebuildGuard[1]!=0)return;uint i=tid.x;
 if(i<16){savedPose[i]=baPoses[kf.frameRow*16+i];if(kf.latest!=0)currentWorldPose[i]=savedPose[i];}
 if(i<kf.lowWidth*kf.lowHeight){uint p=kf.frameRow*kf.lowWidth*kf.lowHeight+i;float z=savedMeasuredLowDepth[i],rho=baInverseDepth[p];if(z>0&&rho>0&&isfinite(rho))savedLowDepth[i]=1/rho;}
 if(i<kf.width*kf.height){uint x=i%kf.width,y=i/kf.width,lx=x*kf.lowWidth/kf.width,ly=y*kf.lowHeight/kf.height;
  uint lowPixel=ly*kf.lowWidth+lx,p=kf.frameRow*kf.lowWidth*kf.lowHeight+lowPixel;float z=savedMeasuredLowDepth[lowPixel],rho=baInverseDepth[p];
  if(z>0&&rho>0&&isfinite(rho))savedDepth[i]=savedRawDepth[i]/(z*rho);
 }

}
[shader("compute")][numthreads(256,1,1)]
void refreshLandmarks(uint3 tid:SV_DispatchThreadID) {
 if(rebuildGuard[1]!=0)return;uint i=tid.x;if(i>=min(savedCount[0],kf.maxFeatures))return;
 float4 p=savedPositions[i];int2 pixel=int2(floor(p.xy+.5));float z=savedDepth[uint(pixel.y)*kf.width+uint(pixel.x)];float x,y,pz;
 caspar_backproject(p.x,p.y,z,kf.fx,kf.fy,kf.cx,kf.cy,x,y,pz);savedPositions[i].z=z;savedLandmarks[i]=float4(x,y,pz,z>0?1:0);
}
RWStructuredBuffer<float> histograms;
groupshared uint bins[256];
[shader("compute")][numthreads(256,1,1)]
void featureHistogram(uint3 tid:SV_DispatchThreadID,uint3 lane:SV_GroupThreadID) {
 bins[lane.x]=0;GroupMemoryBarrierWithGroupSync();
 for(uint i=lane.x;i<min(featureCount[0],kf.maxFeatures);i+=256){uint bucket=featureDescriptors[i*8]&255;InterlockedAdd(bins[bucket],1);}
 GroupMemoryBarrierWithGroupSync();histograms[kf.slot*256+lane.x]=float(bins[lane.x]);
}
StructuredBuffer<float> queryHistogram;
StructuredBuffer<float> archiveHistograms;
StructuredBuffer<uint> archiveFrameIds;
RWStructuredBuffer<float> retrievalScores;
RWStructuredBuffer<uint> retrievalEvent;
[shader("compute")][numthreads(64,1,1)]
void retrieveKeyframes(uint3 tid:SV_DispatchThreadID) {
 uint slot=tid.x;if(slot>=kf.capacity)return;uint age=archiveFrameIds[slot];float score=-1;
 if(age!=0&&kf.frameIndex>=age+kf.minLoopFrames){float a=0,b=0,c=0;
  for(uint i=0;i<256;i++){float q=queryHistogram[i],v=archiveHistograms[slot*256+i];a+=q*v;b+=q*q;c+=v*v;}if(b*c>0)score=a/sqrt(b*c);
 }
 retrievalScores[slot]=score;
}
[shader("compute")][numthreads(1,1,1)]
void selectCandidate(){uint best=0xffffffff;float score=kf.retrievalThreshold;for(uint i=0;i<kf.capacity;i++)if(retrievalScores[i]>score){best=i;score=retrievalScores[i];}retrievalEvent[0]=best;retrievalEvent[1]=asuint(score);}

RWStructuredBuffer<uint> tailGuard;
[shader("compute")][numthreads(1,1,1)]
void makeTailGuard(){tailGuard[0]=0;tailGuard[1]=(trackingStatus[1]!=0||rebuildGuard[1]==0)?1:0;}
