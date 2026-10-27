RWStructuredBuffer<float> worldPose;
StructuredBuffer<float> referenceWorldPose;
RWStructuredBuffer<float> relativePose;
RWStructuredBuffer<float> initialPose;
RWStructuredBuffer<uint> trackingStatus;
[shader("compute")]
[numthreads(1,1,1)]
void resetRelative() {
    for(uint i=0;i<16;i++) { float v=(i%5==0) ? 1.0 : 0.0; relativePose[i]=v; initialPose[i]=v; }
    trackingStatus[0]=0; trackingStatus[1]=0;
}
[shader("compute")]
[numthreads(1,1,1)]
void composeWorld() {
    if(trackingStatus[1]!=0) return;
    for(uint col=0;col<4;col++) {
        float flag=col==3 ? 1.0 : 0.0;
        float x,y,z;
        caspar_transform(relativePose[col],relativePose[4+col],relativePose[8+col],
            referenceWorldPose[0],referenceWorldPose[1],referenceWorldPose[2],
            referenceWorldPose[4],referenceWorldPose[5],referenceWorldPose[6],
            referenceWorldPose[8],referenceWorldPose[9],referenceWorldPose[10],
            flag*referenceWorldPose[3],flag*referenceWorldPose[7],flag*referenceWorldPose[11],x,y,z);
        worldPose[col]=x; worldPose[4+col]=y; worldPose[8+col]=z;
    }
}
