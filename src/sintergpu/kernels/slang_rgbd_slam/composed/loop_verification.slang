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

