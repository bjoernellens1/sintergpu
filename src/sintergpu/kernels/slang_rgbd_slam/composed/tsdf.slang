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

// Sparse block hash: signed 10-bit coordinates; EMPTY cannot be a valid key.
struct TSDFParams {
 uint width; uint height; float fx; float fy; float cx; float cy;
 float voxelSize; float mu; float maxWeight; float nearDepth; float farDepth;
 uint maxBlocks; uint hashCapacity; uint frameIndex; uint surfaceCapacity;
};
ConstantBuffer<TSDFParams> cfg;
RWStructuredBuffer<uint> hashKeys;
RWStructuredBuffer<uint> hashValues; // allocation id + 1; zero means unavailable
RWStructuredBuffer<uint> stamps;
RWStructuredBuffer<uint> blockKeys;
RWStructuredBuffer<uint> activeBlocks;
RWStructuredBuffer<uint> fusionArgs;
RWStructuredBuffer<uint> counters; // attempted allocations, overflow, active, surfaces, domain errors, probe exhaustion
RWStructuredBuffer<float> distances;
RWStructuredBuffer<float> weights;
RWStructuredBuffer<float> colorWeights;
RWStructuredBuffer<float> colors; // RGB normalized float triplets
StructuredBuffer<float> depth;
StructuredBuffer<float> rgb;
StructuredBuffer<float> pose;
StructuredBuffer<uint> trackingStatus;
RWStructuredBuffer<float> rayDepth;
RWStructuredBuffer<float> surfacePoints;
RWStructuredBuffer<float> surfaceColors;

float3 backproject(float u,float v,float z) {
 float x,y,w; caspar_backproject(u,v,z,cfg.fx,cfg.fy,cfg.cx,cfg.cy,x,y,w); return float3(x,y,w);
}
float3 worldPoint(float3 p) {
 float x,y,z; caspar_transform(p.x,p.y,p.z,pose[0],pose[1],pose[2],pose[4],pose[5],pose[6],pose[8],pose[9],pose[10],pose[3],pose[7],pose[11],x,y,z); return float3(x,y,z);
}
float3 cameraPoint(float3 p) {
 p-=float3(pose[3],pose[7],pose[11]);
 float x,y,z; caspar_transform(p.x,p.y,p.z,pose[0],pose[4],pose[8],pose[1],pose[5],pose[9],pose[2],pose[6],pose[10],0,0,0,x,y,z); return float3(x,y,z);
}
float2 projectPoint(float3 p) {
 float u,v; caspar_project(p.x,p.y,p.z,cfg.fx,cfg.fy,cfg.cx,cfg.cy,u,v); return float2(u,v);
}
uint encodeKey(int3 b) { return uint(b.x+512)|(uint(b.y+512)<<10)|(uint(b.z+512)<<20); }
int3 decodeKey(uint k) { return int3(int(k&1023)-512,int((k>>10)&1023)-512,int((k>>20)&1023)-512); }
bool validBlock(int3 b) { return all(b>=int3(-512)) && all(b<=int3(511)); }
uint hash(uint k) { k^=k>>16; k*=0x7feb352d; k^=k>>15; k*=0x846ca68b; return (k^(k>>16))&(cfg.hashCapacity-1); }
uint lookup(uint key) {
 uint slot=hash(key);
 for(uint i=0;i<min(cfg.hashCapacity,128u);i++) { uint k=hashKeys[slot]; if(k==key) return slot; if(k==0xffffffff) return 0xffffffff; slot=(slot+1)&(cfg.hashCapacity-1); }
 return 0xffffffff;
}
void allocate(int3 b) {
 if(!validBlock(b)) { uint ignored; InterlockedAdd(counters[4],1,ignored); return; }
 uint key=encodeKey(b),slot=hash(key);
 for(uint probe=0;probe<min(cfg.hashCapacity,128u);probe++) {
  uint existing=hashKeys[slot]; if(existing==key) return;
  if(existing==0xffffffff) {
   uint allocated;InterlockedAdd(counters[0],0,allocated);
   if(allocated>=cfg.maxBlocks) {uint ignored;InterlockedAdd(counters[1],1,ignored);return;}
  }
  uint previous; InterlockedCompareExchange(hashKeys[slot],0xffffffff,key,previous);
  if(previous==key) return; // Never spin on an unpublished value: next dispatch collects.
  if(previous==0xffffffff) {
   uint id; InterlockedAdd(counters[0],1,id);
   if(id<cfg.maxBlocks) { blockKeys[id]=key; hashValues[slot]=id+1; }
   else { uint ignored; InterlockedAdd(counters[1],1,ignored); }
   return;
  }
  slot=(slot+1)&(cfg.hashCapacity-1);
 }
 uint ignored; InterlockedAdd(counters[1],1,ignored);InterlockedAdd(counters[5],1,ignored);
}
void collect(int3 b) {
 if(!validBlock(b)) return;
 uint slot=lookup(encodeKey(b)); if(slot==0xffffffff || hashValues[slot]==0) return;
 uint previous; InterlockedExchange(stamps[slot],cfg.frameIndex,previous);
 if(previous!=cfg.frameIndex) { uint i; InterlockedAdd(counters[2],1,i); if(i<cfg.maxBlocks) activeBlocks[i]=hashValues[slot]-1; }
}
// Exact Amanatides-Woo traversal of the finite truncation segment.
void visitRay(uint pixel, bool allocating) {
 float z=depth[pixel]; if(!isfinite(z)||z<cfg.nearDepth||z>cfg.farDepth) return;
 float u=float(pixel%cfg.width),v=float(pixel/cfg.width);
 float3 a=worldPoint(backproject(u,v,max(cfg.nearDepth,z-cfg.mu)))/(cfg.voxelSize*8);
 float3 b=worldPoint(backproject(u,v,z+cfg.mu))/(cfg.voxelSize*8);
 float3 d=b-a; int3 cell=int3(floor(a)),last=int3(floor(b));
 int3 step=int3(d.x>0?1:(d.x<0?-1:0),d.y>0?1:(d.y<0?-1:0),d.z>0?1:(d.z<0?-1:0));
 float3 delta=float3(abs(d.x)>1e-20?abs(1/d.x):1e30,abs(d.y)>1e-20?abs(1/d.y):1e30,abs(d.z)>1e-20?abs(1/d.z):1e30);
 float3 boundary=float3(cell)+float3(step.x>0?1:0,step.y>0?1:0,step.z>0?1:0);
 float3 next=float3(step.x!=0?(boundary.x-a.x)/d.x:1e30,step.y!=0?(boundary.y-a.y)/d.y:1e30,step.z!=0?(boundary.z-a.z)/d.z:1e30);
 uint visits=uint(abs(last.x-cell.x)+abs(last.y-cell.y)+abs(last.z-cell.z))+1;
 for(uint n=0;n<visits;n++) {
  if(allocating) allocate(cell); else collect(cell);
  if(all(cell==last)) break;
  if(next.x<=next.y && next.x<=next.z) {cell.x+=step.x; next.x+=delta.x;}
  else if(next.y<=next.z) {cell.y+=step.y; next.y+=delta.y;}
  else {cell.z+=step.z; next.z+=delta.z;}
 }
}
[shader("compute")][numthreads(256,1,1)]
void initialize(uint3 tid:SV_DispatchThreadID) {
 if(trackingStatus[1]!=0)return;
 uint i=tid.x;
 if(i<cfg.hashCapacity) {hashKeys[i]=0xffffffff;hashValues[i]=0;stamps[i]=0;}
 if(i<cfg.maxBlocks*512) {distances[i]=1;weights[i]=0;colorWeights[i]=0;colors[i*3]=0;colors[i*3+1]=0;colors[i*3+2]=0;}
 if(i<6) counters[i]=0;
}
[shader("compute")][numthreads(1,1,1)]
void resetFrame(uint3 tid:SV_DispatchThreadID) { if(trackingStatus[1]==0) counters[2]=0; }
[shader("compute")][numthreads(256,1,1)]
void allocateBlocks(uint3 tid:SV_DispatchThreadID) {if(trackingStatus[1]==0 && tid.x<cfg.width*cfg.height) visitRay(tid.x,true);}
[shader("compute")][numthreads(256,1,1)]
void collectBlocks(uint3 tid:SV_DispatchThreadID) {if(trackingStatus[1]==0 && tid.x<cfg.width*cfg.height) visitRay(tid.x,false);}
[shader("compute")][numthreads(1,1,1)]
void prepareFusion(uint3 tid:SV_DispatchThreadID) {fusionArgs[0]=trackingStatus[1]==0?counters[2]*2:0;fusionArgs[1]=1;fusionArgs[2]=1;}
[shader("compute")][numthreads(256,1,1)]
void integrateVoxels(uint3 tid:SV_DispatchThreadID) {
 uint list=tid.x/512,local=tid.x%512;
 if(trackingStatus[1]!=0 || list>=counters[2]) return;
 uint block=activeBlocks[list],i=block*512+local;
 int3 v=decodeKey(blockKeys[block])*8+int3(local%8,(local/8)%8,local/64);
 float3 p=cameraPoint((float3(v)+.5)*cfg.voxelSize);
 if(p.z<=0) return; float2 uv=projectPoint(p); int2 pixel=int2(floor(uv+.5));
 if(any(pixel<0)||pixel.x>=int(cfg.width)||pixel.y>=int(cfg.height)) return;
 uint k=uint(pixel.y)*cfg.width+uint(pixel.x); float z=depth[k];
 if(!isfinite(z)||z<cfg.nearDepth||z>cfg.farDepth) return;
 float sdf=z-p.z; if(sdf < -cfg.mu) return;
 float w=1/(1+z*z),oldW=weights[i],nextD,nextW;
 caspar_tsdf_fuse(distances[i],oldW,min(1.,sdf/cfg.mu),w,cfg.maxWeight,nextD,nextW);
 distances[i]=nextD; weights[i]=nextW;
 if(abs(sdf)<=cfg.mu) {
  float oldColorW=colorWeights[i],nextColorW;
  for(uint c=0;c<3;c++) {float value;caspar_tsdf_fuse(colors[i*3+c],oldColorW,rgb[k*3+c],w,cfg.maxWeight,value,nextColorW);colors[i*3+c]=value;}
  colorWeights[i]=nextColorW;
 }
}
bool voxelColor(int3 v,out float value,out float3 color,out bool colorValid) {
 colorValid=false;
 int3 b=int3(floor(float3(v)/8)); if(!validBlock(b)) {value=1;color=0;return false;}
 uint slot=lookup(encodeKey(b)); if(slot==0xffffffff || hashValues[slot]==0) {value=1;color=0;return false;}
 int3 l=v-b*8; uint i=(hashValues[slot]-1)*512+uint(l.x+l.y*8+l.z*64);
 value=distances[i];color=float3(colors[i*3],colors[i*3+1],colors[i*3+2]);colorValid=colorWeights[i]>0;return weights[i]>0;
}
bool voxel(int3 v,out float value,out float3 color) {bool valid;return voxelColor(v,value,color,valid);}
RWStructuredBuffer<float> rayIntensity;
float sampleIntensity(float3 p) {
 float3 q=p/cfg.voxelSize-.5;int3 base=int3(floor(q));float3 f=q-float3(base);float3 result=0;
 for(uint z=0;z<2;z++) for(uint y=0;y<2;y++) for(uint x=0;x<2;x++) {
  float d;float3 color;bool colorValid;if(!voxelColor(base+int3(x,y,z),d,color,colorValid)||!colorValid) return -1;
  result+=color*(x==0?1-f.x:f.x)*(y==0?1-f.y:f.y)*(z==0?1-f.z:f.z);
 }
 return dot(result,float3(.299,.587,.114))/255.;
}
bool sampleTSDF(float3 p,out float value) {
 float3 q=p/cfg.voxelSize-.5; int3 base=int3(floor(q));float3 f=q-float3(base);value=0;
 for(uint z=0;z<2;z++) for(uint y=0;y<2;y++) for(uint x=0;x<2;x++) {
  float d;float3 color;if(!voxel(base+int3(x,y,z),d,color)) return false;
  value+=d*(x==0?1-f.x:f.x)*(y==0?1-f.y:f.y)*(z==0?1-f.z:f.z);
 }
 return true;
}
[shader("compute")][numthreads(16,16,1)]
void raycast(uint3 tid:SV_DispatchThreadID) {
 if(tid.x>=cfg.width || tid.y>=cfg.height) return;
 uint pixel=tid.y*cfg.width+tid.x;rayDepth[pixel]=0;rayIntensity[pixel]=-1;
 float3 origin=worldPoint(float3(0)), direction=worldPoint(backproject(float(tid.x),float(tid.y),1))-origin;
 float t=cfg.nearDepth,previous=0,previousT=t;bool havePrevious=false;
 // Bound based on near/far and smallest voxel step; no fixed scene-dependent cutoff.
 float rayLength=(cfg.farDepth-cfg.nearDepth)*length(direction);
 uint limit=uint(ceil(rayLength/(cfg.voxelSize*.5)))+uint(ceil(rayLength/(cfg.voxelSize*8)))*3+16;
 for(uint n=0;n<limit && t<=cfg.farDepth;n++) {
  float3 p=origin+direction*t; int3 b=int3(floor(p/(cfg.voxelSize*8)));
  uint slot=validBlock(b)?lookup(encodeKey(b)):0xffffffff;
  if(slot==0xffffffff || hashValues[slot]==0) {
   float3 cell=float3(b),bound=(cell+float3(direction.x>0?1:0,direction.y>0?1:0,direction.z>0?1:0))*cfg.voxelSize*8;
   float3 crossing=float3(abs(direction.x)>1e-20?(bound.x-origin.x)/direction.x:1e30,abs(direction.y)>1e-20?(bound.y-origin.y)/direction.y:1e30,abs(direction.z)>1e-20?(bound.z-origin.z)/direction.z:1e30);
   t=max(t+1e-5,min(crossing.x,min(crossing.y,crossing.z))+1e-5);havePrevious=false;continue;
  }
  float value;bool valid=sampleTSDF(p,value);
  if(valid && havePrevious && previous>0 && value<=0) {rayDepth[pixel]=previousT+(t-previousT)*previous/(previous-value);rayIntensity[pixel]=sampleIntensity(origin+direction*rayDepth[pixel]);return;}
  havePrevious=valid;previous=value;previousT=t;
  t+=cfg.voxelSize*.5/length(direction);
 }
}
[shader("compute")][numthreads(1,1,1)]
void resetSurface(uint3 tid:SV_DispatchThreadID) {counters[3]=0;}
[shader("compute")][numthreads(256,1,1)]
void extractSurface(uint3 tid:SV_DispatchThreadID) {
 uint block=tid.x/512,local=tid.x%512;if(block>=min(counters[0],cfg.maxBlocks)) return;
 int3 v=decodeKey(blockKeys[block])*8+int3(local%8,(local/8)%8,local/64);
 float a;float3 ca;if(!voxel(v,a,ca)) return;
 for(uint axis=0;axis<3;axis++) {
  int3 offset=int3(axis==0?1:0,axis==1?1:0,axis==2?1:0);float b;float3 cb;
  if(!voxel(v+offset,b,cb)||a*b>0||a==b) continue;
  float alpha=a/(a-b);float3 p=(float3(v)+.5+float3(offset)*alpha)*cfg.voxelSize;
  uint i;InterlockedAdd(counters[3],1,i);if(i>=cfg.surfaceCapacity) continue;
  float3 c=lerp(ca,cb,alpha);for(uint j=0;j<3;j++){surfacePoints[i*3+j]=p[j];surfaceColors[i*3+j]=c[j];}
 }
}

