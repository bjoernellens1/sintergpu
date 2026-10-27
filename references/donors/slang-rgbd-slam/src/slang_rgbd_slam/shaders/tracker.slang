struct MapParams { uint width; uint height; float fx; float fy; float cx; float cy; float depthJump; };
ConstantBuffer<MapParams> mapParams;
StructuredBuffer<float> inputDepth;
RWStructuredBuffer<float> outputDepth;
RWStructuredBuffer<float4> vertices;
RWStructuredBuffer<float4> normals;

[shader("compute")]
[numthreads(16,16,1)]
void downsample(uint3 tid : SV_DispatchThreadID) {
    uint x=tid.x, y=tid.y;
    if(x>=mapParams.width || y>=mapParams.height) return;
    uint w=mapParams.width*2;
    float z[4];
    z[0]=inputDepth[(y*2)*w+x*2]; z[1]=inputDepth[(y*2)*w+x*2+1];
    z[2]=inputDepth[(y*2+1)*w+x*2]; z[3]=inputDepth[(y*2+1)*w+x*2+1];
    float sum=0, minimum=1e20, maximum=0; uint count=0;
    for(uint k=0;k<4;k++) if(z[k]>0) {sum+=z[k]; count++; minimum=min(minimum,z[k]); maximum=max(maximum,z[k]);}
    outputDepth[y*mapParams.width+x]=(count>0 && maximum-minimum<=mapParams.depthJump) ? sum/count : 0;
}
[shader("compute")]
[numthreads(16,16,1)]
void makeVertices(uint3 tid : SV_DispatchThreadID) {
    if(tid.x>=mapParams.width || tid.y>=mapParams.height) return;
    uint i=tid.y*mapParams.width+tid.x; float z=inputDepth[i];
    float x,y,pz;
    caspar_backproject(float(tid.x),float(tid.y),z,mapParams.fx,mapParams.fy,mapParams.cx,mapParams.cy,x,y,pz);
    vertices[i]=z>0 ? float4(x,y,pz,1) : float4(0);
}
[shader("compute")]
[numthreads(16,16,1)]
void makeNormals(uint3 tid : SV_DispatchThreadID) {
    if(tid.x>=mapParams.width || tid.y>=mapParams.height) return;
    uint i=tid.y*mapParams.width+tid.x; float4 n=0;
    if(tid.x>0 && tid.y>0 && tid.x+1<mapParams.width && tid.y+1<mapParams.height) {
        float4 a=vertices[i-1], b=vertices[i+1], c=vertices[i-mapParams.width], d=vertices[i+mapParams.width];
        if(a.w*b.w*c.w*d.w>0 && abs(a.z-b.z)<mapParams.depthJump && abs(c.z-d.z)<mapParams.depthJump) {
            float3 crossValue=cross(b.xyz-a.xyz,d.xyz-c.xyz); float lengthValue=length(crossValue);
            if(lengthValue>1e-10) n=float4(crossValue/lengthValue,1);
        }
    }
    normals[i]=n;
}
struct TrackParams { uint count; uint width; uint height; float fx; float fy; float cx; float cy; float distance; float normalCos; float huber; uint minCorrespondences; };
ConstantBuffer<TrackParams> trackParams;
StructuredBuffer<float4> sourceVertices;
StructuredBuffer<float4> sourceNormals;
StructuredBuffer<float4> referenceVertices;
StructuredBuffer<float4> referenceNormals;
RWStructuredBuffer<float4> points;
RWStructuredBuffer<float4> targets;
RWStructuredBuffer<float4> weightedNormals;
RWStructuredBuffer<float> pose;
StructuredBuffer<float> initialPose;
RWStructuredBuffer<uint> trackingStatus; // correspondence count, persistent failure
RWStructuredBuffer<float> diagnostics; // weighted sum of residual squares
StructuredBuffer<float> delta;
StructuredBuffer<uint> solveStatus;
StructuredBuffer<float> normalTerms;
float3 rotatePose(float3 p) {
    float x,y,z;
    caspar_transform(p.x,p.y,p.z,pose[0],pose[1],pose[2],pose[4],pose[5],pose[6],pose[8],pose[9],pose[10],0,0,0,x,y,z);
    return float3(x,y,z);
}
[shader("compute")]
[numthreads(1,1,1)]
void resetCount() { trackingStatus[0]=0; }
[shader("compute")]
[numthreads(64,1,1)]
void associate(uint3 tid : SV_DispatchThreadID) {
    uint i=tid.x; bool valid=false;
    if(i<trackParams.count) {
        points[i]=0; targets[i]=0; weightedNormals[i]=0;
        float4 source=sourceVertices[i], sn=sourceNormals[i];
        float3 p=rotatePose(source.xyz)+float3(pose[3],pose[7],pose[11]);
        if(source.w>0 && sn.w>0 && p.z>0 && trackingStatus[1]==0) {
            float u,v;
            caspar_project(p.x,p.y,p.z,trackParams.fx,trackParams.fy,trackParams.cx,trackParams.cy,u,v);
            int2 pixel=int2(round(float2(u,v)));
            if(all(pixel>=0) && pixel.x<int(trackParams.width) && pixel.y<int(trackParams.height)) {
                uint j=uint(pixel.y)*trackParams.width+uint(pixel.x);
                float4 q=referenceVertices[j], n=referenceNormals[j];
                valid=q.w>0 && n.w>0 && length(p-q.xyz)<trackParams.distance && dot(rotatePose(sn.xyz),n.xyz)>trackParams.normalCos;
                if(valid) {
                    float r=dot(n.xyz,p-q.xyz); float weight=min(1.0,trackParams.huber/max(abs(r),1e-20));
                    points[i]=float4(p,0); targets[i]=float4(q.xyz,0); weightedNormals[i]=float4(n.xyz*sqrt(weight),0);
                }
            }
        }
    }
    uint count=WaveActiveCountBits(valid);
    if(WaveIsFirstLane()) InterlockedAdd(trackingStatus[0],count);
}
[shader("compute")]
[numthreads(1,1,1)]
void updatePose() {
    diagnostics[0]=normalTerms[42];
    if(trackingStatus[0]<trackParams.minCorrespondences || solveStatus[0]!=0) trackingStatus[1]=1;
    if(trackingStatus[1]!=0) {for(uint k=0;k<16;k++) pose[k]=initialPose[k]; return;}
    float3 w=float3(delta[0],delta[1],delta[2]), v=float3(delta[3],delta[4],delta[5]);
    float a,b,c;
    caspar_se3_coefficients(dot(w,w),a,b,c);
    float oldPose[16]; float updatedPose[16];
    for(uint k=0;k<16;k++) {oldPose[k]=pose[k]; updatedPose[k]=pose[k];}
    for(uint col=0;col<4;col++) {
        float3 p=float3(oldPose[col],oldPose[4+col],oldPose[8+col]);
        float x,y,z;
        caspar_twist_point(w.x,w.y,w.z,v.x,v.y,v.z,a,b,c,p.x,p.y,p.z,col==3 ? 1.0 : 0.0,x,y,z);
        p=float3(x,y,z);
        updatedPose[col]=p.x; updatedPose[4+col]=p.y; updatedPose[8+col]=p.z;
    }
    bool finitePose=true;
    for(uint k=0;k<16;k++) finitePose=finitePose && isfinite(updatedPose[k]);
    if(!finitePose) trackingStatus[1]=1;
    for(uint k=0;k<16;k++) pose[k]=finitePose ? updatedPose[k] : initialPose[k];
}
