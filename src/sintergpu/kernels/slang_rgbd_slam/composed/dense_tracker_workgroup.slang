#define DENSE_GROUP_SIZE 64
#define DENSE_USE_WAVE 0
#define DENSE_MIN_WAVE_SIZE 1
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

// SymForce/Caspar DABSEG kernel: point_plane
void caspar_point_plane(float px, float py, float pz, float qx, float qy, float qz, float nx, float ny, float nz, out float out_0, out float out_1, out float out_2, out float out_3, out float out_4, out float out_5, out float out_6)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f;
    r0 = nz;
    r1 = pz;
    r2 = qz;
    r3 = -1.00000000000000000e+00;
    r2 = fma(r2, r3, r1);
    r4 = ny;
    r5 = py;
    r6 = qy;
    r6 = fma(r6, r3, r5);
    r6 = fma(r4, r6, r0 * r2);
    r2 = nx;
    r7 = px;
    r8 = qx;
    r8 = fma(r8, r3, r7);
    r6 = fma(r2, r8, r6);
    out_0 = r6;
    r6 = r1 * r4;
    r6 = fma(r3, r6, r5 * r0);
    out_1 = r6;
    r6 = r7 * r0;
    r6 = fma(r1, r2, r3 * r6);
    out_2 = r6;
    r6 = r5 * r2;
    r6 = fma(r3, r6, r7 * r4);
    out_3 = r6;
    out_4 = r2;
    out_5 = r4;
    out_6 = r0;
}

// SymForce/Caspar DABSEG kernel: photo_factor
void caspar_photo_factor(float x, float y, float z, float fx, float fy, float gx, float gy, float targetI, float sourceI, out float out_0, out float out_1, out float out_2, out float out_3, out float out_4, out float out_5, out float out_6)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f, r9 = 0.0f, r10 = 0.0f, r11 = 0.0f, r12 = 0.0f;
    r0 = targetI;
    r1 = sourceI;
    r2 = -1.00000000000000000e+00;
    r1 = fma(r1, r2, r0);
    out_0 = r1;
    r1 = fx;
    r0 = gx;
    r3 = r1 * r0;
    r4 = x;
    r5 = z;
    r6 = r5 * r5;
    r6 = 1.0 / r6;
    r7 = r4 * r6;
    r8 = r3 * r7;
    r9 = y;
    r10 = r9 * r2;
    r11 = fy;
    r12 = gy;
    r12 = r11 * r12;
    r6 = r6 * r10;
    r11 = fma(r9, r6, r2);
    r11 = fma(r11, r12, r10 * r8);
    out_1 = r11;
    r11 = r9 * r12;
    r8 = 1.00000000000000000e+00;
    r8 = fma(r4, r7, r8);
    r8 = fma(r8, r3, r7 * r11);
    out_2 = r8;
    r5 = 1.0 / r5;
    r8 = r4 * r5;
    r0 = r1 * r0;
    r0 = r0 * r5;
    r10 = fma(r10, r0, r12 * r8);
    out_3 = r10;
    out_4 = r0;
    r0 = r5 * r12;
    out_5 = r0;
    r0 = r2 * r3;
    r6 = fma(r12, r6, r7 * r0);
    out_6 = r6;
}

// SymForce/Caspar DABSEG kernel: dense_factor
void caspar_dense_factor(float wx, float wy, float wz, float x, float y, float z, float tr00, float tr01, float tr02, float tr10, float tr11, float tr12, float tr20, float tr21, float tr22, float sr00, float sr01, float sr02, float sr10, float sr11, float sr12, float sr20, float sr21, float sr22, float rx, float ry, float rz, float rho, float qx, float qy, float qz, float nx, float ny, float nz, float targetIntensity, float sourceIntensity, float gx, float gy, float fx, float fy, out float out_0, out float out_1, out float out_2, out float out_3, out float out_4, out float out_5, out float out_6, out float out_7, out float out_8, out float out_9, out float out_10, out float out_11, out float out_12, out float out_13, out float out_14, out float out_15)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f, r9 = 0.0f, r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f, r14 = 0.0f, r15 = 0.0f, r16 = 0.0f, r17 = 0.0f, r18 = 0.0f, r19 = 0.0f, r20 = 0.0f, r21 = 0.0f, r22 = 0.0f, r23 = 0.0f, r24 = 0.0f, r25 = 0.0f, r26 = 0.0f, r27 = 0.0f, r28 = 0.0f, r29 = 0.0f, r30 = 0.0f, r31 = 0.0f, r32 = 0.0f, r33 = 0.0f, r34 = 0.0f;
    r0 = ny;
    r1 = y;
    r2 = qy;
    r3 = -1.00000000000000000e+00;
    r2 = fma(r2, r3, r1);
    r4 = nx;
    r5 = x;
    r6 = qx;
    r6 = fma(r6, r3, r5);
    r6 = fma(r4, r6, r0 * r2);
    r2 = nz;
    r7 = z;
    r8 = qz;
    r8 = fma(r8, r3, r7);
    r6 = fma(r2, r8, r6);
    out_0 = r6;
    r6 = targetIntensity;
    r8 = sourceIntensity;
    r8 = fma(r8, r3, r6);
    out_1 = r8;
    r8 = wz;
    r6 = tr01;
    r9 = r8 * r6;
    r10 = wy;
    r11 = tr02;
    r9 = fma(r10, r11, r3 * r9);
    r12 = tr22;
    r13 = tr21;
    r14 = r8 * r13;
    r14 = fma(r3, r14, r10 * r12);
    r15 = fma(r2, r14, r4 * r9);
    r16 = tr12;
    r17 = tr11;
    r18 = r8 * r17;
    r18 = fma(r3, r18, r10 * r16);
    r15 = fma(r0, r18, r15);
    out_2 = r15;
    r15 = wx;
    r19 = r15 * r11;
    r20 = tr00;
    r19 = fma(r8, r20, r3 * r19);
    r21 = r15 * r16;
    r22 = tr10;
    r21 = fma(r8, r22, r3 * r21);
    r23 = fma(r0, r21, r4 * r19);
    r24 = r15 * r12;
    r25 = tr20;
    r24 = fma(r8, r25, r3 * r24);
    r23 = fma(r2, r24, r23);
    out_3 = r23;
    r23 = r10 * r25;
    r23 = fma(r15, r13, r3 * r23);
    r26 = r10 * r22;
    r26 = fma(r15, r17, r3 * r26);
    r27 = fma(r0, r26, r2 * r23);
    r28 = r10 * r20;
    r28 = fma(r3, r28, r15 * r6);
    r27 = fma(r4, r28, r27);
    out_4 = r27;
    r27 = fma(r22, r0, r20 * r4);
    r27 = fma(r25, r2, r27);
    out_5 = r27;
    r27 = fma(r17, r0, r6 * r4);
    r27 = fma(r13, r2, r27);
    out_6 = r27;
    r27 = fma(r16, r0, r11 * r4);
    r27 = fma(r12, r2, r27);
    out_7 = r27;
    r27 = gx;
    r29 = fx;
    r29 = r27 * r29;
    r27 = 1.0 / r7;
    r7 = r7 * r7;
    r7 = 1.0 / r7;
    r7 = r3 * r7;
    r5 = r5 * r7;
    r9 = fma(r14, r5, r9 * r27);
    r30 = gy;
    r31 = fy;
    r31 = r30 * r31;
    r30 = r1 * r14;
    r30 = fma(r7, r30, r18 * r27);
    r30 = fma(r30, r31, r9 * r29);
    out_8 = r30;
    r19 = fma(r24, r5, r19 * r27);
    r30 = r1 * r24;
    r30 = fma(r7, r30, r21 * r27);
    r30 = fma(r30, r31, r19 * r29);
    out_9 = r30;
    r30 = r1 * r23;
    r30 = fma(r7, r30, r26 * r27);
    r28 = fma(r23, r5, r28 * r27);
    r28 = fma(r28, r29, r30 * r31);
    out_10 = r28;
    r28 = r1 * r25;
    r28 = fma(r7, r28, r22 * r27);
    r30 = fma(r25, r5, r20 * r27);
    r30 = fma(r30, r29, r28 * r31);
    out_11 = r30;
    r30 = r1 * r13;
    r30 = fma(r7, r30, r17 * r27);
    r28 = fma(r13, r5, r6 * r27);
    r28 = fma(r28, r29, r30 * r31);
    out_12 = r28;
    r28 = r1 * r12;
    r28 = fma(r7, r28, r16 * r27);
    r30 = fma(r12, r5, r11 * r27);
    r30 = fma(r30, r29, r28 * r31);
    out_13 = r30;
    r30 = sr00;
    r28 = rx;
    r26 = sr01;
    r19 = ry;
    r26 = fma(r26, r19, r30 * r28);
    r30 = sr02;
    r21 = rz;
    r26 = fma(r30, r21, r26);
    r26 = r3 * r26;
    r30 = rho;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r26 = r26 * r30;
    r9 = r17 * r3;
    r18 = sr12;
    r32 = sr10;
    r32 = fma(r32, r28, r18 * r21);
    r18 = sr11;
    r32 = fma(r18, r19, r32);
    r9 = r9 * r32;
    r9 = fma(r30, r9, r22 * r26);
    r18 = r16 * r3;
    r33 = sr22;
    r34 = sr20;
    r28 = fma(r34, r28, r33 * r21);
    r34 = sr21;
    r28 = fma(r34, r19, r28);
    r18 = r18 * r28;
    r9 = fma(r30, r18, r9);
    r18 = r13 * r3;
    r18 = r18 * r32;
    r18 = fma(r30, r18, r25 * r26);
    r19 = r12 * r3;
    r19 = r19 * r28;
    r18 = fma(r30, r19, r18);
    r2 = fma(r2, r18, r0 * r9);
    r0 = r6 * r3;
    r0 = r0 * r32;
    r0 = fma(r30, r0, r20 * r26);
    r26 = r11 * r3;
    r26 = r26 * r28;
    r0 = fma(r30, r26, r0);
    r2 = fma(r4, r0, r2);
    out_14 = r2;
    r5 = fma(r18, r5, r0 * r27);
    r0 = r1 * r18;
    r0 = fma(r7, r0, r9 * r27);
    r31 = fma(r0, r31, r5 * r29);
    out_15 = r31;
}

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

#ifndef DENSE_GROUP_SIZE
#define DENSE_GROUP_SIZE 64
#endif
#ifndef DENSE_USE_WAVE
#define DENSE_USE_WAVE 1
#endif
#ifndef DENSE_MIN_WAVE_SIZE
#define DENSE_MIN_WAVE_SIZE 1
#endif
struct PhotoParams {float weight;float huber;};
ConstantBuffer<PhotoParams> photoParams;
StructuredBuffer<float> sourceIntensity;
StructuredBuffer<float> referenceIntensity;
StructuredBuffer<float> rgbImage;
RWStructuredBuffer<float> intensityOutput;
RWStructuredBuffer<float> outputTerms;
#if DENSE_USE_WAVE
groupshared float denseTerms[43][(DENSE_GROUP_SIZE+DENSE_MIN_WAVE_SIZE-1)/DENSE_MIN_WAVE_SIZE];
#else
groupshared float denseTerms[43][DENSE_GROUP_SIZE];
#endif
#if !DENSE_USE_WAVE
groupshared uint denseValid[DENSE_GROUP_SIZE];
#endif
[shader("compute")][numthreads(256,1,1)]
void rgbIntensity(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=mapParams.width*mapParams.height)return;
 intensityOutput[i]=dot(float3(rgbImage[i*3],rgbImage[i*3+1],rgbImage[i*3+2]),float3(.299,.587,.114))/255.;
}
[shader("compute")][numthreads(16,16,1)]
void downsampleIntensity(uint3 tid:SV_DispatchThreadID) {
 uint x=tid.x,y=tid.y;if(x>=mapParams.width||y>=mapParams.height)return;uint w=mapParams.width*2;
 float a=sourceIntensity[y*2*w+x*2],b=sourceIntensity[y*2*w+x*2+1],c=sourceIntensity[(y*2+1)*w+x*2],d=sourceIntensity[(y*2+1)*w+x*2+1];
 intensityOutput[y*mapParams.width+x]=min(min(a,b),min(c,d))<0?-1:(a+b+c+d)*.25;
}
float intensityAt(float2 uv) {
 int2 p=int2(floor(uv));float2 f=uv-float2(p);uint i=uint(p.y)*trackParams.width+uint(p.x);
 float a=referenceIntensity[i],b=referenceIntensity[i+1],c=referenceIntensity[i+trackParams.width],d=referenceIntensity[i+trackParams.width+1];
 if(min(min(a,b),min(c,d))<0)return -1;
 return lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y);
}
[shader("compute")][numthreads(DENSE_GROUP_SIZE,1,1)]
void fusedAssociate(uint3 tid:SV_DispatchThreadID,uint3 lane:SV_GroupThreadID,uint3 group:SV_GroupID) {
 uint i=tid.x;bool valid=false;float r=0,pr=0;float j[6],pj[6];
 for(uint c=0;c<6;c++){j[c]=0;pj[c]=0;}
 if(i<trackParams.count && trackingStatus[1]==0) {
  float4 source=sourceVertices[i],sn=sourceNormals[i];float3 p=rotatePose(source.xyz)+float3(pose[3],pose[7],pose[11]);
  if(source.w>0 && sn.w>0 && p.z>0) {
   float u,v;caspar_project(p.x,p.y,p.z,trackParams.fx,trackParams.fy,trackParams.cx,trackParams.cy,u,v);float2 uv=float2(u,v);int2 pixel=int2(round(uv));
   if(all(pixel>=0)&&pixel.x<int(trackParams.width)&&pixel.y<int(trackParams.height)) {
    uint k=uint(pixel.y)*trackParams.width+uint(pixel.x);float4 q=referenceVertices[k],n=referenceNormals[k];
    valid=q.w>0&&n.w>0&&length(p-q.xyz)<trackParams.distance&&dot(rotatePose(sn.xyz),n.xyz)>trackParams.normalCos;
    if(valid) {
     caspar_point_plane(p.x,p.y,p.z,q.x,q.y,q.z,n.x,n.y,n.z,r,j[0],j[1],j[2],j[3],j[4],j[5]);
     float weight=sqrt(min(1.,trackParams.huber/max(abs(r),1e-20)));r*=weight;for(uint c=0;c<6;c++)j[c]*=weight;
     if(photoParams.weight>0 && all(uv>=float2(2)) && u<float(trackParams.width)-3 && v<float(trackParams.height)-3) {
      float ti=intensityAt(uv),xp=intensityAt(uv+float2(1,0)),xm=intensityAt(uv-float2(1,0)),yp=intensityAt(uv+float2(0,1)),ym=intensityAt(uv-float2(0,1));
      if(ti>=0&&xp>=0&&xm>=0&&yp>=0&&ym>=0) {
       float gx=(xp-xm)*.5,gy=(yp-ym)*.5;
       caspar_photo_factor(p.x,p.y,p.z,trackParams.fx,trackParams.fy,gx,gy,ti,sourceIntensity[i],pr,pj[0],pj[1],pj[2],pj[3],pj[4],pj[5]);
       float pw=photoParams.weight*sqrt(min(1.,photoParams.huber/max(abs(pr),1e-20)));pr*=pw;for(uint c=0;c<6;c++)pj[c]*=pw;
      }
     }
    }
   }
  }
 }
#if DENSE_USE_WAVE
 uint count=WaveActiveCountBits(valid);if(WaveIsFirstLane())InterlockedAdd(trackingStatus[0],count);
#else
 denseValid[lane.x]=valid?1:0;
#endif
 for(uint t=0;t<43;t++) {
  float value=t<36?j[t/6]*j[t%6]+pj[t/6]*pj[t%6]:(t<42?j[t-36]*r+pj[t-36]*pr:r*r+pr*pr);
#if DENSE_USE_WAVE
  float sum=WaveActiveSum(value);if(WaveIsFirstLane())denseTerms[t][lane.x/WaveGetLaneCount()]=sum;
#else
  denseTerms[t][lane.x]=value;
#endif
 }
 GroupMemoryBarrierWithGroupSync();
#if DENSE_USE_WAVE
 uint waves=(DENSE_GROUP_SIZE+WaveGetLaneCount()-1)/WaveGetLaneCount();
 for(uint t=lane.x;t<43;t+=DENSE_GROUP_SIZE){float sum=0;for(uint w=0;w<waves;w++)sum+=denseTerms[t][w];outputTerms[group.x*43+t]=sum;}
#else
 for(uint stride=DENSE_GROUP_SIZE/2;stride>0;stride>>=1){if(lane.x<stride){for(uint t=0;t<43;t++)denseTerms[t][lane.x]+=denseTerms[t][lane.x+stride];denseValid[lane.x]+=denseValid[lane.x+stride];}GroupMemoryBarrierWithGroupSync();}
 if(lane.x==0){InterlockedAdd(trackingStatus[0],denseValid[0]);for(uint t=0;t<43;t++)outputTerms[group.x*43+t]=denseTerms[t][0];}
#endif
}

