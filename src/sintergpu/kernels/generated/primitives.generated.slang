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
