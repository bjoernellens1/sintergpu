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

// SymForce/Caspar DABSEG kernel: pose_graph_factor
void caspar_pose_graph_factor(float ix, float iy, float iz, float iw, float itx, float ity, float itz, float jx, float jy, float jz, float jw, float jtx, float jty, float jtz, float mx, float my, float mz, float mw, float mtx, float mty, float mtz, out float out_0, out float out_1, out float out_2, out float out_3, out float out_4, out float out_5, out float out_6, out float out_7, out float out_8, out float out_9, out float out_10, out float out_11, out float out_12, out float out_13, out float out_14, out float out_15, out float out_16, out float out_17, out float out_18, out float out_19, out float out_20, out float out_21, out float out_22, out float out_23, out float out_24, out float out_25, out float out_26, out float out_27, out float out_28, out float out_29, out float out_30, out float out_31, out float out_32, out float out_33, out float out_34, out float out_35, out float out_36, out float out_37, out float out_38, out float out_39, out float out_40, out float out_41, out float out_42, out float out_43, out float out_44, out float out_45, out float out_46, out float out_47, out float out_48, out float out_49, out float out_50, out float out_51, out float out_52, out float out_53, out float out_54, out float out_55, out float out_56, out float out_57, out float out_58, out float out_59, out float out_60, out float out_61, out float out_62, out float out_63, out float out_64, out float out_65, out float out_66, out float out_67, out float out_68, out float out_69, out float out_70, out float out_71, out float out_72, out float out_73, out float out_74, out float out_75, out float out_76, out float out_77)
{
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f, r8 = 0.0f, r9 = 0.0f, r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f, r14 = 0.0f, r15 = 0.0f, r16 = 0.0f, r17 = 0.0f, r18 = 0.0f, r19 = 0.0f, r20 = 0.0f, r21 = 0.0f, r22 = 0.0f, r23 = 0.0f, r24 = 0.0f, r25 = 0.0f, r26 = 0.0f, r27 = 0.0f, r28 = 0.0f, r29 = 0.0f, r30 = 0.0f, r31 = 0.0f, r32 = 0.0f, r33 = 0.0f, r34 = 0.0f, r35 = 0.0f, r36 = 0.0f, r37 = 0.0f, r38 = 0.0f, r39 = 0.0f, r40 = 0.0f, r41 = 0.0f, r42 = 0.0f, r43 = 0.0f, r44 = 0.0f, r45 = 0.0f, r46 = 0.0f, r47 = 0.0f, r48 = 0.0f, r49 = 0.0f, r50 = 0.0f, r51 = 0.0f, r52 = 0.0f, r53 = 0.0f, r54 = 0.0f, r55 = 0.0f, r56 = 0.0f, r57 = 0.0f, r58 = 0.0f, r59 = 0.0f, r60 = 0.0f, r61 = 0.0f, r62 = 0.0f, r63 = 0.0f, r64 = 0.0f, r65 = 0.0f, r66 = 0.0f, r67 = 0.0f, r68 = 0.0f, r69 = 0.0f, r70 = 0.0f, r71 = 0.0f, r72 = 0.0f, r73 = 0.0f, r74 = 0.0f, r75 = 0.0f, r76 = 0.0f, r77 = 0.0f, r78 = 0.0f, r79 = 0.0f, r80 = 0.0f, r81 = 0.0f, r82 = 0.0f, r83 = 0.0f, r84 = 0.0f, r85 = 0.0f, r86 = 0.0f, r87 = 0.0f, r88 = 0.0f, r89 = 0.0f;
    r0 = mz;
    r1 = ix;
    r2 = jz;
    r3 = iy;
    r4 = jw;
    r5 = r3 * r4;
    r6 = -1.00000000000000000e+00;
    r5 = fma(r6, r5, r1 * r2);
    r7 = iw;
    r8 = jy;
    r9 = r7 * r8;
    r10 = iz;
    r11 = jx;
    r12 = r10 * r11;
    r5 = fma(r6, r12, r5);
    r5 = r5 + r9;
    r12 = mw;
    r13 = r1 * r4;
    r14 = r3 * r2;
    r14 = fma(r6, r14, r6 * r13);
    r14 = fma(r7, r11, r14);
    r14 = fma(r10, r8, r14);
    r13 = fma(r12, r14, r0 * r5);
    r15 = mx;
    r16 = r15 * r6;
    r17 = r1 * r11;
    r18 = r3 * r8;
    r18 = fma(r6, r18, r6 * r17);
    r17 = r10 * r2;
    r18 = fma(r6, r17, r18);
    r18 = fma(r6, r18, r7 * r4);
    r13 = fma(r18, r16, r13);
    r17 = r1 * r8;
    r17 = fma(r3, r11, r6 * r17);
    r19 = r10 * r4;
    r17 = fma(r7, r2, r17);
    r17 = fma(r6, r19, r17);
    r20 = my;
    r21 = r20 * r6;
    r13 = fma(r17, r21, r13);
    r16 = 2.00000000000000000e+00;
    r22 = 1.00000000000000000e+00;
    r23 = 0.00000000000000000e+00;
    r24 = r0 * r6;
    r25 = r15 * r6;
    r25 = fma(r14, r25, r17 * r24);
    r25 = fma(r5, r21, r25);
    r25 = fma(r12, r18, r6 * r25);
    r24 = r25 < r23 ? r6 : r22;
    r24 = r16 * r24;
    r26 = r13 * r24;
    out_0 = r26;
    r26 = fma(r12, r5, r15 * r17);
    r27 = r0 * r6;
    r26 = fma(r14, r27, r26);
    r26 = fma(r18, r21, r26);
    r27 = r26 * r24;
    out_1 = r27;
    r27 = r15 * r6;
    r27 = fma(r5, r27, r12 * r17);
    r17 = r0 * r6;
    r27 = fma(r18, r17, r27);
    r27 = fma(r20, r14, r27);
    r17 = r27 * r24;
    out_2 = r17;
    r17 = mtx;
    r14 = r1 * r3;
    r14 = r14 * r16;
    r18 = r10 * r7;
    r18 = r18 * r16;
    r5 = r14 + r18;
    r28 = jty;
    r29 = ity;
    r30 = fma(r29, r6, r28);
    r17 = fma(r5, r30, r17 * r6);
    r31 = jtz;
    r32 = itz;
    r33 = fma(r32, r6, r31);
    r34 = r1 * r10;
    r34 = r34 * r16;
    r35 = -2.00000000000000000e+00;
    r36 = r3 * r35;
    r37 = r7 * r36;
    r38 = r34 + r37;
    r39 = jtx;
    r40 = itx;
    r41 = fma(r40, r6, r39);
    r42 = r3 * r36;
    r43 = r10 * r10;
    r43 = fma(r35, r43, r22);
    r44 = r42 + r43;
    r17 = fma(r38, r33, r17);
    r17 = fma(r41, r44, r17);
    out_3 = r17;
    r17 = mty;
    r45 = r3 * r10;
    r45 = r45 * r16;
    r46 = r1 * r7;
    r46 = r46 * r16;
    r47 = r45 + r46;
    r17 = fma(r33, r47, r17 * r6);
    r48 = r10 * r7;
    r48 = r48 * r35;
    r14 = r14 + r48;
    r49 = r1 * r1;
    r49 = r49 * r35;
    r43 = r49 + r43;
    r17 = fma(r41, r14, r17);
    r17 = fma(r30, r43, r17);
    out_4 = r17;
    r17 = mtz;
    r50 = r1 * r7;
    r50 = r50 * r35;
    r45 = r45 + r50;
    r17 = fma(r30, r45, r17 * r6);
    r51 = r3 * r7;
    r51 = r51 * r16;
    r34 = r34 + r51;
    r42 = r22 + r42;
    r42 = r42 + r49;
    r17 = fma(r41, r34, r17);
    r17 = fma(r33, r42, r17);
    out_5 = r17;
    r13 = r16 * r13;
    r23 = r25 < r23 ? r23 : r23;
    r13 = r13 * r23;
    r25 = r15 * r6;
    r17 = r7 * r11;
    r49 = 5.00000000000000000e-01;
    r17 = r17 * r49;
    r22 = r10 * r8;
    r52 = -5.00000000000000000e-01;
    r22 = r22 * r52;
    r53 = r17 + r22;
    r54 = r1 * r52;
    r55 = r4 * r54;
    r56 = r2 * r49;
    r57 = r3 * r56;
    r58 = r55 + r57;
    r59 = r53 + r58;
    r60 = r3 * r8;
    r60 = r60 * r49;
    r61 = r7 * r4;
    r61 = r61 * r52;
    r62 = r60 + r61;
    r63 = r11 * r54;
    r64 = r10 * r56;
    r65 = r63 + r64;
    r66 = r62 + r65;
    r25 = fma(r12, r66, r59 * r25);
    r67 = r3 * r11;
    r67 = r67 * r52;
    r68 = r8 * r54;
    r69 = r67 + r68;
    r70 = r7 * r56;
    r71 = r49 * r19;
    r72 = r70 + r71;
    r73 = r69 + r72;
    r74 = r3 * r4;
    r74 = r74 * r52;
    r54 = r2 * r54;
    r75 = r74 + r54;
    r76 = r10 * r11;
    r76 = r76 * r52;
    r77 = r52 * r9;
    r78 = r76 + r77;
    r79 = r75 + r78;
    r25 = fma(r0, r73, r25);
    r25 = fma(r79, r21, r25);
    r25 = fma(r25, r24, r13);
    out_6 = r25;
    r25 = r7 * r2;
    r25 = r25 * r52;
    r19 = r52 * r19;
    r80 = r25 + r19;
    r69 = r69 + r80;
    r64 = r61 + r64;
    r61 = r1 * r11;
    r61 = r61 * r49;
    r81 = r3 * r8;
    r81 = r81 * r52;
    r82 = r61 + r81;
    r64 = r64 + r82;
    r83 = fma(r0, r64, r12 * r69);
    r84 = r15 * r6;
    r85 = r10 * r11;
    r85 = r85 * r49;
    r9 = r49 * r9;
    r86 = r85 + r9;
    r75 = r75 + r86;
    r83 = fma(r75, r84, r83);
    r87 = r1 * r4;
    r87 = r87 * r49;
    r88 = r3 * r2;
    r88 = r88 * r52;
    r89 = r87 + r88;
    r53 = r53 + r89;
    r83 = fma(r53, r21, r83);
    r83 = fma(r83, r24, r13);
    out_7 = r83;
    r88 = r22 + r88;
    r22 = r7 * r11;
    r22 = r22 * r52;
    r88 = r88 + r22;
    r88 = r88 + r55;
    r55 = r15 * r6;
    r83 = r1 * r8;
    r83 = r83 * r49;
    r67 = r67 + r83;
    r67 = r67 + r70;
    r67 = r67 + r19;
    r55 = fma(r67, r55, r0 * r88);
    r19 = r3 * r4;
    r19 = r19 * r49;
    r76 = r76 + r19;
    r76 = r76 + r54;
    r76 = r76 + r9;
    r9 = r10 * r2;
    r9 = r9 * r52;
    r61 = r61 + r9;
    r61 = r61 + r62;
    r55 = fma(r12, r76, r55);
    r55 = fma(r61, r21, r55);
    r55 = fma(r55, r24, r13);
    out_8 = r55;
    out_9 = r13;
    out_10 = r13;
    out_11 = r13;
    r26 = r16 * r26;
    r26 = r26 * r23;
    r55 = r0 * r6;
    r55 = fma(r12, r73, r66 * r55);
    r55 = fma(r15, r79, r55);
    r55 = fma(r59, r21, r55);
    r55 = fma(r55, r24, r26);
    out_12 = r55;
    r55 = r0 * r6;
    r55 = fma(r69, r55, r15 * r53);
    r55 = fma(r12, r64, r55);
    r55 = fma(r75, r21, r55);
    r55 = fma(r55, r24, r26);
    out_13 = r55;
    r55 = fma(r15, r61, r12 * r88);
    r62 = r0 * r6;
    r55 = fma(r76, r62, r55);
    r55 = fma(r67, r21, r55);
    r55 = fma(r55, r24, r26);
    out_14 = r55;
    out_15 = r26;
    out_16 = r26;
    out_17 = r26;
    r27 = r16 * r27;
    r27 = r27 * r23;
    r23 = r0 * r6;
    r66 = fma(r20, r66, r59 * r23);
    r23 = r15 * r6;
    r66 = fma(r73, r23, r66);
    r66 = fma(r12, r79, r66);
    r66 = fma(r66, r24, r27);
    out_18 = r66;
    r69 = fma(r20, r69, r12 * r53);
    r53 = r15 * r6;
    r69 = fma(r64, r53, r69);
    r64 = r0 * r6;
    r69 = fma(r75, r64, r69);
    r69 = fma(r69, r24, r27);
    out_19 = r69;
    r69 = r15 * r6;
    r64 = r0 * r6;
    r64 = fma(r67, r64, r88 * r69);
    r64 = fma(r12, r61, r64);
    r64 = fma(r20, r76, r64);
    r64 = fma(r64, r24, r27);
    out_20 = r64;
    out_21 = r27;
    out_22 = r27;
    out_23 = r27;
    r64 = r1 * r10;
    r64 = r64 * r35;
    r51 = r51 + r64;
    r51 = fma(r30, r51, r5 * r33);
    r35 = r29 * r6;
    r51 = fma(r38, r35, r51);
    r51 = fma(r32, r5, r51);
    out_24 = r51;
    r51 = r10 * r10;
    r35 = r7 * r7;
    r35 = r35 * r6;
    r76 = r51 + r35;
    r61 = r3 * r3;
    r69 = r1 * r1;
    r69 = r69 * r6;
    r67 = r61 + r69;
    r88 = r76 + r67;
    r88 = fma(r33, r88, r38 * r41);
    r53 = r32 * r6;
    r88 = fma(r44, r53, r88);
    r88 = fma(r40, r38, r88);
    out_25 = r88;
    r88 = r1 * r36;
    r48 = r48 + r88;
    r53 = r1 * r1;
    r75 = r10 * r10;
    r75 = r75 * r6;
    r66 = r53 + r75;
    r79 = r7 * r7;
    r23 = r3 * r3;
    r23 = r23 * r6;
    r73 = r79 + r23;
    r59 = r66 + r73;
    r59 = fma(r30, r59, r41 * r48);
    r48 = r40 * r6;
    r59 = fma(r5, r48, r59);
    r59 = fma(r29, r44, r59);
    out_26 = r59;
    r59 = r6 * r44;
    out_27 = r59;
    r59 = r6 * r5;
    out_28 = r59;
    r59 = r6 * r38;
    out_29 = r59;
    r36 = r10 * r36;
    r50 = r50 + r36;
    r59 = r29 * r6;
    r59 = fma(r47, r59, r30 * r50);
    r75 = r79 + r75;
    r75 = r75 + r67;
    r59 = fma(r33, r75, r59);
    r59 = fma(r32, r43, r59);
    out_30 = r59;
    r88 = r18 + r88;
    r88 = fma(r33, r88, r41 * r47);
    r18 = r32 * r6;
    r88 = fma(r14, r18, r88);
    r88 = fma(r40, r47, r88);
    out_31 = r88;
    r23 = r53 + r23;
    r23 = r23 + r76;
    r23 = fma(r41, r23, r30 * r14);
    r76 = r40 * r6;
    r23 = fma(r43, r76, r23);
    r23 = fma(r29, r14, r23);
    out_32 = r23;
    r23 = r6 * r14;
    out_33 = r23;
    r23 = r6 * r43;
    out_34 = r23;
    r23 = r6 * r47;
    out_35 = r23;
    r23 = fma(r32, r45, r33 * r45);
    r35 = r61 + r35;
    r35 = r35 + r66;
    r66 = r29 * r6;
    r23 = fma(r42, r66, r23);
    r23 = fma(r30, r35, r23);
    out_36 = r23;
    r37 = r64 + r37;
    r69 = r51 + r69;
    r69 = r69 + r73;
    r69 = fma(r41, r69, r33 * r37);
    r37 = r32 * r6;
    r69 = fma(r34, r37, r69);
    r69 = fma(r40, r42, r69);
    out_37 = r69;
    r36 = r46 + r36;
    r30 = fma(r30, r34, r41 * r36);
    r36 = r40 * r6;
    r30 = fma(r45, r36, r30);
    r30 = fma(r29, r34, r30);
    out_38 = r30;
    r30 = r6 * r34;
    out_39 = r30;
    r30 = r6 * r45;
    out_40 = r30;
    r30 = r6 * r42;
    out_41 = r30;
    r30 = r15 * r6;
    r36 = r10 * r8;
    r36 = r36 * r49;
    r22 = r22 + r36;
    r89 = r89 + r22;
    r41 = r3 * r11;
    r41 = r41 * r49;
    r83 = r83 + r41;
    r80 = r80 + r83;
    r30 = fma(r0, r80, r89 * r30);
    r7 = r7 * r4;
    r7 = r7 * r49;
    r9 = r9 + r7;
    r82 = r82 + r9;
    r56 = r1 * r56;
    r19 = r19 + r56;
    r86 = r86 + r19;
    r30 = fma(r12, r82, r30);
    r30 = fma(r86, r21, r30);
    r30 = fma(r30, r24, r13);
    out_42 = r30;
    r63 = r60 + r63;
    r63 = r63 + r9;
    r83 = r72 + r83;
    r72 = fma(r12, r83, r0 * r63);
    r9 = r15 * r6;
    r19 = r78 + r19;
    r72 = fma(r19, r9, r72);
    r22 = r58 + r22;
    r72 = fma(r22, r21, r72);
    r72 = fma(r72, r24, r13);
    out_43 = r72;
    r87 = r17 + r87;
    r87 = r87 + r36;
    r87 = r87 + r57;
    r57 = r15 * r6;
    r41 = r25 + r41;
    r41 = r41 + r68;
    r41 = r41 + r71;
    r57 = fma(r41, r57, r0 * r87);
    r85 = r74 + r85;
    r85 = r85 + r77;
    r85 = r85 + r56;
    r7 = r81 + r7;
    r7 = r7 + r65;
    r57 = fma(r12, r85, r57);
    r57 = fma(r7, r21, r57);
    r57 = fma(r57, r24, r13);
    out_44 = r57;
    out_45 = r13;
    out_46 = r13;
    out_47 = r13;
    r13 = r0 * r6;
    r13 = fma(r82, r13, r12 * r80);
    r13 = fma(r15, r86, r13);
    r13 = fma(r89, r21, r13);
    r13 = fma(r13, r24, r26);
    out_48 = r13;
    r13 = fma(r12, r63, r15 * r22);
    r57 = r0 * r6;
    r13 = fma(r83, r57, r13);
    r13 = fma(r19, r21, r13);
    r13 = fma(r13, r24, r26);
    out_49 = r13;
    r13 = fma(r15, r7, r12 * r87);
    r57 = r0 * r6;
    r13 = fma(r85, r57, r13);
    r13 = fma(r41, r21, r13);
    r13 = fma(r13, r24, r26);
    out_50 = r13;
    out_51 = r26;
    out_52 = r26;
    out_53 = r26;
    r26 = r0 * r6;
    r13 = r15 * r6;
    r13 = fma(r80, r13, r89 * r26);
    r13 = fma(r20, r82, r13);
    r13 = fma(r12, r86, r13);
    r13 = fma(r13, r24, r27);
    out_54 = r13;
    r13 = r15 * r6;
    r13 = fma(r63, r13, r12 * r22);
    r22 = r0 * r6;
    r13 = fma(r19, r22, r13);
    r13 = fma(r20, r83, r13);
    r13 = fma(r13, r24, r27);
    out_55 = r13;
    r13 = r15 * r6;
    r22 = r0 * r6;
    r22 = fma(r41, r22, r87 * r13);
    r22 = fma(r12, r7, r22);
    r22 = fma(r20, r85, r22);
    r24 = fma(r22, r24, r27);
    out_56 = r24;
    out_57 = r27;
    out_58 = r27;
    out_59 = r27;
    r27 = r31 * r6;
    r27 = fma(r28, r38, r5 * r27);
    out_60 = r27;
    r27 = r39 * r6;
    r27 = fma(r38, r27, r31 * r44);
    out_61 = r27;
    r27 = r28 * r6;
    r27 = fma(r39, r5, r44 * r27);
    out_62 = r27;
    out_63 = r44;
    out_64 = r5;
    out_65 = r38;
    r38 = r31 * r6;
    r38 = fma(r43, r38, r28 * r47);
    out_66 = r38;
    r38 = r39 * r6;
    r38 = fma(r31, r14, r47 * r38);
    out_67 = r38;
    r38 = r28 * r6;
    r38 = fma(r14, r38, r39 * r43);
    out_68 = r38;
    out_69 = r14;
    out_70 = r43;
    out_71 = r47;
    r47 = r31 * r6;
    r47 = fma(r28, r42, r45 * r47);
    out_72 = r47;
    r47 = r39 * r6;
    r47 = fma(r31, r34, r42 * r47);
    out_73 = r47;
    r47 = r28 * r6;
    r47 = fma(r34, r47, r39 * r45);
    out_74 = r47;
    out_75 = r34;
    out_76 = r45;
    out_77 = r42;
}


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

