// SPDX-License-Identifier: Apache-2.0
//
// Compact 3DGS math adapted from the approach used by NVIDIA's Apache-2.0
// nvpro-samples/vk_gaussian_splatting shaders. Rewritten for xsplat-vk's small
// differentiable compute path; no nvpro framework dependencies are retained.

#include "common.slang"

[Differentiable]
float3x3 rotationMatrix(float4 quaternion)
{
  float4 q = normalize(quaternion);
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return float3x3(
    1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w),
    2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w),
    2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y));
}

[Differentiable]
float3x3 covariance3D(float3 logScale, float4 rotation)
{
  float3 s = exp(logScale);
  float3x3 scale = float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z);
  float3x3 L = mul(rotationMatrix(rotation), scale);
  return mul(L, transpose(L));
}

[Differentiable]
float3 covariance2D(float3x3 cov3D, float3 centerView, float2 focal, float3x3 worldToViewRotation)
{
  float invZ = 1.0 / centerView.z;
  float invZ2 = invZ * invZ;
  float3x3 J = float3x3(
    focal.x * invZ, 0, -focal.x * centerView.x * invZ2,
    0, focal.y * invZ, -focal.y * centerView.y * invZ2,
    0, 0, 0);
  float3x3 T = mul(J, worldToViewRotation);
  float3x3 cov = mul(mul(T, cov3D), transpose(T));
  return float3(cov[0][0], cov[0][1], cov[1][1]);
}

[Differentiable]
float3 inverseCovariance2D(float3 covariance, float dilation)
{
  float a = covariance.x + dilation;
  float b = covariance.y;
  float d = covariance.z + dilation;
  float invDet = 1.0 / max(a*d - b*b, 1e-12);
  return float3(d * invDet, -b * invDet, a * invDet);
}
