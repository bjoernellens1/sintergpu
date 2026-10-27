// SPDX-License-Identifier: Apache-2.0
// Original xsplat-vk implementation. Mathematical conventions follow standard 3DGS.

struct CameraData
{
  float4x4 worldToCamera;
  float2 focal;
  float2 principal;
  uint2 imageSize;
  uint gaussianCount;
};

struct GaussianData
{
  float3 mean;
  float3 logScale;
  float4 rotation;
  float opacityLogit;
  float3 sh0;
};

struct ProjectedGaussian
{
  float2 meanPx;
  float3 conic; // xx, xy, yy
  float depth;
  uint visible;
};

float sigmoid(float x)
{
  return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
float gaussianAlpha(float2 delta, float3 conic, float opacity)
{
  float q = conic.x * delta.x * delta.x + 2.0 * conic.y * delta.x * delta.y + conic.z * delta.y * delta.y;
  return min(0.999, opacity * exp(-0.5 * q));
}
