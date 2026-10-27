// SPDX-License-Identifier: Apache-2.0
#include "gaussian_math.slang"

[[vk::binding(0, 0)]] StructuredBuffer<GaussianData> gaussians;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(2, 0)]] ConstantBuffer<CameraData> camera;

[shader("compute")]
[numthreads(128, 1, 1)]
void projectMain(uint3 tid : SV_DispatchThreadID)
{
  if (tid.x >= camera.gaussianCount)
    return;

  GaussianData g = gaussians[tid.x];
  float4 center = mul(camera.worldToCamera, float4(g.mean, 1.0));
  ProjectedGaussian out = {};
  out.depth = center.z;
  if (center.z <= 1e-4)
  {
    projected[tid.x] = out;
    return;
  }

  float invZ = 1.0 / center.z;
  out.meanPx = camera.focal * center.xy * invZ + camera.principal;
  out.visible = uint(all(out.meanPx >= 0.0) && all(out.meanPx < float2(camera.imageSize)));

  float3x3 viewRotation = float3x3(camera.worldToCamera);
  float3 cov = covariance2D(covariance3D(g.logScale, g.rotation), center.xyz,
                            camera.focal, viewRotation);
  out.conic = inverseCovariance2D(cov, 0.3);
  projected[tid.x] = out;
}
