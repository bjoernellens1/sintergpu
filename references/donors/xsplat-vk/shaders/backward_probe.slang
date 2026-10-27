// SPDX-License-Identifier: Apache-2.0
#include "composite.slang"

// Tiny compile/runtime probe for Slang reverse mode. This is intentionally not
// the rasterizer: it proves the shader toolchain can differentiate the exact
// compositing primitive used by the future tile kernel.
struct BackwardProbe
{
  float3 dRgb;
  float dAlpha;
  float dInputTransmittance;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<BackwardProbe> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void backwardProbeMain(uint3 tid : SV_DispatchThreadID)
{
  CompositeState state = {float3(0.1, 0.2, 0.3), 0.72};
  float3 rgb = float3(0.8, 0.4, 0.2);
  float alpha = 0.31;
  float3 dLossRgb = float3(0.7, -0.2, 0.5);
  float dLossTransmittance = 0.13;

  var dState = diffPair(state);
  var dRgb = diffPair(rgb);
  var dAlpha = diffPair(alpha);
  CompositeState dOut = {dLossRgb, dLossTransmittance};
  bwd_diff(compositeOne)(dState, dRgb, dAlpha, dOut);

  output[0].dRgb = dRgb.d;
  output[0].dAlpha = dAlpha.d;
  output[0].dInputTransmittance = dState.d.transmittance;
}
