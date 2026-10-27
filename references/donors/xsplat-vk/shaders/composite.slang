// SPDX-License-Identifier: Apache-2.0
#include "common.slang"

struct CompositeState
{
  float3 rgb;
  float transmittance;
};

[Differentiable]
CompositeState compositeOne(CompositeState state, float3 rgb, float alpha)
{
  float weight = state.transmittance * alpha;
  state.rgb += weight * rgb;
  state.transmittance *= 1.0 - alpha;
  return state;
}

// Training uses this explicit function instead of Vulkan fixed-function blending.
// The full raster backward kernel will call bwd_diff(compositeOne) or a custom
// derivative after profiling. Sorting/visibility are deliberately not differentiated.
