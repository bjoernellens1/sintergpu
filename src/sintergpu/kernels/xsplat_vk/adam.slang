// SPDX-License-Identifier: Apache-2.0
// Standalone Adam keeps optimization policy separate from VkSplat projection VJP.
#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

struct Params
{
  uint count;
  uint step;
  float learningRate;
  float beta1;
  float beta2;
  float epsilon;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> values;
[[vk::binding(1, 0)]] StructuredBuffer<float> gradients;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> firstMoment;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> secondMoment;

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Params p)
{
  if (tid.x >= p.count)
    return;

  uint i = tid.x;
  float g = gradients[i];
  float m = firstMoment[i] = p.beta1 * firstMoment[i] + (1.0 - p.beta1) * g;
  float v = secondMoment[i] = p.beta2 * secondMoment[i] + (1.0 - p.beta2) * g * g;
  float t = float(max(p.step, 1u));
  float mh = m / (1.0 - pow(p.beta1, t));
  float vh = v / (1.0 - pow(p.beta2, t));
  values[i] -= p.learningRate * mh / (sqrt(vh) + p.epsilon);
}
