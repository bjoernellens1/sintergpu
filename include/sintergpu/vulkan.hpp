#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>

namespace sintergpu {
// Pipeline/descriptors must match reflection. Caller owns barriers and submission.
inline void record_compute(VkCommandBuffer command, VkPipeline pipeline,
                           VkPipelineLayout layout, VkDescriptorSet descriptors,
                           uint32_t x, uint32_t y = 1, uint32_t z = 1) {
    if (!command || !pipeline || !layout || !descriptors)
        throw std::invalid_argument("borrowed Vulkan handles must be valid");
    if (!x || !y || !z) return;
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &descriptors, 0, nullptr);
    vkCmdDispatch(command, x, y, z);
}
} // namespace sintergpu
