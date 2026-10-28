// SPDX-License-Identifier: Apache-2.0
// Fixed fixture allocation requirements; no Vulkan calls or implicit allocation.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>

struct SinterTimingAllocationSpec {
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags requiredMemory;
};
using SinterTimingAllocationPlan = std::array<std::array<SinterTimingAllocationSpec,3>,7>;

inline SinterTimingAllocationPlan sinterFixedPointPlaneAllocationPlan(const VkPhysicalDeviceLimits& l) {
    constexpr std::array<VkDeviceSize,7> sizes={8192,16512,16512,4156,24796,4100,32};
    if(l.maxMemoryAllocationCount<21 || l.maxStorageBufferRange<24796 || l.maxUniformBufferRange<32 ||
       l.maxPerStageDescriptorStorageBuffers<6 || l.maxDescriptorSetStorageBuffers<6 ||
       l.maxPerStageDescriptorUniformBuffers<1 || l.maxDescriptorSetUniformBuffers<1 ||
       l.maxPerStageResources<7 || l.maxBoundDescriptorSets<1) throw std::invalid_argument("fixture device limits");
    constexpr auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    SinterTimingAllocationPlan plan{};
    for(size_t i=0;i<sizes.size();++i){
        plan[i][0]={sizes[i],VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host};
        plan[i][1]={sizes[i],VkBufferUsageFlags(VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|
            (i==6?VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)),VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
        plan[i][2]={sizes[i],VK_BUFFER_USAGE_TRANSFER_DST_BIT,host};
    }
    return plan;
}
