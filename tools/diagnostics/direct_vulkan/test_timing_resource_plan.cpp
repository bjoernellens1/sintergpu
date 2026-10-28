// SPDX-License-Identifier: Apache-2.0
#include "timing_resource_plan.h"
#include <cassert>
int main(){
 VkPhysicalDeviceLimits l{};l.maxMemoryAllocationCount=21;l.maxStorageBufferRange=24796;l.maxUniformBufferRange=32;
 l.maxPerStageDescriptorStorageBuffers=6;l.maxDescriptorSetStorageBuffers=6;l.maxPerStageDescriptorUniformBuffers=1;
 l.maxDescriptorSetUniformBuffers=1;l.maxPerStageResources=7;l.maxBoundDescriptorSets=1;
 auto p=sinterFixedPointPlaneAllocationPlan(l);
 for(size_t i=0;i<7;++i){assert(p[i][0].size==p[i][1].size&&p[i][1].size==p[i][2].size);assert(p[i][0].usage==VK_BUFFER_USAGE_TRANSFER_SRC_BIT);assert(p[i][2].usage==VK_BUFFER_USAGE_TRANSFER_DST_BIT);assert(p[i][1].requiredMemory==VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);}
 assert(p[6][1].size==32&&(p[6][1].usage&VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
 assert((p[0][1].usage&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)!=0);
 l.maxMemoryAllocationCount=20;try{sinterFixedPointPlaneAllocationPlan(l);assert(false);}catch(const std::invalid_argument&){}
 l.maxMemoryAllocationCount=21;l.maxUniformBufferRange=31;try{sinterFixedPointPlaneAllocationPlan(l);assert(false);}catch(const std::invalid_argument&){}
}
