// SPDX-License-Identifier: Apache-2.0
// Explicit fixed-fixture Vulkan resources. Caller owns the device and must wait
// for all uses before destruction. This observes Vulkan handles, not physical backing.
#pragma once
#include "timing_resource_plan.h"
#include "vulkan_transfer_pass.h"
#include <algorithm>
#include <cstdint>

struct SinterResourceAPI {
 PFN_vkCreateBuffer createBuffer; PFN_vkDestroyBuffer destroyBuffer;
 PFN_vkGetBufferMemoryRequirements requirements; PFN_vkAllocateMemory allocate;
 PFN_vkFreeMemory free; PFN_vkBindBufferMemory bind; PFN_vkMapMemory map; PFN_vkUnmapMemory unmap;
};
struct SinterTimingResource { VkBuffer buffer=VK_NULL_HANDLE; VkDeviceMemory memory=VK_NULL_HANDLE; void* mapped=nullptr; VkDeviceSize size=0; };
class SinterTimingResources {
public:
 SinterTimingResources(VkDevice device,const VkPhysicalDeviceMemoryProperties& properties,
                       const SinterTimingAllocationPlan& plan,SinterResourceAPI api):device_(device),api_(api){
  if(!device||!api.createBuffer||!api.destroyBuffer||!api.requirements||!api.allocate||!api.free||!api.bind||!api.map||!api.unmap||properties.memoryTypeCount>VK_MAX_MEMORY_TYPES)throw std::invalid_argument("resource API");
  try {for(size_t i=0;i<7;++i)for(size_t j=0;j<3;++j){
   const auto& spec=plan[i][j];auto& r=resources_[i][j];if(!spec.size||!spec.usage)throw std::invalid_argument("empty resource");
   r.size=spec.size;VkBufferCreateInfo b{};b.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;b.size=r.size;b.usage=spec.usage;b.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
   check(api_.createBuffer(device_,&b,nullptr,&r.buffer));if(!r.buffer)throw std::runtime_error("null buffer");
   for(size_t x=0;x<=i;++x)for(size_t y=0;y<3;++y){if(x==i&&y>=j)break;if(resources_[x][y].buffer==r.buffer){r.buffer=VK_NULL_HANDLE;throw std::runtime_error("duplicate buffer");}}
   VkMemoryRequirements req{};api_.requirements(device_,r.buffer,&req);if(req.size<r.size||!req.alignment)throw std::runtime_error("buffer requirements");
   uint32_t type=0;while(type<properties.memoryTypeCount&&(!(req.memoryTypeBits&(uint32_t(1)<<type))||(properties.memoryTypes[type].propertyFlags&spec.requiredMemory)!=spec.requiredMemory))++type;
   if(type==properties.memoryTypeCount)throw std::runtime_error("memory type unavailable");
   VkMemoryAllocateInfo a{};a.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;a.allocationSize=req.size;a.memoryTypeIndex=type;check(api_.allocate(device_,&a,nullptr,&r.memory));if(!r.memory)throw std::runtime_error("null memory");
   for(size_t x=0;x<=i;++x)for(size_t y=0;y<3;++y){if(x==i&&y>=j)break;if(resources_[x][y].memory==r.memory){r.memory=VK_NULL_HANDLE;throw std::runtime_error("duplicate memory");}}
   check(api_.bind(device_,r.buffer,r.memory,0));
   if(spec.requiredMemory&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT){check(api_.map(device_,r.memory,0,r.size,0,&r.mapped));if(!r.mapped)throw std::runtime_error("null mapping");}
  }}catch(...){clear();throw;}
 }
 SinterTimingResources(const SinterTimingResources&)=delete;SinterTimingResources& operator=(const SinterTimingResources&)=delete;
 ~SinterTimingResources(){clear();}
 const SinterTimingResource& at(size_t i,size_t j)const{return resources_.at(i).at(j);}
 std::array<SinterTransferBuffer,7> transfers()const{std::array<SinterTransferBuffer,7> out{};for(size_t i=0;i<7;++i)out[i]={resources_[i][0].buffer,resources_[i][1].buffer,resources_[i][2].buffer,resources_[i][1].size};return out;}
private:
 static void check(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("Vulkan allocation failed");}
 void clear()noexcept{for(auto& row:resources_)for(auto& r:row){if(r.mapped)api_.unmap(device_,r.memory);if(r.buffer)api_.destroyBuffer(device_,r.buffer,nullptr);if(r.memory)api_.free(device_,r.memory,nullptr);r={};}}
 VkDevice device_;SinterResourceAPI api_;std::array<std::array<SinterTimingResource,3>,7> resources_{};
};
