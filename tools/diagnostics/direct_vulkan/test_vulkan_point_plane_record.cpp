// SPDX-License-Identifier: Apache-2.0
// Synthetic dispatch functions; no real Vulkan loader/device is used.
#include "vulkan_point_plane_record.h"
#include <cassert>
#include <vector>
static std::vector<int> events;
static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice,const VkQueryPoolCreateInfo*,const VkAllocationCallbacks*,VkQueryPool* out){*out=(VkQueryPool)0x20;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice,VkQueryPool,const VkAllocationCallbacks*){}
static VKAPI_ATTR void VKAPI_CALL reset(VkCommandBuffer,VkQueryPool,uint32_t,uint32_t){events.push_back(1);}
static VKAPI_ATTR void VKAPI_CALL timestamp(VkCommandBuffer,VkPipelineStageFlagBits,VkQueryPool,uint32_t i){events.push_back(10+i);}
static VKAPI_ATTR VkResult VKAPI_CALL read(VkDevice,VkQueryPool,uint32_t,uint32_t,size_t,void*,VkDeviceSize,VkQueryResultFlags){return VK_NOT_READY;}
static VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,VkDependencyFlags,uint32_t,const VkMemoryBarrier*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*){events.push_back(2);}
static VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer,VkPipelineBindPoint,VkPipeline){events.push_back(3);}
static VKAPI_ATTR void VKAPI_CALL sets(VkCommandBuffer,VkPipelineBindPoint,VkPipelineLayout,uint32_t,uint32_t,const VkDescriptorSet*,uint32_t,const uint32_t*){events.push_back(4);}
static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer,uint32_t x,uint32_t y,uint32_t z){assert(x==5&&y==1&&z==1);events.push_back(5);}
static VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer,VkBuffer,VkBuffer,uint32_t,const VkBufferCopy*){events.push_back(6);}
int main(){VkCommandBuffer cmd=(VkCommandBuffer)1;SinterPointPlaneCommand p{(VkPipeline)1,(VkPipelineLayout)2,(VkDescriptorSet)3,5};SinterPointPlaneDispatch api{bind,sets,dispatch,barrier};SinterTimestampPair t((VkDevice)1,32,1,{create,destroy,reset,timestamp,read});
    sinterRecordResidentPointPlane(cmd,p,api,t);
    assert(events==std::vector<int>({1,10,2,11,3,4,5,12,2,13}));
    std::array<SinterTransferBuffer,7> b{};for(size_t i=0;i<7;i++)b[i]={(VkBuffer)(uintptr_t)(i*3+1),(VkBuffer)(uintptr_t)(i*3+2),(VkBuffer)(uintptr_t)(i*3+3),32};
    events.clear();sinterRecordTransferPointPlane(cmd,p,b,api,copy);assert(events.size()==21&&events[8]==2&&events[9]==3&&events[10]==4&&events[11]==5&&events[12]==2);
    p.pipeline=VK_NULL_HANDLE;events.clear();try{sinterRecordTransferPointPlane(cmd,p,b,api,copy);assert(false);}catch(const std::invalid_argument&){}assert(events.empty());
}
