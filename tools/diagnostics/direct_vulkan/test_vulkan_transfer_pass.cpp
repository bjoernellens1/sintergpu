// SPDX-License-Identifier: Apache-2.0
// CPU injected Vulkan calls only; no GPU or loader.
#include "vulkan_transfer_pass.h"
#include <cassert>
#include <vector>
static std::vector<int> events;
static VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer,VkBuffer s,VkBuffer d,uint32_t n,const VkBufferCopy* region){assert(s&&d&&s!=d&&n==1&&region->size);events.push_back(1);}
static VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags,uint32_t n,const VkMemoryBarrier* b,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*){assert(n==1&&b);if(s==VK_PIPELINE_STAGE_HOST_BIT&&d==VK_PIPELINE_STAGE_TRANSFER_BIT){assert(b->srcAccessMask==VK_ACCESS_HOST_WRITE_BIT&&b->dstAccessMask==VK_ACCESS_TRANSFER_READ_BIT);events.push_back(2);}else if(s==VK_PIPELINE_STAGE_TRANSFER_BIT&&d==(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT)){assert(b->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&b->dstAccessMask==(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT));events.push_back(3);}else if(s==VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT&&d==VK_PIPELINE_STAGE_TRANSFER_BIT){assert(b->srcAccessMask==(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT)&&b->dstAccessMask==VK_ACCESS_TRANSFER_READ_BIT);events.push_back(4);}else if(s==VK_PIPELINE_STAGE_TRANSFER_BIT&&d==VK_PIPELINE_STAGE_HOST_BIT){assert(b->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&b->dstAccessMask==VK_ACCESS_HOST_READ_BIT);events.push_back(5);}else assert(false);}
int main(){std::array<SinterTransferBuffer,7> b{};for(size_t i=0;i<7;i++)b[i]={(VkBuffer)(uintptr_t)(i*3+1),(VkBuffer)(uintptr_t)(i*3+2),(VkBuffer)(uintptr_t)(i*3+3),32};
    sinterRecordTransferPass((VkCommandBuffer)0x1,b,{copy,barrier},[](VkCommandBuffer){events.push_back(6);});
    assert(events.size()==19&&events[0]==2&&events[8]==3&&events[9]==6&&events[10]==4&&events[18]==5);
    auto bad=b;bad[6].download=b[0].device;size_t n=events.size();try{sinterRecordTransferPass((VkCommandBuffer)0x1,bad,{copy,barrier},[](VkCommandBuffer){});assert(false);}catch(const std::invalid_argument&){}assert(events.size()==n);
    bad=b;bad[1].size=0;try{sinterRecordTransferPass((VkCommandBuffer)0x1,bad,{copy,barrier},[](VkCommandBuffer){});assert(false);}catch(const std::invalid_argument&){}assert(events.size()==n);
}
