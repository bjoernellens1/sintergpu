// SPDX-License-Identifier: Apache-2.0
// Pure CPU fake Vulkan callbacks; no loader or device.
#include "vulkan_timestamp_pair.h"
#include <cassert>
#include <vector>
struct Event { int kind; uint32_t query; VkPipelineStageFlagBits stage; };
static std::vector<Event> events;static bool ready=false;static unsigned destroys=0;
static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice,const VkQueryPoolCreateInfo* p,const VkAllocationCallbacks*,VkQueryPool* out){assert(p->queryCount==4 && p->queryType==VK_QUERY_TYPE_TIMESTAMP);*out=(VkQueryPool)0x11;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice,VkQueryPool,const VkAllocationCallbacks*){++destroys;}
static VKAPI_ATTR void VKAPI_CALL reset(VkCommandBuffer,VkQueryPool,uint32_t first,uint32_t count){assert(first==0&&count==4);events.push_back({0,0,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT});}
static VKAPI_ATTR void VKAPI_CALL write(VkCommandBuffer,VkPipelineStageFlagBits stage,VkQueryPool,uint32_t query){events.push_back({1,query,stage});}
static VKAPI_ATTR VkResult VKAPI_CALL read(VkDevice,VkQueryPool,uint32_t first,uint32_t count,size_t size,void* data,VkDeviceSize stride,VkQueryResultFlags flags){assert(first==0&&count==4&&size==32&&stride==8&&flags==VK_QUERY_RESULT_64_BIT);if(!ready)return VK_NOT_READY;auto p=(uint64_t*)data;p[0]=250;p[1]=251;p[2]=2;p[3]=3;return VK_SUCCESS;}
int main(){SinterTimestampDispatch api{create,destroy,reset,write,read};VkDevice device=(VkDevice)0x1;VkCommandBuffer cmd=(VkCommandBuffer)0x2;
    try{SinterTimestampPair invalid(device,0,1,api);assert(false);}catch(const std::invalid_argument&){}
    { SinterTimestampPair t(device,8,2,api);std::array<uint64_t,4> raw{};
      try{t.beginKernel(cmd);assert(false);}catch(const std::logic_error&){}
      t.beginPipeline(cmd);t.beginKernel(cmd);t.endKernel(cmd);t.endPipeline(cmd);
      assert(t.read(raw)==VK_NOT_READY);ready=true;assert(t.read(raw)==VK_SUCCESS);
      assert(raw[0]==250&&raw[1]==251&&raw[2]==2&&raw[3]==3);
      assert(events.size()==5&&events[1].query==0&&events[1].stage==VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
      assert(events[2].query==1&&events[2].stage==VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      assert(events[3].query==2&&events[3].stage==VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      assert(events[4].query==3&&events[4].stage==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      t.beginPipeline(cmd);try{t.beginKernel((VkCommandBuffer)0x3);assert(false);}catch(const std::logic_error&){}
    } assert(destroys==1);
}
