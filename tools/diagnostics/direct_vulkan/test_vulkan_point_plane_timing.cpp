// SPDX-License-Identifier: Apache-2.0
// CPU-only fake Vulkan dispatch; never links a Vulkan loader.
#include "vulkan_point_plane_timing.h"
#include <cassert>
#include <vector>
static std::vector<int> ev;
static int failure=0;
static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice,const VkQueryPoolCreateInfo*,const VkAllocationCallbacks*,VkQueryPool* out){*out=(VkQueryPool)0x20;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice,VkQueryPool,const VkAllocationCallbacks*){}
static VKAPI_ATTR void VKAPI_CALL qreset(VkCommandBuffer,VkQueryPool,uint32_t,uint32_t){ev.push_back(3);}
static VKAPI_ATTR void VKAPI_CALL timestamp(VkCommandBuffer,VkPipelineStageFlagBits,VkQueryPool,uint32_t i){ev.push_back(10+i);}
static VKAPI_ATTR VkResult VKAPI_CALL read(VkDevice,VkQueryPool,uint32_t,uint32_t,size_t,void* out,VkDeviceSize,VkQueryResultFlags){auto* p=(uint64_t*)out;p[0]=10;p[1]=11;p[2]=14;p[3]=15;ev.push_back(30);return failure==5?VK_NOT_READY:VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,VkDependencyFlags,uint32_t,const VkMemoryBarrier*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*){ev.push_back(4);}
static VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer,VkPipelineBindPoint,VkPipeline){ev.push_back(5);}
static VKAPI_ATTR void VKAPI_CALL sets(VkCommandBuffer,VkPipelineBindPoint,VkPipelineLayout,uint32_t,uint32_t,const VkDescriptorSet*,uint32_t,const uint32_t*){ev.push_back(6);}
static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer,uint32_t,uint32_t,uint32_t){ev.push_back(7);}
static VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer,VkBuffer,VkBuffer,uint32_t,const VkBufferCopy*){ev.push_back(8);}
static VKAPI_ATTR VkResult VKAPI_CALL reset(VkCommandBuffer,VkCommandBufferResetFlags){ev.push_back(1);return failure==1?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
static VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer,const VkCommandBufferBeginInfo*){ev.push_back(2);return failure==2?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
static VKAPI_ATTR VkResult VKAPI_CALL end(VkCommandBuffer){ev.push_back(20);return failure==3?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
static VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue,uint32_t n,const VkSubmitInfo* info,VkFence){assert(n==1&&info->commandBufferCount==1);ev.push_back(21);return failure==4?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
static VKAPI_ATTR VkResult VKAPI_CALL wait(VkQueue){ev.push_back(22);return failure==6?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
int main(){
 SinterTimestampPair q((VkDevice)1,32,1,{create,destroy,qreset,timestamp,read});
 SinterTimingDispatch api{reset,begin,end,submit,wait,copy,{bind,sets,dispatch,barrier}};
 SinterPointPlaneCommand p{(VkPipeline)1,(VkPipelineLayout)2,(VkDescriptorSet)3,5};
 SinterPointPlaneTiming t((VkQueue)1,(VkCommandBuffer)1,api,q);
 std::array<SinterTimingStaging,7> slots{};std::array<std::array<unsigned char,4>,7> upload{},download{},initial{},out{};
 for(size_t i=0;i<7;i++){initial[i].fill((unsigned char)(i+1));download[i].fill((unsigned char)(20+i));slots[i]={{(VkBuffer)(uintptr_t)(3*i+1),(VkBuffer)(uintptr_t)(3*i+2),(VkBuffer)(uintptr_t)(3*i+3),4},upload[i].data(),download[i].data(),initial[i].data(),out[i].data()};}
 try{t.transfer(p,slots);assert(false);}catch(const std::logic_error&){}
 auto r=t.resident(p);assert((r.rawTicks==std::array<uint64_t,4>({10,11,14,15})));assert(r.hostCompletionUpperBoundNs>0);assert(ev[ev.size()-3]==21&&ev[ev.size()-2]==22&&ev.back()==30);
 try{t.resident(p);assert(false);}catch(const std::logic_error&){}t.markResidentValidated();ev.clear();assert(t.transfer(p,slots)>0);assert(upload==initial&&out==download);assert(ev[ev.size()-2]==21&&ev.back()==22);t.markTransferValidated();
 // The diagnostic subdivisions share exactly the original host interval.
 t.resident(p);t.markResidentValidated();SinterTransferComponents parts{};
 auto total=t.transfer(p,slots,&parts);
 assert(total==parts.uploadNs+parts.recordNs+parts.submitWaitNs+parts.downloadNs);
 assert(upload==initial&&out==download);t.markTransferValidated();
 for(int injected=1;injected<=6;++injected){
   SinterTimestampPair fresh((VkDevice)1,32,1,{create,destroy,qreset,timestamp,read});
   SinterPointPlaneTiming f((VkQueue)1,(VkCommandBuffer)1,api,fresh);ev.clear();failure=injected;
   try{f.resident(p);assert(false);}catch(const std::runtime_error&){}
   failure=0;auto count=ev.size();try{f.resident(p);assert(false);}catch(const std::logic_error&){}
   assert(ev.size()==count);
 }
 // A failure in the separately recorded transfer also poisons the instance.
 SinterTimestampPair fresh((VkDevice)1,32,1,{create,destroy,qreset,timestamp,read});
 SinterPointPlaneTiming f((VkQueue)1,(VkCommandBuffer)1,api,fresh);f.resident(p);f.markResidentValidated();failure=6;
 try{f.transfer(p,slots);assert(false);}catch(const std::runtime_error&){}failure=0;
 try{f.transfer(p,slots);assert(false);}catch(const std::logic_error&){}
}
