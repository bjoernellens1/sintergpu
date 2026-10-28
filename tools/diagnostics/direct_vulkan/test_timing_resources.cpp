// SPDX-License-Identifier: Apache-2.0
#include "timing_resources.h"
#include <cassert>
static uintptr_t nextBuffer=1,nextMemory=100;static int creates=0,destroys=0,allocs=0,frees=0,maps=0,unmaps=0,failAt=0;
static unsigned char memory[65536];
static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice,const VkBufferCreateInfo* p,const VkAllocationCallbacks*,VkBuffer* b){assert(p->size&&p->usage);++creates;*b=(VkBuffer)nextBuffer++;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice,VkBuffer,const VkAllocationCallbacks*){++destroys;}
static VKAPI_ATTR void VKAPI_CALL requirements(VkDevice,VkBuffer,VkMemoryRequirements* r){r->size=65536;r->alignment=4;r->memoryTypeBits=3;}
static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice,const VkMemoryAllocateInfo*,const VkAllocationCallbacks*,VkDeviceMemory* m){if(failAt&&allocs+1==failAt)return VK_ERROR_OUT_OF_DEVICE_MEMORY;++allocs;*m=(VkDeviceMemory)nextMemory++;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL freeMemory(VkDevice,VkDeviceMemory,const VkAllocationCallbacks*){++frees;}
static VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice,VkBuffer,VkDeviceMemory,VkDeviceSize offset){assert(offset==0);return VK_SUCCESS;}
static VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice,VkDeviceMemory,VkDeviceSize,VkDeviceSize,VkMemoryMapFlags,void** p){++maps;*p=memory;return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL unmap(VkDevice,VkDeviceMemory){++unmaps;}
int main(){
 VkPhysicalDeviceLimits l{};l.maxMemoryAllocationCount=21;l.maxStorageBufferRange=24796;l.maxUniformBufferRange=32;l.maxPerStageDescriptorStorageBuffers=6;l.maxDescriptorSetStorageBuffers=6;l.maxPerStageDescriptorUniformBuffers=1;l.maxDescriptorSetUniformBuffers=1;l.maxPerStageResources=7;l.maxBoundDescriptorSets=1;
 VkPhysicalDeviceMemoryProperties p{};p.memoryTypeCount=2;p.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;p.memoryTypes[1].propertyFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
 SinterResourceAPI api{create,destroy,requirements,allocate,freeMemory,bind,map,unmap};
 {SinterTimingResources r((VkDevice)1,p,sinterFixedPointPlaneAllocationPlan(l),api);auto transfers=r.transfers();assert(transfers[6].size==32);assert(r.at(0,0).mapped&&r.at(0,2).mapped&&!r.at(0,1).mapped);assert(creates==21&&allocs==21&&maps==14);}
 assert(destroys==21&&frees==21&&unmaps==14);
 creates=destroys=allocs=frees=maps=unmaps=0;failAt=5;
 try{SinterTimingResources r((VkDevice)1,p,sinterFixedPointPlaneAllocationPlan(l),api);assert(false);}catch(const std::runtime_error&){}
 assert(creates==5&&destroys==5&&allocs==4&&frees==4&&maps==3&&unmaps==3);
 return 0;
}
