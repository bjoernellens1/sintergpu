// SPDX-License-Identifier: Apache-2.0
// CPU-only fake: abort any unexpected Vulkan call by never creating a device.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
void sintergpu_test_only_fake_vulkan_marker(void) {}
static void event(const char *name) {const char *path=getenv("SINTERGPU_FAKE_EVENTS");if(path){FILE *f=fopen(path,"a");if(f){fprintf(f,"%s\n",name);fclose(f);}}}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* i,const VkAllocationCallbacks* a,VkInstance* out) {(void)i;(void)a;event("create_instance");*out=(VkInstance)0x101;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance i,uint32_t* n,VkPhysicalDevice* p) {(void)i;event("enumerate");if(!p){*n=1;return VK_SUCCESS;}*n=1;p[0]=(VkPhysicalDevice)0x102;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice p,VkPhysicalDeviceProperties2* out) {(void)p;event("properties");memset(&out->properties,0,sizeof(out->properties));out->properties.apiVersion=VK_API_VERSION_1_1;out->properties.limits.timestampPeriod=0.001f;out->properties.driverVersion=7;
    VkPhysicalDeviceLimits* l=&out->properties.limits;l->maxComputeWorkGroupCount[0]=1024;l->maxComputeWorkGroupInvocations=1024;l->maxComputeWorkGroupSize[0]=1024;l->maxStorageBufferRange=1<<20;l->maxUniformBufferRange=1<<20;l->maxPerStageDescriptorStorageBuffers=8;l->maxDescriptorSetStorageBuffers=8;l->maxPerStageDescriptorUniformBuffers=8;l->maxDescriptorSetUniformBuffers=8;l->maxPerStageResources=16;l->maxBoundDescriptorSets=8;l->maxMemoryAllocationCount=4096;VkPhysicalDeviceIDProperties* id=out->pNext;memset(id->deviceUUID,1,VK_UUID_SIZE);
    if(getenv("SINTERGPU_FAKE_4070")) { const unsigned char uuid[16]={0x96,0xd2,0xdb,0xa0,0x95,0xfd,0x5c,0x00,0xe8,0x49,0x2e,0x06,0x10,0xf2,0x4e,0x06};memcpy(id->deviceUUID,uuid,16);out->properties.driverVersion=2497233024u;out->properties.vendorID=0x10de;out->properties.deviceID=0x2782;}
}
VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance i,const VkAllocationCallbacks* a) {(void)i;(void)a;event("destroy_instance");}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice p,const VkDeviceCreateInfo* i,const VkAllocationCallbacks* a,VkDevice* d) {(void)p;(void)i;(void)a;(void)d;if(getenv("SINTERGPU_FAKE_POSITIVE")){event("create_device");*d=(VkDevice)0x103;return VK_SUCCESS;}event("FORBIDDEN_create_device");return VK_ERROR_INITIALIZATION_FAILED;}

// Remaining functions are CPU fakes for the separate positive orchestration test.
// Outputs are copied from a test-provided oracle, never computed by a GPU.
typedef struct { VkDeviceSize size; void* memory; } FakeBuffer;
static FakeBuffer* bound[7];
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice p,uint32_t* n,VkQueueFamilyProperties* out) {(void)p;event("queue_properties");if(!out){*n=1;return;}memset(out,0,sizeof(*out));out->queueFlags=VK_QUEUE_COMPUTE_BIT;out->queueCount=1;out->timestampValidBits=64;*n=1;}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice p,VkPhysicalDeviceMemoryProperties* out) {(void)p;event("memory_properties");memset(out,0,sizeof(*out));out->memoryTypeCount=1;out->memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice d,uint32_t f,uint32_t n,VkQueue* out) {(void)d;(void)f;(void)n;*out=(VkQueue)0x104;}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice d,const VkBufferCreateInfo* i,const VkAllocationCallbacks* a,VkBuffer* out) {(void)d;(void)a;FakeBuffer* b=calloc(1,sizeof(*b));b->size=i->size;*out=(VkBuffer)b;event("create_buffer");return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements* out) {(void)d;out->size=((FakeBuffer*)b)->size;out->alignment=4;out->memoryTypeBits=1;}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo* i,const VkAllocationCallbacks* a,VkDeviceMemory* out) {(void)d;(void)a;*out=(VkDeviceMemory)calloc(1,i->allocationSize);event("allocate_memory");return *out?VK_SUCCESS:VK_ERROR_OUT_OF_HOST_MEMORY;}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize o) {(void)d;if(o)return VK_ERROR_INITIALIZATION_FAILED;((FakeBuffer*)b)->memory=(void*)m;event("bind_memory");return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize n,VkMemoryMapFlags f,void** out) {(void)d;(void)n;(void)f;*out=(char*)m+o;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice d,VkDeviceMemory m) {(void)d;(void)m;event("unmap_memory");}
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice d,VkBuffer b,const VkAllocationCallbacks* a) {(void)d;(void)a;free((void*)b);event("destroy_buffer");}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks* a) {(void)d;(void)a;free((void*)m);event("free_memory");}
#define MAKE_HANDLE(name,CreateInfo,Handle,value) VKAPI_ATTR VkResult VKAPI_CALL name(VkDevice d,const CreateInfo* i,const VkAllocationCallbacks* a,Handle* out) {(void)d;(void)i;(void)a;*out=(Handle)(uintptr_t)value;event(#name);return VK_SUCCESS;}
#define DESTROY_HANDLE(name,Handle) VKAPI_ATTR void VKAPI_CALL name(VkDevice d,Handle x,const VkAllocationCallbacks* a) {(void)d;(void)x;(void)a;event(#name);}
MAKE_HANDLE(vkCreateDescriptorSetLayout,VkDescriptorSetLayoutCreateInfo,VkDescriptorSetLayout,0x201)
MAKE_HANDLE(vkCreatePipelineLayout,VkPipelineLayoutCreateInfo,VkPipelineLayout,0x202)
MAKE_HANDLE(vkCreateShaderModule,VkShaderModuleCreateInfo,VkShaderModule,0x203)
MAKE_HANDLE(vkCreateDescriptorPool,VkDescriptorPoolCreateInfo,VkDescriptorPool,0x205)
MAKE_HANDLE(vkCreateCommandPool,VkCommandPoolCreateInfo,VkCommandPool,0x206)
DESTROY_HANDLE(vkDestroyDescriptorSetLayout,VkDescriptorSetLayout)
DESTROY_HANDLE(vkDestroyPipelineLayout,VkPipelineLayout)
DESTROY_HANDLE(vkDestroyShaderModule,VkShaderModule)
DESTROY_HANDLE(vkDestroyPipeline,VkPipeline)
DESTROY_HANDLE(vkDestroyDescriptorPool,VkDescriptorPool)
DESTROY_HANDLE(vkDestroyCommandPool,VkCommandPool)
static uintptr_t nextPipeline=0x204;static VkPipeline currentPipeline;
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice d,VkPipelineCache c,uint32_t n,const VkComputePipelineCreateInfo* i,const VkAllocationCallbacks* a,VkPipeline* out) {(void)d;(void)c;(void)i;(void)a;if(n!=1)return VK_ERROR_INITIALIZATION_FAILED;*out=(VkPipeline)nextPipeline++;event("create_pipeline");return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice d,const VkDescriptorSetAllocateInfo* i,VkDescriptorSet* out) {(void)d;(void)i;*out=(VkDescriptorSet)0x207;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice d,uint32_t n,const VkWriteDescriptorSet* w,uint32_t copies,const VkCopyDescriptorSet* c) {(void)d;(void)copies;(void)c;if(n!=7)abort();for(uint32_t i=0;i<n;i++){if(w[i].dstBinding>=7)abort();bound[w[i].dstBinding]=(FakeBuffer*)w[i].pBufferInfo->buffer;}for(int i=0;i<7;i++)for(int j=0;j<i;j++)if(bound[i]->memory==bound[j]->memory)abort();event("update_descriptors_unique");}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice d,const VkCommandBufferAllocateInfo* i,VkCommandBuffer* out) {(void)d;(void)i;*out=(VkCommandBuffer)0x208;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer c,const VkCommandBufferBeginInfo* i) {(void)c;(void)i;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer c) {(void)c;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer c,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags f,uint32_t n,const VkMemoryBarrier* m,uint32_t b,const VkBufferMemoryBarrier* bm,uint32_t i,const VkImageMemoryBarrier* im) {(void)c;(void)s;(void)d;(void)f;(void)n;(void)m;(void)b;(void)bm;(void)i;(void)im;event("barrier");}
VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer c,VkPipelineBindPoint b,VkPipeline p) {(void)c;(void)b;currentPipeline=p;}
VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer c,VkPipelineBindPoint b,VkPipelineLayout l,uint32_t first,uint32_t n,const VkDescriptorSet* s,uint32_t dc,const uint32_t* ds) {(void)c;(void)b;(void)l;(void)first;(void)n;(void)s;(void)dc;(void)ds;}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer c,uint32_t x,uint32_t y,uint32_t z) {(void)c;if(y!=1||z!=1||!((currentPipeline==(VkPipeline)0x204&&x==2)||(currentPipeline==(VkPipeline)0x205&&x==5)))abort();const char* path=getenv("SINTERGPU_FAKE_EXPECTED_OUTPUT");if(!path)abort();FILE* f=fopen(path,"rb");if(!f)abort();if(fseek(f,8192+16512+16512,SEEK_SET))abort();if(fread(bound[3]->memory,1,bound[3]->size,f)!=bound[3]->size)abort();if(fread(bound[4]->memory,1,bound[4]->size,f)!=bound[4]->size)abort();fclose(f);event("dispatch_fake_cpu_oracle_copy");}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q,uint32_t n,const VkSubmitInfo* i,VkFence f) {(void)q;(void)n;(void)i;(void)f;event("queue_submit");return getenv("SINTERGPU_FAKE_FAIL_SUBMIT") ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;}
static int wait_count;
VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) {(void)q;event("queue_wait");++wait_count;const char* f=getenv("SINTERGPU_FAKE_FAIL_WAIT");return f && wait_count==atoi(f) ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice d) {(void)d;event("device_wait");return getenv("SINTERGPU_FAKE_FAIL_TEARDOWN") ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice d,const VkAllocationCallbacks* a) {(void)d;(void)a;event("destroy_device");}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer c,VkCommandBufferResetFlags f){(void)c;(void)f;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer c,VkBuffer a,VkBuffer b,uint32_t n,const VkBufferCopy* r){(void)c;if(n!=1)abort();FakeBuffer *src=(FakeBuffer*)a,*dst=(FakeBuffer*)b;if(r->size>src->size||r->size>dst->size)abort();memcpy((char*)dst->memory+r->dstOffset,(char*)src->memory+r->srcOffset,r->size);}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice d,const VkQueryPoolCreateInfo* i,const VkAllocationCallbacks* a,VkQueryPool* q){(void)d;(void)i;(void)a;*q=(VkQueryPool)0x210;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice d,VkQueryPool q,const VkAllocationCallbacks* a){(void)d;(void)q;(void)a;event("destroy_query");}
VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer c,VkQueryPool q,uint32_t a,uint32_t b){(void)c;(void)q;(void)a;(void)b;}
VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer c,VkPipelineStageFlagBits s,VkQueryPool q,uint32_t i){(void)c;(void)s;(void)q;(void)i;event("timestamp_write");}
VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice d,VkQueryPool q,uint32_t f,uint32_t n,size_t size,void* out,VkDeviceSize stride,VkQueryResultFlags flags){(void)d;(void)q;(void)f;(void)size;(void)flags;if(n!=4||stride!=8)abort();uint64_t* x=out;x[0]=1;x[1]=2;x[2]=3;x[3]=4;return VK_SUCCESS;}
