// SPDX-License-Identifier: Apache-2.0
#include "timing_pipelines.h"
#include <cassert>
static int shaders=0,destroyedShaders=0,pipelines=0,destroyedPipelines=0,fail=0;
static VKAPI_ATTR VkResult VKAPI_CALL cs(VkDevice,const VkShaderModuleCreateInfo* p,const VkAllocationCallbacks*,VkShaderModule* o){assert(p->codeSize==8&&p->pCode[0]==0x07230203);*o=(VkShaderModule)(uintptr_t)(++shaders);return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL ds(VkDevice,VkShaderModule,const VkAllocationCallbacks*){++destroyedShaders;}
static VKAPI_ATTR VkResult VKAPI_CALL cp(VkDevice,VkPipelineCache,uint32_t n,const VkComputePipelineCreateInfo* p,const VkAllocationCallbacks*,VkPipeline* o){assert(n==1&&p->layout&&p->stage.module);if(fail&&pipelines+1==fail)return VK_ERROR_OUT_OF_DEVICE_MEMORY;*o=(VkPipeline)(uintptr_t)(++pipelines);return VK_SUCCESS;}
static VKAPI_ATTR void VKAPI_CALL dp(VkDevice,VkPipeline,const VkAllocationCallbacks*){++destroyedPipelines;}
int main(){std::array<std::vector<uint32_t>,2> words{{{0x07230203,1},{0x07230203,2}}};SinterPipelineAPI api{cs,ds,cp,dp};{SinterTimingPipelines p((VkDevice)1,(VkPipelineLayout)1,words,api);assert(p.at(0)!=p.at(1));}assert(shaders==2&&destroyedShaders==2&&pipelines==2&&destroyedPipelines==2);shaders=destroyedShaders=pipelines=destroyedPipelines=0;fail=2;try{SinterTimingPipelines p((VkDevice)1,(VkPipelineLayout)1,words,api);assert(false);}catch(const std::runtime_error&){}assert(shaders==2&&destroyedShaders==2&&pipelines==1&&destroyedPipelines==1);}
