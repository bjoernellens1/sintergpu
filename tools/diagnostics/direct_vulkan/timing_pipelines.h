// SPDX-License-Identifier: Apache-2.0
// Caller verifies SPIR-V identities and descriptor ABI before constructing this.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>
#include <vector>
struct SinterPipelineAPI { PFN_vkCreateShaderModule createShader; PFN_vkDestroyShaderModule destroyShader; PFN_vkCreateComputePipelines createPipelines; PFN_vkDestroyPipeline destroyPipeline; };
class SinterTimingPipelines {
public:
 SinterTimingPipelines(VkDevice d,VkPipelineLayout layout,const std::array<std::vector<uint32_t>,2>& words,SinterPipelineAPI api):device_(d),api_(api){
  if(!d||!layout||!api.createShader||!api.destroyShader||!api.createPipelines||!api.destroyPipeline)throw std::invalid_argument("pipeline API");
  try{for(size_t i=0;i<2;++i){if(words[i].empty()||words[i][0]!=0x07230203)throw std::invalid_argument("SPIR-V magic");VkShaderModuleCreateInfo sm{};sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;sm.codeSize=words[i].size()*sizeof(uint32_t);sm.pCode=words[i].data();check(api_.createShader(device_,&sm,nullptr,&shaders_[i]));if(!shaders_[i])throw std::runtime_error("null shader");VkComputePipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;pi.layout=layout;pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=shaders_[i];pi.stage.pName="main";check(api_.createPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipelines_[i]));if(!pipelines_[i])throw std::runtime_error("null pipeline");}}
  catch(...){clear();throw;}
 }
 SinterTimingPipelines(const SinterTimingPipelines&)=delete;SinterTimingPipelines& operator=(const SinterTimingPipelines&)=delete;
 ~SinterTimingPipelines(){clear();}
 VkPipeline at(size_t i)const{return pipelines_.at(i);}
private:
 static void check(VkResult result){if(result!=VK_SUCCESS)throw std::runtime_error("pipeline creation failed");}
 void clear()noexcept{for(auto p:pipelines_)if(p)api_.destroyPipeline(device_,p,nullptr);for(auto s:shaders_)if(s)api_.destroyShader(device_,s,nullptr);pipelines_={};shaders_={};}
 VkDevice device_;SinterPipelineAPI api_;std::array<VkShaderModule,2> shaders_{};std::array<VkPipeline,2> pipelines_{};
};
