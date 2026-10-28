// SPDX-License-Identifier: Apache-2.0
// Explicit reset/upload and complete readback outside measured intervals.
#pragma once
#include "timing_resources.h"
#include <cstring>
#include <vector>
struct SinterReadbackAPI { PFN_vkResetCommandBuffer reset;PFN_vkBeginCommandBuffer begin;PFN_vkEndCommandBuffer end;PFN_vkQueueSubmit submit;PFN_vkQueueWaitIdle wait;PFN_vkCmdCopyBuffer copy;PFN_vkCmdPipelineBarrier barrier; };
inline void sinterFixtureTransfer(VkCommandBuffer cmd,VkQueue queue,const SinterTimingResources& resources,SinterReadbackAPI api,bool upload){
 if(!cmd||!queue||!api.reset||!api.begin||!api.end||!api.submit||!api.wait||!api.copy||!api.barrier)throw std::invalid_argument("fixture transfer API");
 auto check=[](VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("fixture transfer failed");};
 check(api.reset(cmd,0));VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;check(api.begin(cmd,&begin));
 VkMemoryBarrier b{};b.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
 b.srcAccessMask=upload?VK_ACCESS_HOST_WRITE_BIT:VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
 b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
 api.barrier(cmd,upload?VK_PIPELINE_STAGE_HOST_BIT:VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);
 for(size_t i=0;i<7;++i){auto& src=resources.at(i,upload?0:1);auto& dst=resources.at(i,upload?1:2);VkBufferCopy region{0,0,src.size};api.copy(cmd,src.buffer,dst.buffer,1,&region);}
 b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=upload?VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_UNIFORM_READ_BIT:VK_ACCESS_HOST_READ_BIT;
 api.barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,upload?VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT:VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);
 check(api.end(cmd));VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;check(api.submit(queue,1,&si,VK_NULL_HANDLE));check(api.wait(queue));
}
