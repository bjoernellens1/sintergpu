// SPDX-License-Identifier: Apache-2.0
// Experimental command recording only. The caller owns device, buffers, memory,
// allocation-disjointness, command buffer, submission, waiting and timing.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>

struct SinterTransferDispatch { PFN_vkCmdCopyBuffer copy; PFN_vkCmdPipelineBarrier barrier; };
struct SinterTransferBuffer { VkBuffer upload; VkBuffer device; VkBuffer download; VkDeviceSize size; };

// Exactly seven nonempty buffers: six fixture buffers and one parameter buffer.
// Caller has initialized/flushed upload memory, provided allocation-disjoint
// buffers, and will wait/invalidate downloads before reading. Recording performs
// no CPU transfers, submission, waits, allocation or validation.
template<typename RecordDispatch>
void sinterRecordTransferPass(VkCommandBuffer command, const std::array<SinterTransferBuffer,7>& buffers,
                              SinterTransferDispatch api, RecordDispatch dispatch) {
    if(!command || !api.copy || !api.barrier)throw std::invalid_argument("transfer command/dispatch");
    for(size_t i=0;i<buffers.size();++i){
        const auto& b=buffers[i];
        if(!b.upload || !b.device || !b.download || !b.size)throw std::invalid_argument("transfer buffer");
        const VkBuffer handles[3]={b.upload,b.device,b.download};
        for(int x=0;x<3;++x){for(size_t j=0;j<i;++j){const auto& p=buffers[j];if(handles[x]==p.upload||handles[x]==p.device||handles[x]==p.download)throw std::invalid_argument("shared buffer handle");}for(int y=0;y<x;++y)if(handles[x]==handles[y])throw std::invalid_argument("shared buffer handle");}
    }
    VkMemoryBarrier before{};before.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;before.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;before.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    api.barrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&before,0,nullptr,0,nullptr);
    for(const auto& b:buffers){VkBufferCopy region{0,0,b.size};api.copy(command,b.upload,b.device,1,&region);}
    VkMemoryBarrier upload{};upload.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;upload.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;upload.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
    api.barrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&upload,0,nullptr,0,nullptr);
    dispatch(command);
    VkMemoryBarrier compute{};compute.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;compute.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;compute.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    api.barrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&compute,0,nullptr,0,nullptr);
    for(const auto& b:buffers){VkBufferCopy region{0,0,b.size};api.copy(command,b.device,b.download,1,&region);}
    VkMemoryBarrier done{};done.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;done.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;done.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    api.barrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&done,0,nullptr,0,nullptr);
}
