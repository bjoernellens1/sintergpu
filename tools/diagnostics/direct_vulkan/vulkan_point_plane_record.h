// SPDX-License-Identifier: Apache-2.0
// Experimental native point-plane command recording; caller owns all resources,
// completion, validated fixture data and timing. No device, queue or wait here.
#pragma once
#include "vulkan_timestamp_pair.h"
#include "vulkan_transfer_pass.h"

struct SinterPointPlaneDispatch {
    PFN_vkCmdBindPipeline bindPipeline;
    PFN_vkCmdBindDescriptorSets bindSets;
    PFN_vkCmdDispatch dispatch;
    PFN_vkCmdPipelineBarrier barrier;
};
struct SinterPointPlaneCommand {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptors;
    uint32_t groups;
};

inline void sinterRecordPointPlaneDispatch(VkCommandBuffer cmd,const SinterPointPlaneCommand& p,SinterPointPlaneDispatch api) {
    if(!cmd||!p.pipeline||!p.layout||!p.descriptors||!p.groups||!api.bindPipeline||!api.bindSets||!api.dispatch)
        throw std::invalid_argument("point-plane command");
    api.bindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.pipeline);
    api.bindSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.layout,0,1,&p.descriptors,0,nullptr);
    api.dispatch(cmd,p.groups,1,1);
}

// Resident pipeline includes barriers and dispatch; query pool reset is before
// the first timestamp. Caller waits for completion and validates all output
// before resetting any buffers. Query semantics and timestamp period require
// hardware qualification; the two compute-stage timestamps are diagnostic.
inline void sinterRecordResidentPointPlane(VkCommandBuffer cmd,const SinterPointPlaneCommand& p,
                                            SinterPointPlaneDispatch api,SinterTimestampPair& times) {
    if(!api.barrier || !api.bindPipeline || !api.bindSets || !api.dispatch || !cmd || !p.pipeline || !p.layout || !p.descriptors || !p.groups)
        throw std::invalid_argument("resident command");
    times.beginPipeline(cmd);
    VkMemoryBarrier before{};before.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;before.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_UNIFORM_READ_BIT;
    api.barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);
    times.beginKernel(cmd);sinterRecordPointPlaneDispatch(cmd,p,api);times.endKernel(cmd);
    VkMemoryBarrier after{};after.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;after.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    api.barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&after,0,nullptr,0,nullptr);
    times.endPipeline(cmd);
}

inline void sinterRecordTransferPointPlane(VkCommandBuffer cmd,const SinterPointPlaneCommand& p,
                                            const std::array<SinterTransferBuffer,7>& buffers,
                                            SinterPointPlaneDispatch api,PFN_vkCmdCopyBuffer copy) {
    if(!api.bindPipeline||!api.bindSets||!api.dispatch||!p.pipeline||!p.layout||!p.descriptors||!p.groups)
        throw std::invalid_argument("transfer point-plane command");
    sinterRecordTransferPass(cmd,buffers,{copy,api.barrier},[&](VkCommandBuffer c){sinterRecordPointPlaneDispatch(c,p,api);});
}
