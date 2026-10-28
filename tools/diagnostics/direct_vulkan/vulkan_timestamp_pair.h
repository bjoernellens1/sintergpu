// SPDX-License-Identifier: Apache-2.0
// Explicit caller-owned Vulkan device/command buffer; no submission, wait,
// allocation, synchronization or performance decision is hidden in recording.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>

struct SinterTimestampDispatch {
    PFN_vkCreateQueryPool create;
    PFN_vkDestroyQueryPool destroy;
    PFN_vkCmdResetQueryPool reset;
    PFN_vkCmdWriteTimestamp write;
    PFN_vkGetQueryPoolResults read;
};

class SinterTimestampPair {
public:
    SinterTimestampPair(VkDevice device, uint32_t validBits, float period, SinterTimestampDispatch api)
        : device_(device),bits_(validBits),period_(period),api_(api) {
        if(!device || !validBits || validBits>64 || (!std::isfinite(period) || !(period>0)) || !api.create || !api.destroy || !api.reset || !api.write || !api.read)
            throw std::invalid_argument("timestamp capability or dispatch unavailable");
        VkQueryPoolCreateInfo desc{};desc.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;desc.queryType=VK_QUERY_TYPE_TIMESTAMP;desc.queryCount=4;
        if(api_.create(device_,&desc,nullptr,&pool_)!=VK_SUCCESS || !pool_)throw std::runtime_error("timestamp query pool creation failed");
    }
    SinterTimestampPair(const SinterTimestampPair&)=delete;
    SinterTimestampPair& operator=(const SinterTimestampPair&)=delete;
    ~SinterTimestampPair(){if(pool_)api_.destroy(device_,pool_,nullptr);}
    // Must call in order, on one command buffer, only after prior result read.
    void beginPipeline(VkCommandBuffer command) {
        if(!command || state_!=0)throw std::logic_error("timestamp begin pipeline order");
        command_=command;api_.reset(command,pool_,0,4);api_.write(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,pool_,0);state_=1;
    }
    void beginKernel(VkCommandBuffer command){write(command,1,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,1,2);}
    void endKernel(VkCommandBuffer command){write(command,2,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,2,3);}
    void endPipeline(VkCommandBuffer command){write(command,3,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,3,4);}
    // Caller waits for its own submission first. Never uses WAIT_BIT or waits here.
    VkResult read(std::array<uint64_t,4>& ticks) {
        if(state_!=4)throw std::logic_error("timestamp read order");
        auto result=api_.read(device_,pool_,0,4,sizeof(ticks),ticks.data(),sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
        if(result==VK_SUCCESS){const uint64_t mask=bits_==64?~uint64_t(0):((uint64_t(1)<<bits_)-1);for(auto x:ticks)if(x&~mask)throw std::runtime_error("timestamp exceeds valid bits");state_=0;command_=VK_NULL_HANDLE;}
        return result;
    }
    uint32_t validBits()const{return bits_;}
    float periodNs()const{return period_;}
private:
    void write(VkCommandBuffer command,uint32_t query,VkPipelineStageFlagBits stage,int expected,int next){
        if(!command || command!=command_ || state_!=expected)throw std::logic_error("timestamp command/order");
        api_.write(command,stage,pool_,query);state_=next;
    }
    VkDevice device_;uint32_t bits_;float period_;SinterTimestampDispatch api_;VkQueryPool pool_=VK_NULL_HANDLE;VkCommandBuffer command_=VK_NULL_HANDLE;int state_=0;
};
