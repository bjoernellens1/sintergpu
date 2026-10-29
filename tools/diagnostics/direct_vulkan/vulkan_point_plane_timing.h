// SPDX-License-Identifier: Apache-2.0
// Experimental actual Vulkan submission/timing primitive, not a benchmark runner.
// Caller owns device/queue/command buffer and the canonical GPU lock. This code
// does not authenticate identities, memory backing, clocks or qualification.
#pragma once
#include "vulkan_point_plane_record.h"
#include <array>
#include <chrono>
#include <cstring>
#include <limits>

struct SinterTimingDispatch {
    PFN_vkResetCommandBuffer resetCommand;
    PFN_vkBeginCommandBuffer begin;
    PFN_vkEndCommandBuffer end;
    PFN_vkQueueSubmit submit;
    PFN_vkQueueWaitIdle wait;
    PFN_vkCmdCopyBuffer copy;
    SinterPointPlaneDispatch point;
};
struct SinterTimingStaging {
    SinterTransferBuffer gpu;
    void* mappedUpload;
    const void* mappedDownload;
    const void* initialBytes;
    void* outputBytes;
};
// Optional host-clock subdivision of the same transfer interval. Clock reads
// perturb the measured path; this is diagnostic, not an acceptance measurement.
struct SinterTransferComponents { uint64_t uploadNs, recordNs, submitWaitNs, downloadNs; };
struct SinterResidentTiming { std::array<uint64_t,4> rawTicks; uint64_t hostCompletionUpperBoundNs; };

// The seven staging triplets must be distinct and backed by host-coherent
// memory, with device-visible barriers and valid sizes. Caller verifies memory
// properties, bindings, aliasing, actual device limits and lock independently.
// mark*Validated is caller attestation only: production adapters must check every
// input/output/guard byte against donor and independent oracle before calling it.
class SinterPointPlaneTiming {
public:
    SinterPointPlaneTiming(VkQueue queue,VkCommandBuffer command,SinterTimingDispatch api,SinterTimestampPair& timestamps)
        : queue_(queue),command_(command),api_(api),timestamps_(timestamps) {
        if(!queue || !command || !api.resetCommand || !api.begin || !api.end || !api.submit || !api.wait || !api.copy)
            throw std::invalid_argument("timing dispatch unavailable");
    }
    // Caller resets poisoned device outputs before this call. Recording is
    // outside the host submission bound; query reset and two barriers are inside
    // the GPU pipeline markers. No raw readback occurs in this timed interval.
    SinterResidentTiming resident(const SinterPointPlaneCommand& p) {
        require(0);state_=-1;begin();sinterRecordResidentPointPlane(command_,p,api_.point,timestamps_);end();
        auto before=Clock::now();submitAndWait();auto elapsed=ns(before,Clock::now());
        std::array<uint64_t,4> ticks{};VkResult result=timestamps_.read(ticks);
        if(result!=VK_SUCCESS)throw std::runtime_error("timestamps unavailable after queue completion");
        state_=1;return {ticks,elapsed};
    }
    void markResidentValidated(){require(1);state_=2;}
    // Caller must reset output poison before this call. Interval covers all seven
    // initial host uploads, command record, submit/wait and seven raw host reads;
    // output interpretation, oracle validation and reset are outside the interval.
    // Recording and its argument checks are included. A failed operation poisons
    // this instance; caller must tear down after any recording/submission error.
    uint64_t transfer(const SinterPointPlaneCommand& p,const std::array<SinterTimingStaging,7>& slots, SinterTransferComponents* diagnostic=nullptr) {
        require(2);std::array<SinterTransferBuffer,7> buffers{};
        for(size_t i=0;i<slots.size();++i){const auto& x=slots[i];if(!x.mappedUpload||!x.mappedDownload||!x.initialBytes||!x.outputBytes||!x.gpu.size||x.gpu.size>std::numeric_limits<size_t>::max())throw std::invalid_argument("staging mapping");buffers[i]=x.gpu;}
        state_=-1;auto before=Clock::now();
        for(const auto& x:slots)std::memcpy(x.mappedUpload,x.initialBytes,(size_t)x.gpu.size);
        auto afterUpload=diagnostic?Clock::now():before;
        begin();sinterRecordTransferPointPlane(command_,p,buffers,api_.point,api_.copy);end();
        auto afterRecord=diagnostic?Clock::now():before;
        submitAndWait();
        auto afterWait=diagnostic?Clock::now():before;
        for(const auto& x:slots)std::memcpy(x.outputBytes,x.mappedDownload,(size_t)x.gpu.size);
        auto afterDownload=Clock::now();
        uint64_t elapsed=ns(before,afterDownload);
        if(diagnostic){
            auto part=[](Clock::time_point a,Clock::time_point b){return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();};
            auto upload=part(before,afterUpload),record=part(before,afterRecord),wait=part(before,afterWait);
            *diagnostic={upload,record-upload,wait-record,elapsed-wait};
        }
        state_=3;return elapsed;
    }
    void markTransferValidated(){require(3);state_=0;}
private:
    using Clock=std::chrono::steady_clock;
    static uint64_t ns(Clock::time_point a,Clock::time_point b){auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();if(n<=0)throw std::runtime_error("host clock interval");return (uint64_t)n;}
    void require(int state){if(state_!=state)throw std::logic_error("validation/order required");}
    static void check(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("Vulkan timing operation failed");}
    void begin(){check(api_.resetCommand(command_,0));VkCommandBufferBeginInfo info{};info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;check(api_.begin(command_,&info));}
    void end(){check(api_.end(command_));}
    void submitAndWait(){VkSubmitInfo info{};info.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;info.commandBufferCount=1;info.pCommandBuffers=&command_;check(api_.submit(queue_,1,&info,VK_NULL_HANDLE));check(api_.wait(queue_));}
    VkQueue queue_;VkCommandBuffer command_;SinterTimingDispatch api_;SinterTimestampPair& timestamps_;int state_=0;
};
