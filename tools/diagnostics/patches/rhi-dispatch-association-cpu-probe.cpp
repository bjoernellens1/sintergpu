// SPDX-License-Identifier: Apache-2.0
// Synthetic state and fake Vulkan teardown callbacks; no GPU/device initialization.
#include "vulkan/vk-command.h"
#include "vulkan/vk-device.h"
#include "vulkan/vk-pipeline.h"
#include "vulkan/vk-shader-program.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
using namespace rhi;
using namespace rhi::vk;
static VKAPI_ATTR void VKAPI_CALL fakeFree(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroyPipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroyShader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroy(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
int main() {
    auto* device = new DeviceImpl(); device->addRef();
    device->m_api.vkFreeCommandBuffers = fakeFree;
    device->m_api.vkDestroyCommandPool = fakeDestroy;
    device->m_api.vkDestroyPipeline = fakeDestroyPipeline;
    device->m_api.vkDestroyShaderModule = fakeDestroyShader;
    auto* command = new CommandBufferImpl(device, nullptr); command->addRef();
    auto before = command->getReferenceCount();
    try {
        Slang::ComPtr<IVulkanRecordedDispatchObservationV1> guard;
        assert(SLANG_SUCCEEDED(command->queryInterface(IVulkanRecordedDispatchObservationV1::getTypeGuid(), reinterpret_cast<void**>(guard.writeRef()))));
        assert(command->getReferenceCount()==before+1);
        throw std::runtime_error("synthetic conversion failure");
    } catch (const std::runtime_error&) {}
    assert(command->getReferenceCount()==before);
    IVulkanRecordedDispatchObservationV1* query = nullptr;
    assert(SLANG_SUCCEEDED(command->queryInterface(IVulkanRecordedDispatchObservationV1::getTypeGuid(), (void**)&query)));
    uint32_t count=0; assert(SLANG_FAILED(query->getDispatchCount(&count)));
    command->m_observationReady=true; command->m_observationEligible=true; command->m_observationGeneration=7;
    command->m_commandBuffer=(VkCommandBuffer)0x1234;
    // Synthetic specialized program: concrete pipeline references this program,
    // while a different parent program exists but is not selected by it.
    auto* parent = new ShaderProgramImpl(device, ShaderProgramDesc{}); parent->addRef();
    auto* concrete = new ShaderProgramImpl(device, ShaderProgramDesc{}); concrete->addRef();
    concrete->m_compiledShaders = true;
    concrete->m_modules.emplace_back();
    auto& module = concrete->m_modules.back();
    module.shaderModule=(VkShaderModule)0x25; module.observedCreationSucceeded=true;
    module.observedCode={0x07230203u,0x12345678u};
    VkPipelineShaderStageCreateInfo stage{}; stage.module=module.shaderModule;
    stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.pName="main";
    concrete->m_stageCreateInfos.push_back(stage);
    ComputePipelineDesc desc{}; desc.program=concrete;
    auto* pipeline = new ComputePipelineImpl(device,desc); pipeline->addRef();
    pipeline->m_pipeline=(VkPipeline)0x15;
    pipeline->m_dispatchObservationReady=true;
    pipeline->m_dispatchObservationModule=module.shaderModule;
    pipeline->m_dispatchObservationCode=module.observedCode;
    auto* second = new ComputePipelineImpl(device,desc); second->addRef();
    second->m_pipeline=(VkPipeline)0x16;
    second->m_dispatchObservationReady=true;
    second->m_dispatchObservationModule=module.shaderModule;
    second->m_dispatchObservationCode=module.observedCode;
    assert(appendDirectDispatchObservation(command,pipeline,5,1,1));
    assert(appendDirectDispatchObservation(command,second,2,3,4));
    assert(command->m_dispatchObservations.size()==2);
    auto& item=command->m_dispatchObservations[0];
    assert(command->m_dispatchObservations[1].groups[0]==2 && command->m_dispatchObservations[1].groups[1]==3 && command->m_dispatchObservations[1].groups[2]==4);
    second->m_dispatchObservationReady = false;
    assert(!appendDirectDispatchObservation(command,second,1,1,1));
    assert(command->m_dispatchObservations.size()==2);
    second->m_dispatchObservationReady = true;
    assert(SLANG_SUCCEEDED(query->getDispatchCount(&count)) && count==2);
    VulkanRecordedDispatchV1 out{};
    assert(SLANG_FAILED(query->getDispatch(0,nullptr)));
    out.schemaVersion=2;
    assert(SLANG_FAILED(query->getDispatch(0,&out)));
    out.schemaVersion=1;
    assert(SLANG_SUCCEEDED(query->getDispatch(0,&out)));
    assert(out.recordingGeneration==7 && out.commandBuffer==0x1234 && out.pipeline==0x15 && out.shaderModule==0x25);
    assert(out.groups[0]==5 && out.groups[1]==1 && out.groups[2]==1 && out.codeSize==8);
    assert(std::memcmp(out.code,item.code.data(),out.codeSize)==0);
    assert(SLANG_SUCCEEDED(query->getDispatch(1,&out)) && out.pipeline==0x16 && out.groups[0]==2 && out.groups[1]==3 && out.groups[2]==4);
    assert(SLANG_FAILED(query->getDispatch(2,&out)));
    command->m_observationEligible=false;
    assert(SLANG_FAILED(query->getDispatchCount(&count)));
    command->m_observationReady=false;
    command->m_commandBuffer=VK_NULL_HANDLE; // fake destructor callbacks, no real Vulkan
    command->release();
    assert(SLANG_FAILED(query->getDispatchCount(&count)));
    query->release(); second->release(); pipeline->release(); concrete->release(); parent->release(); device->release();
}
