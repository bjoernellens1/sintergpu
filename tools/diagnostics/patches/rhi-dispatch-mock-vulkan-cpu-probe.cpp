// SPDX-License-Identifier: Apache-2.0
// Test-only: compile the actual command recorder into this CPU executable and
// supply fake Vulkan functions. No Vulkan loader, device or GPU is initialized.
#include "vulkan/vk-command.cpp"
#include "vulkan/vk-shader-program.h"
#include "vulkan/vk-shader-object-layout.h"
#include <cassert>
#include <cstring>
#include <vector>
using namespace rhi;
using namespace rhi::vk;
struct MockDebug : IDebugCallback {
    int errors = 0;
    SLANG_NO_THROW void SLANG_MCALL handleMessage(DebugMessageType type, DebugMessageSource, const char*) override {
        if (type == DebugMessageType::Error) ++errors;
    }
};
static bool endFails = false, beginFails = false, resetFails = false, createFails = false;
static unsigned bindCount = 0, dispatchCount = 0, indirectCount = 0, callbackCount = 0, createCount = 0;
static VKAPI_ATTR VkResult VKAPI_CALL fakeBegin(VkCommandBuffer,const VkCommandBufferBeginInfo*) { return beginFails ? VK_ERROR_UNKNOWN : VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fakeEnd(VkCommandBuffer) { return endFails ? VK_ERROR_UNKNOWN : VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fakeReset(VkDevice,VkCommandPool,VkCommandPoolResetFlags) { return resetFails ? VK_ERROR_UNKNOWN : VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fakeBind(VkCommandBuffer,VkPipelineBindPoint,VkPipeline) { ++bindCount; }
static VKAPI_ATTR void VKAPI_CALL fakeDispatch(VkCommandBuffer,uint32_t,uint32_t,uint32_t) { ++dispatchCount; }
static VKAPI_ATTR void VKAPI_CALL fakeIndirect(VkCommandBuffer,VkBuffer,VkDeviceSize) { ++indirectCount; }
static VKAPI_ATTR void VKAPI_CALL fakeBarrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,VkDependencyFlags,
    uint32_t,const VkMemoryBarrier*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*) {}
static VKAPI_ATTR void VKAPI_CALL fakeFree(VkDevice,VkCommandPool,uint32_t,const VkCommandBuffer*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroyPool(VkDevice,VkCommandPool,const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroyPipeline(VkDevice,VkPipeline,const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL fakeDestroyShader(VkDevice,VkShaderModule,const VkAllocationCallbacks*) {}
static void SLANG_MCALL fakeCallback(const ExecuteCallbackContext*,void*,const void*,Size) { ++callbackCount; }
static VKAPI_ATTR VkResult VKAPI_CALL fakeCreatePipeline(VkDevice,VkPipelineCache,uint32_t count,
    const VkComputePipelineCreateInfo* infos,const VkAllocationCallbacks*,VkPipeline* output) {
    assert(count==1 && infos && infos[0].stage.module==(VkShaderModule)0x25);
    ++createCount; if (createFails) return VK_ERROR_UNKNOWN;
    *output=(VkPipeline)(uintptr_t)(0x100+createCount); return VK_SUCCESS;
}
class FakeBuffer : public BufferImpl {
public:
    FakeBuffer(Device* device,const BufferDesc& desc) : BufferImpl(device,desc) {}
    void deleteThis() override { delete this; }
};
static void addDirect(CommandBufferImpl* c, ComputePipelineImpl* p, BindingDataImpl* binding, uint32_t x, IBuffer* indirect=nullptr) {
    c->m_commandList.write(commands::BeginComputePass{});
    commands::SetComputeState state{}; state.pipeline=p;state.bindingData=binding;
    c->m_commandList.write(std::move(state));
    commands::DispatchCompute dispatch{}; dispatch.x=x;dispatch.y=1;dispatch.z=1;
    c->m_commandList.write(std::move(dispatch));
    if (indirect) {
        commands::DispatchComputeIndirect i{}; i.argBuffer.buffer=indirect;
        c->m_commandList.write(std::move(i));
    }
    c->m_commandList.write(commands::EndComputePass{});
}
int main() {
    auto* device = new DeviceImpl(); device->addRef();
    MockDebug debug; device->m_debugCallback=&debug;
    auto& api=device->m_api;
    api.vkBeginCommandBuffer=fakeBegin; api.vkEndCommandBuffer=fakeEnd;
    api.vkCmdBindPipeline=fakeBind;api.vkCmdDispatch=fakeDispatch;
    api.vkCmdDispatchIndirect=fakeIndirect;api.vkCmdPipelineBarrier=fakeBarrier;
    api.vkFreeCommandBuffers=fakeFree;api.vkDestroyCommandPool=fakeDestroyPool;
    api.vkDestroyPipeline=fakeDestroyPipeline;api.vkDestroyShaderModule=fakeDestroyShader;
    api.vkCreateComputePipelines=fakeCreatePipeline;api.vkResetCommandPool=fakeReset;
    auto* program=new ShaderProgramImpl(device,ShaderProgramDesc{});program->addRef();
    program->m_compiledShaders=true;program->m_modules.emplace_back();
    auto& module=program->m_modules.back();module.shaderModule=(VkShaderModule)0x25;
    module.observedCreationSucceeded=true;module.observedCode={0x07230203u,0x12345678u};
    VkPipelineShaderStageCreateInfo stage{};stage.module=module.shaderModule;
    stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;stage.pName="main";
    program->m_stageCreateInfos.push_back(stage);
    program->m_rootShaderObjectLayout=new RootShaderObjectLayoutImpl();
    ComputePipelineDesc desc{};desc.program=program;
    IComputePipeline* raw=nullptr;
    assert(SLANG_SUCCEEDED(device->createComputePipeline2(desc,&raw)) && raw && createCount==1);
    auto* pipeline=checked_cast<ComputePipelineImpl*>(raw);
    assert(pipeline->m_dispatchObservationReady && pipeline->m_dispatchObservationCode==module.observedCode);
    // A non-null Vulkan specialization payload is deliberately unobservable.
    VkSpecializationInfo unsupported{};
    program->m_stageCreateInfos[0].pSpecializationInfo=&unsupported;
    IComputePipeline* unsupportedRaw=nullptr;
    assert(SLANG_SUCCEEDED(device->createComputePipeline2(desc,&unsupportedRaw)) && unsupportedRaw);
    assert(!checked_cast<ComputePipelineImpl*>(unsupportedRaw)->m_dispatchObservationReady);
    unsupportedRaw->release();
    program->m_stageCreateInfos[0].pSpecializationInfo=nullptr;
    createFails=true; IComputePipeline* failed=nullptr;
    assert(SLANG_FAILED(device->createComputePipeline2(desc,&failed)) && !failed);
    createFails=false;
    auto* command=new CommandBufferImpl(device,nullptr);command->addRef();command->m_commandBuffer=(VkCommandBuffer)0x1234;
    BindingDataImpl binding{};
    addDirect(command,pipeline,&binding,5);
    CommandRecorder recorder(device);
    assert(SLANG_SUCCEEDED(recorder.record(command)));
    assert(bindCount==1 && dispatchCount==1 && command->m_observationReady);
    uint32_t count=0;assert(SLANG_SUCCEEDED(command->getDispatchCount(&count)) && count==1);
    VulkanRecordedDispatchV1 obs{};assert(SLANG_SUCCEEDED(command->getDispatch(0,&obs)));
    assert(obs.pipeline==0x101 && obs.shaderModule==0x25 && obs.groups[0]==5 && obs.codeSize==8);
    auto oldGeneration=obs.recordingGeneration;
    assert(SLANG_SUCCEEDED(command->reset()) && SLANG_FAILED(command->getDispatchCount(&count)));
    addDirect(command,pipeline,&binding,3);
    CommandRecorder reuse(device);assert(SLANG_SUCCEEDED(reuse.record(command)));
    assert(SLANG_SUCCEEDED(command->getDispatch(0,&obs)) && obs.recordingGeneration!=oldGeneration && obs.groups[0]==3);
    resetFails=true;
    assert(SLANG_FAILED(command->reset()) && SLANG_FAILED(command->getDispatchCount(&count)));
    resetFails=false;
    assert(SLANG_SUCCEEDED(command->reset()));
    BufferDesc bufferDesc{};bufferDesc.size=64;bufferDesc.defaultState=ResourceState::IndirectArgument;
    auto* indirectBuffer=new FakeBuffer(device,bufferDesc);indirectBuffer->addRef();
    indirectBuffer->m_buffer.m_buffer=(VkBuffer)0xabc;
    addDirect(command,pipeline,&binding,2,indirectBuffer);
    CommandRecorder indirectRecorder(device);assert(SLANG_SUCCEEDED(indirectRecorder.record(command)));
    assert(indirectCount==1 && SLANG_FAILED(command->getDispatchCount(&count)));
    assert(SLANG_SUCCEEDED(command->reset()));
    indirectBuffer->release();
    addDirect(command,pipeline,&binding,2);
    commands::ExecuteCallback callback{};callback.desc.callback=fakeCallback;command->m_commandList.write(std::move(callback));
    CommandRecorder callbackRecorder(device);assert(SLANG_SUCCEEDED(callbackRecorder.record(command)));
    assert(callbackCount==1 && SLANG_FAILED(command->getDispatchCount(&count)));
    assert(SLANG_SUCCEEDED(command->reset()));addDirect(command,pipeline,&binding,1);
    beginFails=true;CommandRecorder beginFailRecorder(device);assert(SLANG_FAILED(beginFailRecorder.record(command)));
    assert(SLANG_FAILED(command->getDispatchCount(&count)));
    beginFails=false;
    endFails=true;CommandRecorder failRecorder(device);assert(SLANG_FAILED(failRecorder.record(command)));
    assert(SLANG_FAILED(command->getDispatchCount(&count)) && debug.errors==3);
    endFails=false;command->m_commandBuffer=VK_NULL_HANDLE;
    command->release();raw->release();program->release();device->release();
}
