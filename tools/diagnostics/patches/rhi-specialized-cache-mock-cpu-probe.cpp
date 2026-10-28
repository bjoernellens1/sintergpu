// SPDX-License-Identifier: Apache-2.0
// Test-only fake Vulkan: exercise actual cached specialization resolution and recorder.
// No Vulkan loader/device or GPU initialized; specialization is seeded, not compiled.
#include "vulkan/vk-command.cpp"
#include "vulkan/vk-shader-program.h"
#include "vulkan/vk-shader-object-layout.h"
#include <cassert>
#include <cstring>
using namespace rhi;
using namespace rhi::vk;
static unsigned pipelineCreates=0, binds=0, dispatches=0;
static VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer,const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL end(VkCommandBuffer) { return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer,VkPipelineBindPoint,VkPipeline) { ++binds; }
static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer,uint32_t,uint32_t,uint32_t) { ++dispatches; }
static VKAPI_ATTR void VKAPI_CALL freeCmd(VkDevice,VkCommandPool,uint32_t,const VkCommandBuffer*) {}
static VKAPI_ATTR void VKAPI_CALL destroyPool(VkDevice,VkCommandPool,const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL destroyPipe(VkDevice,VkPipeline,const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL destroyShader(VkDevice,VkShaderModule,const VkAllocationCallbacks*) {}
static VKAPI_ATTR VkResult VKAPI_CALL createPipe(VkDevice,VkPipelineCache,uint32_t n,const VkComputePipelineCreateInfo* info,const VkAllocationCallbacks*,VkPipeline* out) {
    assert(n==1 && info->stage.module); ++pipelineCreates; *out=(VkPipeline)(uintptr_t)(0x100+pipelineCreates); return VK_SUCCESS;
}
static ShaderProgramImpl* program(DeviceImpl* device, uint64_t shader, uint32_t word) {
    auto* p=new ShaderProgramImpl(device,ShaderProgramDesc{});p->addRef();p->m_compiledShaders=true;
    p->m_modules.emplace_back();auto& mod=p->m_modules.back();mod.shaderModule=(VkShaderModule)shader;
    mod.observedCreationSucceeded=true;mod.observedCode={0x07230203u,word};
    VkPipelineShaderStageCreateInfo stage{};stage.module=mod.shaderModule;stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;stage.pName="main";
    p->m_stageCreateInfos.push_back(stage);p->m_rootShaderObjectLayout=new RootShaderObjectLayoutImpl();return p;
}
static IComputePipeline* pipeline(DeviceImpl* device, ShaderProgramImpl* p) {
    ComputePipelineDesc d{};d.program=p;IComputePipeline* out=nullptr;
    assert(SLANG_SUCCEEDED(device->createComputePipeline2(d,&out)) && out);return out;
}
int main() {
    auto* device=new DeviceImpl();device->addRef(); auto& api=device->m_api;
    api.vkBeginCommandBuffer=begin;api.vkEndCommandBuffer=end;api.vkCmdBindPipeline=bind;api.vkCmdDispatch=dispatch;
    api.vkFreeCommandBuffers=freeCmd;api.vkDestroyCommandPool=destroyPool;api.vkDestroyPipeline=destroyPipe;
    api.vkDestroyShaderModule=destroyShader;api.vkCreateComputePipelines=createPipe;
    auto* parent=program(device,0x21,0x11111111u);parent->m_isSpecializable=true;
    auto* a=program(device,0x25,0xaaaaaaaau);auto* b=program(device,0x35,0xbbbbbbbbu);
    IComputePipeline* pa=pipeline(device,a);IComputePipeline* pb=pipeline(device,b);assert(pipelineCreates==2);
    ComputePipelineDesc vd{};vd.program=parent;
    auto* virtualPipe=new VirtualComputePipeline(device,vd);virtualPipe->addRef();
    RefPtr<ExtendedShaderObjectTypeListObject> sa=new ExtendedShaderObjectTypeListObject();sa->componentIDs.push_back(17);
    RefPtr<ExtendedShaderObjectTypeListObject> sb=new ExtendedShaderObjectTypeListObject();sb->componentIDs.push_back(29);
    PipelineKey ka{};ka.pipeline=virtualPipe;ka.specializationArgs.push_back(17);ka.updateHash();
    PipelineKey kb{};kb.pipeline=virtualPipe;kb.specializationArgs.push_back(29);kb.updateHash();
    device->m_shaderCache.addSpecializedPipeline(ka,checked_cast<ComputePipeline*>(pa));
    device->m_shaderCache.addSpecializedPipeline(kb,checked_cast<ComputePipeline*>(pb));
    auto* command=new CommandBufferImpl(device,nullptr);command->addRef();command->m_commandBuffer=(VkCommandBuffer)0x1234;
    BindingDataImpl bindings{};command->m_commandList.write(commands::BeginComputePass{});
    commands::SetComputeState astate{};astate.pipeline=virtualPipe;astate.specializationArgs=sa;astate.bindingData=&bindings;
    command->m_commandList.write(std::move(astate));commands::DispatchCompute da{};da.x=5;da.y=1;da.z=1;command->m_commandList.write(std::move(da));
    commands::SetComputeState bstate{};bstate.pipeline=virtualPipe;bstate.specializationArgs=sb;bstate.bindingData=&bindings;
    command->m_commandList.write(std::move(bstate));commands::DispatchCompute db{};db.x=3;db.y=2;db.z=1;command->m_commandList.write(std::move(db));
    command->m_commandList.write(commands::EndComputePass{});
    CommandEncoderImpl encoder(device,nullptr,CommandEncoderDesc{});encoder.m_commandList=&command->m_commandList;
    assert(SLANG_SUCCEEDED(encoder.resolvePipelines(device)));assert(pipelineCreates==2); // both cache hits
    CommandRecorder recorder(device);assert(SLANG_SUCCEEDED(recorder.record(command)));
    assert(binds==2 && dispatches==2);
    uint32_t count=0;assert(SLANG_SUCCEEDED(command->getDispatchCount(&count)) && count==2);
    VulkanRecordedDispatchV1 first{},second{};
    assert(SLANG_SUCCEEDED(command->getDispatch(0,&first)) && SLANG_SUCCEEDED(command->getDispatch(1,&second)));
    assert(first.pipeline==0x101 && first.shaderModule==0x25 && first.groups[0]==5);
    assert(second.pipeline==0x102 && second.shaderModule==0x35 && second.groups[0]==3 && second.groups[1]==2);
    assert(first.codeSize==8 && second.codeSize==8 && ((const uint32_t*)first.code)[1]==0xaaaaaaaau && ((const uint32_t*)second.code)[1]==0xbbbbbbbbu);
    command->m_commandBuffer=VK_NULL_HANDLE;command->release();device->m_shaderCache.free();
    virtualPipe->release();pa->release();pb->release();parent->release();a->release();b->release();device->release();
}
