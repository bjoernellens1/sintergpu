// SPDX-License-Identifier: Apache-2.0
// CPU compiler-backed specialization cache-miss experiment; fake Vulkan only.
#include "vulkan/vk-command.cpp"
#include "vulkan/vk-shader-program.h"
#include "vulkan/vk-shader-object-layout.h"
#include <cassert>
#ifdef NDEBUG
#error "This diagnostic requires assertions enabled"
#endif
#include <cstring>
using namespace rhi;
using namespace rhi::vk;
struct MockDebug : IDebugCallback {
    unsigned errors=0;
    SLANG_NO_THROW void SLANG_MCALL handleMessage(DebugMessageType type,DebugMessageSource,const char*) override { if(type==DebugMessageType::Error) ++errors; }
};
static unsigned pipelineCreates=0, binds=0, dispatches=0;
static VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer,const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
static bool failEnd=false;
static VKAPI_ATTR VkResult VKAPI_CALL end(VkCommandBuffer) { return failEnd ? VK_ERROR_DEVICE_LOST : VK_SUCCESS; }
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

static unsigned layouts=0, modules=0, setLayouts=0;
static std::vector<std::vector<uint32_t>> moduleBytes;
static VKAPI_ATTR VkResult VKAPI_CALL createSetLayout(VkDevice,const VkDescriptorSetLayoutCreateInfo*,const VkAllocationCallbacks*,VkDescriptorSetLayout* out) { ++setLayouts;*out=(VkDescriptorSetLayout)(uintptr_t)(0x300+setLayouts);return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL destroySetLayout(VkDevice,VkDescriptorSetLayout,const VkAllocationCallbacks*) {}
static VKAPI_ATTR VkResult VKAPI_CALL createLayout(VkDevice,const VkPipelineLayoutCreateInfo*,const VkAllocationCallbacks*,VkPipelineLayout* out) { ++layouts; *out=(VkPipelineLayout)0x91;return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL destroyLayout(VkDevice,VkPipelineLayout,const VkAllocationCallbacks*) {}
static VKAPI_ATTR VkResult VKAPI_CALL createModule(VkDevice,const VkShaderModuleCreateInfo* info,const VkAllocationCallbacks*,VkShaderModule* out) { assert(info->codeSize>4 && info->pCode[0]==0x07230203);moduleBytes.emplace_back(info->pCode, info->pCode + info->codeSize/4);++modules;*out=(VkShaderModule)(uintptr_t)(0x200+modules);return VK_SUCCESS; }
int main() {
    auto* device=new DeviceImpl();device->addRef();MockDebug debug;device->m_debugCallback=&debug;auto& api=device->m_api;
    api.vkBeginCommandBuffer=begin;api.vkEndCommandBuffer=end;api.vkCmdBindPipeline=bind;api.vkCmdDispatch=dispatch;
    api.vkFreeCommandBuffers=freeCmd;api.vkDestroyCommandPool=destroyPool;
    api.vkCreateDescriptorSetLayout=createSetLayout;api.vkDestroyDescriptorSetLayout=destroySetLayout;
    api.vkCreatePipelineLayout=createLayout;api.vkDestroyPipelineLayout=destroyLayout;
    api.vkCreateShaderModule=createModule;api.vkDestroyShaderModule=destroyShader;
    api.vkCreateComputePipelines=createPipe;api.vkDestroyPipeline=destroyPipe;
    assert(SLANG_SUCCEEDED(device->m_slangContext.initialize(SlangDesc{},SLANG_SPIRV,"spirv_1_5")));
    auto session=device->m_slangContext.session;
    const char* source=R"(
        interface IFoo { static uint value(); }
        struct One : IFoo { static uint value() { return 1; } }
        struct Two : IFoo { static uint value() { return 2; } }
        RWStructuredBuffer<uint> output;
        [shader("compute")] [numthreads(1,1,1)]
        void main<T : IFoo>(uint3 tid : SV_DispatchThreadID) { output[tid.x] = T.value(); }
    )";
    ComPtr<slang::IBlob> diag;
    auto* module=session->loadModuleFromSourceString("specialization_probe","specialization_probe.slang",source,diag.writeRef());
    if(diag) fprintf(stderr,"%s\n",(const char*)diag->getBufferPointer());assert(module);
    ComPtr<slang::IEntryPoint> entry; assert(SLANG_SUCCEEDED(module->findEntryPointByName("main",entry.writeRef())));
    slang::IComponentType* parts[]={module,entry};ComPtr<slang::IComponentType> composite;
    assert(SLANG_SUCCEEDED(session->createCompositeComponentType(parts,2,composite.writeRef())));
    ComPtr<slang::IComponentType> linked;diag=nullptr;auto link=composite->link(linked.writeRef(),diag.writeRef());
    if(diag) fprintf(stderr,"%s\n",(const char*)diag->getBufferPointer());assert(SLANG_SUCCEEDED(link));
    fprintf(stderr,"specialization parameters: %lld\n",(long long)linked->getSpecializationParamCount());
    assert(linked->getSpecializationParamCount()==1);
    auto* one=linked->getLayout()->findTypeByName("One");auto* two=linked->getLayout()->findTypeByName("Two");assert(one && two);
    ShaderProgramDesc pd{};pd.slangGlobalScope=linked;pd.linkingStyle=LinkingStyle::SingleProgram;
    IShaderProgram* raw=nullptr;assert(SLANG_SUCCEEDED(device->createShaderProgram(pd,&raw,nullptr)) && raw);
    auto* parent=checked_cast<ShaderProgramImpl*>(raw);assert(parent->m_isSpecializable);
    ComputePipelineDesc vd{};vd.program=parent;auto* virtualPipe=new VirtualComputePipeline(device,vd);virtualPipe->addRef();
    Pipeline* failed=nullptr;assert(SLANG_FAILED(device->getConcretePipeline(virtualPipe,nullptr,failed)) && !failed);
    assert(modules==0 && pipelineCreates==0);
    ExtendedShaderObjectTypeList invalid;invalid.componentIDs.push_back(31);
    failed=nullptr;assert(SLANG_FAILED(device->getConcretePipeline(virtualPipe,&invalid,failed)) && !failed);
    assert(modules==0 && pipelineCreates==0);
    ExtendedShaderObjectTypeList a;a.componentIDs.push_back(17);a.components.push_back({slang::SpecializationArg::Kind::Type,{one}});
    Pipeline* first=nullptr;assert(SLANG_SUCCEEDED(device->getConcretePipeline(virtualPipe,&a,first)) && first);
    assert(modules==1 && pipelineCreates==1 && layouts==1);
    Pipeline* again=nullptr;assert(SLANG_SUCCEEDED(device->getConcretePipeline(virtualPipe,&a,again)) && again==first);
    assert(modules==1 && pipelineCreates==1);
    ExtendedShaderObjectTypeList b;b.componentIDs.push_back(29);b.components.push_back({slang::SpecializationArg::Kind::Type,{two}});
    Pipeline* second=nullptr;assert(SLANG_SUCCEEDED(device->getConcretePipeline(virtualPipe,&b,second)) && second && second!=first);
    fprintf(stderr,"counts modules=%u pipeline=%u layouts=%u setLayouts=%u\n",modules,pipelineCreates,layouts,setLayouts);
    assert(modules==2 && pipelineCreates==2 && layouts==2 && setLayouts==3);
    auto* ca=checked_cast<ComputePipelineImpl*>(first);auto* cb=checked_cast<ComputePipelineImpl*>(second);
    assert(ca->m_dispatchObservationReady && cb->m_dispatchObservationReady);
    assert(ca->m_dispatchObservationCode == moduleBytes[0] && cb->m_dispatchObservationCode == moduleBytes[1]);
    assert(ca->m_dispatchObservationCode != cb->m_dispatchObservationCode);
    RefPtr<ExtendedShaderObjectTypeListObject> sa=new ExtendedShaderObjectTypeListObject();sa->componentIDs.push_back(17);sa->components.push_back(a.components[0]);
    RefPtr<ExtendedShaderObjectTypeListObject> sb=new ExtendedShaderObjectTypeListObject();sb->componentIDs.push_back(29);sb->components.push_back(b.components[0]);
    auto* command=new CommandBufferImpl(device,nullptr);command->addRef();command->m_commandBuffer=(VkCommandBuffer)0x1234;
    BindingDataImpl bindings{};command->m_commandList.write(commands::BeginComputePass{});
    commands::SetComputeState stateA{};stateA.pipeline=virtualPipe;stateA.specializationArgs=sa;stateA.bindingData=&bindings;
    command->m_commandList.write(std::move(stateA));commands::DispatchCompute dispatchA{};dispatchA.x=5;dispatchA.y=1;dispatchA.z=1;command->m_commandList.write(std::move(dispatchA));
    commands::SetComputeState stateB{};stateB.pipeline=virtualPipe;stateB.specializationArgs=sb;stateB.bindingData=&bindings;
    command->m_commandList.write(std::move(stateB));commands::DispatchCompute dispatchB{};dispatchB.x=3;dispatchB.y=2;dispatchB.z=1;command->m_commandList.write(std::move(dispatchB));
    command->m_commandList.write(commands::EndComputePass{});
    CommandEncoderImpl encoder(device,nullptr,CommandEncoderDesc{});encoder.m_commandList=&command->m_commandList;
    assert(SLANG_SUCCEEDED(encoder.resolvePipelines(device)));assert(modules==2 && pipelineCreates==2);
    CommandRecorder recorder(device);assert(SLANG_SUCCEEDED(recorder.record(command)));assert(binds==2 && dispatches==2);
    uint32_t count=0;assert(SLANG_SUCCEEDED(command->getDispatchCount(&count)) && count==2);
    VulkanRecordedDispatchV1 obsA{},obsB{};assert(SLANG_SUCCEEDED(command->getDispatch(0,&obsA)) && SLANG_SUCCEEDED(command->getDispatch(1,&obsB)));
    assert(obsA.pipeline==0x101 && obsB.pipeline==0x102 && obsA.shaderModule==0x201 && obsB.shaderModule==0x202);
    assert(obsA.groups[0]==5 && obsA.groups[1]==1 && obsB.groups[0]==3 && obsB.groups[1]==2);
    assert(obsA.codeSize==moduleBytes[0].size()*4 && obsB.codeSize==moduleBytes[1].size()*4);
    assert(!memcmp(obsA.code,moduleBytes[0].data(),obsA.codeSize) && !memcmp(obsB.code,moduleBytes[1].data(),obsB.codeSize));
    unsigned errorsBefore=debug.errors;failEnd=true;assert(SLANG_FAILED(recorder.record(command)));assert(SLANG_FAILED(command->getDispatchCount(&count)));assert(debug.errors==errorsBefore+1);
    command->m_commandBuffer=VK_NULL_HANDLE;command->release();
    device->m_shaderCache.free();virtualPipe->release();parent->release();device->release();
}
