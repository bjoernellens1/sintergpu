// SPDX-License-Identifier: Apache-2.0
// Test-only fake Vulkan: exercise actual persistent pipeline-binary cache hit.
// No Vulkan loader/device or GPU initialized; cache blob and callbacks are synthetic.
#include "vulkan/vk-command.cpp"
#include "vulkan/vk-shader-program.h"
#include "vulkan/vk-shader-object-layout.h"
#include <cassert>
#include "core/blob.h"
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
    assert(n==1 && info->stage.module); assert(info->pNext && ((const VkBaseInStructure*)info->pNext)->sType==VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR); ++pipelineCreates; *out=(VkPipeline)(uintptr_t)(0x100+pipelineCreates); return VK_SUCCESS;
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

// A valid, synthetic cache blob exercises actual deserialization and cache-hit logic.
struct Header { uint32_t magic, version, count; };
struct BinaryHeader { uint32_t keySize; uint8_t key[VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR]; uint32_t dataSize, dataOffset; };
static_assert(sizeof(Header)==12 && sizeof(BinaryHeader)==44);
class Cache : public IPersistentCache, public ComBaseObject {
public:
    SLANG_COM_BASE_IUNKNOWN_ALL
    void* getInterface(const Guid&) { return nullptr; }
    Result SLANG_MCALL writeCache(ISlangBlob*,ISlangBlob*) override { assert(false); return SLANG_FAIL; }
    Result SLANG_MCALL queryCache(ISlangBlob* key,ISlangBlob** out) override {
        assert(key && key->getBufferSize()==20); ++queries;
        struct Data { Header header; BinaryHeader binary; uint8_t payload[4]; } data{};
        data.header={0x12345678,1,1};data.binary.keySize=1;data.binary.key[0]=1;
        data.binary.dataSize=4;data.binary.dataOffset=sizeof(Header)+sizeof(BinaryHeader);
        data.payload[0]=0xa5;auto blob=OwnedBlob::create(&data,sizeof(data));returnComPtr(out,blob);return SLANG_OK;
    }
    unsigned queries=0;
};
static unsigned keys=0, creates=0, destroys=0;
static VKAPI_ATTR VkResult VKAPI_CALL getKey(VkDevice,const VkPipelineCreateInfoKHR*,VkPipelineBinaryKeyKHR* key) {
    ++keys;key->keySize=1;key->key[0]=0x55;return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL createBinary(VkDevice,const VkPipelineBinaryCreateInfoKHR* info,const VkAllocationCallbacks*,VkPipelineBinaryHandlesInfoKHR* out) {
    assert(info->pKeysAndDataInfo && info->pKeysAndDataInfo->binaryCount==1);
    assert(info->pKeysAndDataInfo->pPipelineBinaryData[0].dataSize==4);
    assert(*(const uint8_t*)info->pKeysAndDataInfo->pPipelineBinaryData[0].pData==0xa5);
    assert(out->pipelineBinaryCount==1);out->pPipelineBinaries[0]=(VkPipelineBinaryKHR)0x60;++creates;return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL destroyBinary(VkDevice,VkPipelineBinaryKHR binary,const VkAllocationCallbacks*) { assert(binary==(VkPipelineBinaryKHR)0x60);++destroys; }
int main() {
    auto* device=new DeviceImpl();device->addRef();auto& api=device->m_api;
    api.vkDestroyPipeline=destroyPipe;api.vkDestroyShaderModule=destroyShader;api.vkCreateComputePipelines=createPipe;
    api.vkFreeCommandBuffers=freeCmd;api.vkDestroyCommandPool=destroyPool;
    api.vkGetPipelineKeyKHR=getKey;api.vkCreatePipelineBinariesKHR=createBinary;api.vkDestroyPipelineBinaryKHR=destroyBinary;
    api.m_extendedFeatures.pipelineBinaryFeatures.pipelineBinaries=VK_TRUE;
    auto* cache=new Cache();device->m_persistentPipelineCache=cache;
    auto* p=program(device,0x21,0x11111111u);auto* pipe=pipeline(device,p);
    auto* concrete=checked_cast<ComputePipelineImpl*>(pipe);
    assert(cache->queries==1 && keys==2 && creates==1 && destroys==1 && pipelineCreates==1);
    assert(!concrete->m_dispatchObservationReady);
    auto* command=new CommandBufferImpl(device,nullptr);command->addRef();
    assert(!appendDirectDispatchObservation(command,concrete,1,1,1));command->release();
    pipe->release();p->release();device->m_persistentPipelineCache=nullptr;device->release();
}
