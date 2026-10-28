// SPDX-License-Identifier: Apache-2.0
// Synthetic CPU-only RHI objects; no Vulkan loader, device creation or GPU.
#include "vulkan/vk-base.h"
#define private public
#include "vulkan/vk-device.h"
#undef private
#include "vulkan/vk-buffer.h"
#include <cassert>
#include <slang-com-ptr.h>
#include <stdexcept>
#ifdef NDEBUG
#error Assertions are required
#endif
using namespace rhi;
using namespace rhi::vk;
static bool destroyed=false;
class ProbeBuffer : public BufferImpl {
public:
    ProbeBuffer(DeviceImpl* d) : BufferImpl(d, BufferDesc{}) {}
    ~ProbeBuffer() override { destroyed=true; }
    void deleteThis() override { delete this; }
};
int main() {
    auto* d=new DeviceImpl(); d->addRef();
    auto* b=new ProbeBuffer(d); b->addRef();
    IVulkanBufferDeviceCorrelationV1* q=nullptr;
    assert(SLANG_SUCCEEDED(b->queryInterface(IVulkanBufferDeviceCorrelationV1::getTypeGuid(),(void**)&q)) && q);
    VulkanBufferDeviceCorrelationV1 out{};
    assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));
    b->m_desc.size=16; b->m_buffer.m_buffer=(VkBuffer)0x10; b->m_buffer.m_memory=(VkDeviceMemory)0x20;
    b->m_buffer.m_allocationSize=32; b->m_buffer.m_observable=true;
    assert(SLANG_FAILED(q->getDeviceCorrelation(&out))); // device receipt absent
    d->m_device=(VkDevice)0x55; auto& r=d->m_creationReceipt;
    r.finalized=true;r.ownerGeneration=99;r.deviceUUIDKnown=true;r.deviceUUID[0]=1;r.device={NativeHandleType::VkDevice,0x55};
    assert(SLANG_SUCCEEDED(q->getDeviceCorrelation(&out)));
    assert(out.deviceGeneration==99 && out.device==0x55 && out.allocation.ownerGeneration &&
           out.allocation.buffer==0x10 && out.allocation.memory==0x20 && out.allocation.bufferSize==16);
    assert(SLANG_FAILED(q->getDeviceCorrelation(nullptr)));
    out.schemaVersion=2;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));out.schemaVersion=1;
    out.structSize=0;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));out.structSize=sizeof(out);
    r.deviceUUIDKnown=false;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.deviceUUIDKnown=true;
    r.deviceUUID[0]=0;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.deviceUUID[0]=1;
    r.instanceExternal=true;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.instanceExternal=false;
    r.physicalDeviceExternal=true;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.physicalDeviceExternal=false;
    r.device.type=NativeHandleType::VkInstance;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.device.type=NativeHandleType::VkDevice;
    r.deviceExternal=true;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.deviceExternal=false;
    r.device.value=0x56;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.device.value=0x55;
    r.finalized=false;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));r.finalized=true;
    b->m_observationTainted=true;assert(SLANG_FAILED(q->getDeviceCorrelation(&out)));
    b->m_buffer.m_buffer=VK_NULL_HANDLE;b->m_buffer.m_memory=VK_NULL_HANDLE;
    d->m_device=VK_NULL_HANDLE;
    try {
        Slang::ComPtr<IVulkanBufferDeviceCorrelationV1> held;
        assert(SLANG_SUCCEEDED(b->queryInterface(IVulkanBufferDeviceCorrelationV1::getTypeGuid(),reinterpret_cast<void**>(held.writeRef()))));
        throw std::runtime_error("synthetic conversion failure");
    } catch (const std::runtime_error&) {}
    b->release();assert(!destroyed);q->release();assert(destroyed);d->release();
}
