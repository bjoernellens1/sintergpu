#include "vulkan/vk-buffer.h"
#include <cassert>
using namespace rhi;
using namespace rhi::vk;
static bool destroyed = false;
class ProbeBuffer : public BufferImpl {
public:
    ProbeBuffer() : BufferImpl(nullptr, BufferDesc{}) {}
    ~ProbeBuffer() override { destroyed = true; }
    void deleteThis() override { delete this; }
};
int main() {
    auto* b = new ProbeBuffer();
    b->addRef();
    IVulkanBufferAllocationObservationV1* q = nullptr;
    assert(SLANG_SUCCEEDED(b->queryInterface(IVulkanBufferAllocationObservationV1::getTypeGuid(), (void**)&q)));
    assert(q);
    VulkanBufferAllocationObservationV1 out{};
    assert(SLANG_FAILED(q->getObservation(&out))); // untracked/native-style
    b->m_desc.size = 16;
    b->m_buffer.m_buffer = (VkBuffer)0x10;
    b->m_buffer.m_memory = (VkDeviceMemory)0x20;
    b->m_buffer.m_allocationSize = 32;
    b->m_buffer.m_observable = true;
    assert(SLANG_SUCCEEDED(q->getObservation(&out)));
    assert(out.buffer == 0x10 && out.memory == 0x20 && out.bufferSize == 16 && out.allocationSize == 32 && out.bindingOffset == 0 && out.ownerGeneration);
    b->m_observationTainted = true;
    assert(SLANG_FAILED(q->getObservation(&out)));
    b->release();
    assert(!destroyed); // COM interface retains the RHI object
    q->release();
    assert(destroyed);
}
