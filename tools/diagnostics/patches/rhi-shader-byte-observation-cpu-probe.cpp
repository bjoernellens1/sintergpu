// SPDX-License-Identifier: Apache-2.0
// Synthetic object state only. Does not create a Vulkan device or shader module.
#include "vulkan/vk-shader-program.h"
#include "vulkan/vk-device.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
using namespace rhi;
using namespace rhi::vk;

int main() {
    auto* device = new DeviceImpl();
    device->addRef();
    auto* program = new ShaderProgramImpl(device, ShaderProgramDesc{});
    program->addRef();
    const auto before = program->getReferenceCount();
    try {
        Slang::ComPtr<IVulkanShaderBytesObservationV1> guard;
        assert(SLANG_SUCCEEDED(program->queryInterface(IVulkanShaderBytesObservationV1::getTypeGuid(), reinterpret_cast<void**>(guard.writeRef()))));
        assert(program->getReferenceCount() == before + 1);
        throw std::runtime_error("synthetic binding conversion failure");
    } catch (const std::runtime_error&) {}
    assert(program->getReferenceCount() == before); // guard releases on exception
    IVulkanShaderBytesObservationV1* query = nullptr;
    assert(SLANG_SUCCEEDED(program->queryInterface(IVulkanShaderBytesObservationV1::getTypeGuid(), (void**)&query)));
    uint32_t count = 0;
    assert(SLANG_FAILED(query->getModuleCount(&count)));
    program->m_modules.emplace_back();
    auto& item = program->m_modules.back();
    item.observedCode = {0x07230203u, 0x12345678u};
    item.entryPointName = "main";
    item.shaderModule = (VkShaderModule)0x1234;
    item.observedCreationSucceeded = true;
    assert(SLANG_FAILED(query->getModuleCount(&count))); // full compilation not complete
    program->m_compiledShaders = true;
    assert(SLANG_SUCCEEDED(query->getModuleCount(&count)) && count == 1);
    const void* bytes = nullptr; size_t length = 0; const char* entry = nullptr; uint64_t handle = 0;
    assert(SLANG_SUCCEEDED(query->getModule(0, &bytes, &length, &entry, &handle)));
    assert(length == 8 && std::memcmp(bytes, item.observedCode.data(), length) == 0);
    assert(std::strcmp(entry, "main") == 0 && handle == 0x1234);
    assert(SLANG_FAILED(query->getModule(1, &bytes, &length, &entry, &handle)));
    item.observedCreationSucceeded = false;
    assert(SLANG_FAILED(query->getModuleCount(&count)));
    item.shaderModule = VK_NULL_HANDLE; // synthetic handle must never reach Vulkan teardown
    program->release();
    assert(SLANG_FAILED(query->getModuleCount(&count))); // query retains live program
    query->release();
    device->release();
}
