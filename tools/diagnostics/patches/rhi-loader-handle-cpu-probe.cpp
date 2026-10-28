// SPDX-License-Identifier: Apache-2.0
// Linux-only synthetic shared objects: no Vulkan API function is called.
#include "vulkan/vk-api.h"
#include <link.h>
#include <dlfcn.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#ifdef NDEBUG
#error Assertions are required
#endif
using namespace rhi::vk;
class ObservableModule : public VulkanModule {
public:
    void* borrowedHandle() const { return m_module; }
};
static std::string canonical(const char* path) {
    assert(path && *path);
    char result[PATH_MAX]; assert(realpath(path,result));
    struct stat st{};assert(stat(result,&st)==0 && S_ISREG(st.st_mode));return result;
}
int main(int argc, char** argv) {
    assert(argc==4);
    ObservableModule module;assert(SLANG_SUCCEEDED(module.init()));
    link_map* map=nullptr;assert(dlinfo(module.borrowedHandle(),RTLD_DI_LINKMAP,&map)==0 && map);
    const std::string loaded=canonical(map->l_name);assert(loaded==canonical(argv[1]));
    auto symbol=module.getFunction("vkGetInstanceProcAddr");assert(symbol);
    Dl_info info{};assert(dladdr(reinterpret_cast<void*>(symbol),&info) && info.dli_fname && info.dli_fbase);
    const std::string symbolOwner=canonical(info.dli_fname);assert(symbolOwner==canonical(argv[2]));
    assert((loaded==symbolOwner)==(std::strcmp(argv[3],"same")==0));
    // The mapping identifies a dynamic symbol owner only, not underlying ICD,
    // mapped-page contents, dependency closure or provider authentication.
}
