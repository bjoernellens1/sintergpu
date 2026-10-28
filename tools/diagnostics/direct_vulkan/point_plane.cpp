// SPDX-License-Identifier: Apache-2.0
// Controlled engineering parity probe. Caller owns lock and trusted-workstation
// assumptions. This is not SlangPy, ICD, performance or hardware qualification.
#include <vulkan/vulkan.h>
#include <openssl/sha.h>
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <regex>
#include <algorithm>
#include <limits>

static void check(VkResult r) { if(r!=VK_SUCCESS) throw std::runtime_error("Vulkan failure " + std::to_string(r)); }
static std::vector<unsigned char> read(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f) throw std::runtime_error("input open");
    auto n=f.tellg();if(n<0 || n>1024*1024) throw std::runtime_error("input size");
    std::vector<unsigned char> v((size_t)n);f.seekg(0);if(!f.read((char*)v.data(),n)) throw std::runtime_error("input read");return v;
}
static std::string hash(const std::vector<unsigned char>& bytes) {
    unsigned char d[SHA256_DIGEST_LENGTH];SHA256(bytes.data(),bytes.size(),d);
    const char *hex="0123456789abcdef";std::string s;for(auto c:d){s+=hex[c>>4];s+=hex[c&15];}return s;
}
static std::array<uint8_t,16> uuid(const char* s) {
    if(strlen(s)!=32) throw std::runtime_error("UUID format");
    std::array<uint8_t,16> v{};
    auto digit=[](char x)->int {if(x>='0'&&x<='9')return x-'0';if(x>='a'&&x<='f')return x-'a'+10;throw std::runtime_error("UUID hex");};
    for(int i=0;i<16;i++)v[i]=(digit(s[2*i])<<4)|digit(s[2*i+1]);
    if(std::all_of(v.begin(),v.end(),[](auto x){return x==0;}))throw std::runtime_error("zero UUID");
    return v;
}
static uint32_t parse32(const char* value) {
    if(!*value || *value=='-' || *value=='+')throw std::runtime_error("integer syntax");
    size_t consumed=0;auto n=std::stoull(value,&consumed,0);
    if(consumed!=strlen(value)||n>std::numeric_limits<uint32_t>::max())throw std::runtime_error("integer range");
    return static_cast<uint32_t>(n);
}
#ifndef SINTERGPU_LOCK_PATH
#define SINTERGPU_LOCK_PATH "/tmp/sintergpu-gpu.lock"
#endif
static void verifyLease(uint32_t fd) {
    struct stat a{},b{};if(fstat(fd,&a)||lstat(SINTERGPU_LOCK_PATH,&b)||!S_ISREG(b.st_mode)||a.st_dev!=b.st_dev||a.st_ino!=b.st_ino)throw std::runtime_error("lock inode");
    std::ifstream info("/proc/self/fdinfo/"+std::to_string(fd));if(!info)throw std::runtime_error("lock fdinfo");
    std::string line;bool found=false;std::regex pattern("^lock:[[:space:]]*[0-9]+:[[:space:]]+FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE[[:space:]]+([0-9]+)[[:space:]]+[0-9a-fA-F]+:[0-9a-fA-F]+:([0-9]+)[[:space:]]+0[[:space:]]+EOF[[:space:]]*$");
    while(std::getline(info,line)){std::smatch m;if(std::regex_match(line,m,pattern)&&std::stoull(m[1])==(unsigned)getppid()&&std::stoull(m[2])==a.st_ino)found=true;}
    if(!found)throw std::runtime_error("exclusive parent lock unavailable");
}
struct Resource { VkBuffer buffer=VK_NULL_HANDLE; VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr; VkDeviceSize size=0; };
struct Context {
    VkInstance instance=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;VkCommandPool pool=VK_NULL_HANDLE;
    VkDescriptorSetLayout layout=VK_NULL_HANDLE;VkDescriptorPool descriptors=VK_NULL_HANDLE;VkPipelineLayout pipelineLayout=VK_NULL_HANDLE;
    VkShaderModule shader=VK_NULL_HANDLE;VkPipeline pipeline=VK_NULL_HANDLE;std::array<Resource,7> res{};
    ~Context(){if(device){vkDeviceWaitIdle(device);for(auto& r:res){if(r.mapped)vkUnmapMemory(device,r.memory);if(r.buffer)vkDestroyBuffer(device,r.buffer,nullptr);if(r.memory)vkFreeMemory(device,r.memory,nullptr);}if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(shader)vkDestroyShaderModule(device,shader,nullptr);if(pipelineLayout)vkDestroyPipelineLayout(device,pipelineLayout,nullptr);if(descriptors)vkDestroyDescriptorPool(device,descriptors,nullptr);if(layout)vkDestroyDescriptorSetLayout(device,layout,nullptr);if(pool)vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);}
};
#ifdef SINTERGPU_CPU_TEST_ONLY
extern "C" void sintergpu_test_only_fake_vulkan_marker(void);
#endif
int main(int argc,char** argv) try {
#ifdef SINTERGPU_CPU_TEST_ONLY
    sintergpu_test_only_fake_vulkan_marker();
#endif
#ifndef SINTERGPU_CPU_TEST_ONLY
    throw std::runtime_error("direct Vulkan production path blocked pending independent review");
#endif
    // UUID, raw driver, SPIR-V, SHA-256, entry, groups, input, output.
    if(argc!=10) throw std::runtime_error("usage uuid driver shader sha entry groups input output lockfd");
    verifyLease(parse32(argv[9]));
    auto expected=uuid(argv[1]);uint32_t driver=parse32(argv[2]),groups=parse32(argv[6]);
    if(strcmp(argv[5],"main") || !((groups==2 && std::string(argv[4])=="b34c971681eba7c1c7d9a885a13223fc3d481ab72fa39f2b9bf0ddbda2f00f76") ||
       (groups==5 && std::string(argv[4])=="8e56c1349d3a3fdf0ce9c88ee008f977818abbfae368adb4216d74720691afc6")))throw std::runtime_error("unapproved shader profile");
    auto spv=read(argv[3]);if(spv.empty()||spv.size()%4||hash(spv)!=argv[4])throw std::runtime_error("SPIR-V hash/size");
    std::vector<uint32_t> words(spv.size()/4);memcpy(words.data(),spv.data(),spv.size());if(words[0]!=0x07230203)throw std::runtime_error("SPIR-V magic");
    auto input=read(argv[7]);constexpr std::array<size_t,6> sizes={8192,16512,16512,4156,24796,4100};
    size_t total=0;for(auto n:sizes)total+=n;if(input.size()!=total || hash(input)!="e8d3cf128e6f3c547433f2bc457efc26ce0e24147592dc76bcbe4aee726178f7")throw std::runtime_error("fixture size/hash");
    for(size_t i=total-sizes[5];i<total;i+=4){uint32_t index;memcpy(&index,input.data()+i,4);if(index>=512)throw std::runtime_error("point index range");}
    Context c;VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ic.pApplicationInfo=&app;check(vkCreateInstance(&ic,nullptr,&c.instance));
    uint32_t count=0;check(vkEnumeratePhysicalDevices(c.instance,&count,nullptr));if(!count||count>128)throw std::runtime_error("physical count");
    std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(c.instance,&count,devices.data()));
    VkPhysicalDevice physical=VK_NULL_HANDLE;VkPhysicalDeviceProperties props{};
    for(auto p:devices){VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};p2.pNext=&id;vkGetPhysicalDeviceProperties2(p,&p2);if(!memcmp(id.deviceUUID,expected.data(),16)){if(physical)throw std::runtime_error("duplicate UUID");physical=p;props=p2.properties;}}
    if(!physical||props.driverVersion!=driver)throw std::runtime_error("UUID/driver mismatch");
    if(props.apiVersion<VK_API_VERSION_1_1)throw std::runtime_error("physical Vulkan 1.1 required");
    auto& lim=props.limits;if(lim.maxComputeWorkGroupCount[0]<groups||lim.maxComputeWorkGroupInvocations<(groups==2?1024u:256u)||lim.maxComputeWorkGroupSize[0]<(groups==2?1024u:256u)||lim.maxStorageBufferRange<24796||lim.maxUniformBufferRange<32||lim.maxPerStageDescriptorStorageBuffers<6||lim.maxDescriptorSetStorageBuffers<6||lim.maxPerStageDescriptorUniformBuffers<1||lim.maxDescriptorSetUniformBuffers<1||lim.maxPerStageResources<7||lim.maxBoundDescriptorSets<1||lim.maxMemoryAllocationCount<7)throw std::runtime_error("device limits");
    uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties> qprops(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qprops.data());uint32_t qi=0;while(qi<nq&&(!(qprops[qi].queueFlags&VK_QUEUE_COMPUTE_BIT)||!qprops[qi].queueCount))++qi;if(qi==nq)throw std::runtime_error("compute queue");
    float priority=1;VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=qi;qci.queueCount=1;qci.pQueuePriorities=&priority;
    VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qci;check(vkCreateDevice(physical,&dc,nullptr,&c.device));VkQueue queue;vkGetDeviceQueue(c.device,qi,0,&queue);
    VkPhysicalDeviceMemoryProperties memory{};vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    const uint32_t params[8]={1025,512,1032,1032,1032,1032,0,0};size_t offset=0;
    for(size_t i=0;i<7;i++) {auto& r=c.res[i];r.size=i==6?sizeof(params):sizes[i];VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=r.size;bi.usage=i==6?VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;check(vkCreateBuffer(c.device,&bi,nullptr,&r.buffer));VkMemoryRequirements req;vkGetBufferMemoryRequirements(c.device,r.buffer,&req);uint32_t type=0;while(type<memory.memoryTypeCount&&(!(req.memoryTypeBits&(1u<<type))||(memory.memoryTypes[type].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))!=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))++type;if(type==memory.memoryTypeCount)throw std::runtime_error("host coherent memory unavailable");VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;check(vkAllocateMemory(c.device,&ai,nullptr,&r.memory));check(vkBindBufferMemory(c.device,r.buffer,r.memory,0));check(vkMapMemory(c.device,r.memory,0,r.size,0,&r.mapped));if(i==6)memcpy(r.mapped,params,sizeof(params));else{memcpy(r.mapped,input.data()+offset,r.size);offset+=r.size;}}
    std::array<VkDescriptorSetLayoutBinding,7> binds{};for(uint32_t i=0;i<7;i++){binds[i].binding=i;binds[i].descriptorCount=1;binds[i].descriptorType=i==6?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=7;li.pBindings=binds.data();check(vkCreateDescriptorSetLayout(c.device,&li,nullptr,&c.layout));
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&c.layout;check(vkCreatePipelineLayout(c.device,&pli,nullptr,&c.pipelineLayout));
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=spv.size();sm.pCode=words.data();check(vkCreateShaderModule(c.device,&sm,nullptr,&c.shader));
    VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pi.layout=c.pipelineLayout;pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=c.shader;pi.stage.pName=argv[5];check(vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pi,nullptr,&c.pipeline));
    VkDescriptorPoolSize ps[2]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=ps;check(vkCreateDescriptorPool(c.device,&dp,nullptr,&c.descriptors));VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=c.descriptors;dai.descriptorSetCount=1;dai.pSetLayouts=&c.layout;VkDescriptorSet ds;check(vkAllocateDescriptorSets(c.device,&dai,&ds));
    std::array<VkDescriptorBufferInfo,7> infos{};std::array<VkWriteDescriptorSet,7> writes{};for(uint32_t i=0;i<7;i++){infos[i]={c.res[i].buffer,0,c.res[i].size};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=ds;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=binds[i].descriptorType;writes[i].pBufferInfo=&infos[i];}vkUpdateDescriptorSets(c.device,7,writes.data(),0,nullptr);
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=qi;check(vkCreateCommandPool(c.device,&cp,nullptr,&c.pool));VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=c.pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer cmd;check(vkAllocateCommandBuffers(c.device,&ca,&cmd));VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&begin));
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};before.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipelineLayout,0,1,&ds,0,nullptr);vkCmdDispatch(cmd,groups,1,1);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};after.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,nullptr,0,nullptr);check(vkEndCommandBuffer(cmd));VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;check(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE));check(vkQueueWaitIdle(queue));
    int fd=open(argv[8],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);if(fd<0)throw std::runtime_error("output create");for(size_t i=0;i<6;i++){const char* p=(const char*)c.res[i].mapped;size_t left=c.res[i].size;while(left){ssize_t n=write(fd,p,left);if(n<=0){close(fd);throw std::runtime_error("output write");}p+=n;left-=n;}}if(close(fd))throw std::runtime_error("output close");
    verifyLease(parse32(argv[9]));
    std::cout<<"device="<<props.deviceName<<" driver="<<props.driverVersion<<" queue_family="<<qi<<" shader_sha256="<<hash(spv)<<"\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
