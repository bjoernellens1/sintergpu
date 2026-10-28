// SPDX-License-Identifier: Apache-2.0
// Owner-original, standalone Vulkan point-plane controlled A/B experiment.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;
static void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation)+": VkResult "+std::to_string(result));
}
static Bytes read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "cannot open "+path.string());
    auto size=input.tellg(); require(size>=0, "cannot size "+path.string());
    Bytes bytes(static_cast<size_t>(size)); input.seekg(0);
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), size);
    require(bool(input), "cannot read "+path.string()); return bytes;
}
static void write_bytes(const fs::path& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary); require(bool(out), "cannot write "+path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(bool(out), "write failed "+path.string());
}
static std::string quote(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c:s) {
        if(c=='"' || c=='\\') out << '\\' << char(c);
        else if(c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    } out << '"'; return out.str();
}
static uint64_t fnv1a(const Bytes& bytes) {
    uint64_t h=14695981039346656037ull; for(auto b:bytes) { h^=b; h*=1099511628211ull; } return h;
}
static uint64_t thread_cpu_ns() {
    timespec time{};
    require(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&time)==0,"cannot read thread CPU clock");
    return uint64_t(time.tv_sec)*1000000000ull+uint64_t(time.tv_nsec);
}
static std::string identity(const Bytes& bytes) { std::ostringstream out; out<<std::hex<<fnv1a(bytes); return out.str(); }
static uint32_t number(const std::string& s) {
    require(!s.empty() && s[0]!='-', "invalid unsigned integer: "+s);
    size_t end=0; auto n=std::stoull(s,&end,0);
    require(end==s.size() && n<=UINT32_MAX,"invalid uint32: "+s); return static_cast<uint32_t>(n);
}
struct Options {
    fs::path baseline,candidate,fixture,output;
    uint32_t count=0,blocks=1000,warmups=10,vendor=0,device=0;
    bool validate_only=false;
    std::string completion_mode="wait";
};
static Options options(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help") {
            std::cout<<"point_plane_ab --baseline FILE --candidate FILE --fixture-dir DIR --count N\n"
                " --vendor 0x1002 --device 0x1586 --output FILE [--blocks 1000] [--warmups 10]\n"
                " [--completion-mode wait|spin] [--validate-only]\n"
                " (validate-only: CPU fixture/SPIR-V validation; no Vulkan instance or GPU lock)\n"
                "Allowed devices: AMD 1002:1586 or NVIDIA 10de:2782. Takes exclusive nonblocking\n"
                "flock /tmp/sintergpu-gpu.lock for all Vulkan work. No shader compilation in timed paths.\n";
            std::exit(0);
        }
        if(k=="--validate-only") {o.validate_only=true;continue;}
        require(i+1<argc,"missing value for "+k); std::string v=argv[++i];
        if(k=="--baseline") o.baseline=v; else if(k=="--candidate") o.candidate=v;
        else if(k=="--fixture-dir") o.fixture=v; else if(k=="--output") o.output=v;
        else if(k=="--count") o.count=number(v); else if(k=="--blocks") o.blocks=number(v);
        else if(k=="--warmups") o.warmups=number(v); else if(k=="--vendor") o.vendor=number(v);
        else if(k=="--device") o.device=number(v); else if(k=="--completion-mode") o.completion_mode=v; else throw std::runtime_error("unknown option "+k);
    }
    require(!o.baseline.empty()&&!o.candidate.empty()&&!o.fixture.empty()&&!o.output.empty()&&o.count&&o.blocks,
        "baseline, candidate, fixture-dir, output, positive count and blocks required");
    require((o.vendor==0x1002 && o.device==0x1586)||(o.vendor==0x10de && o.device==0x2782),
        "explicit allowed vendor/device required: 1002:1586 or 10de:2782");
    require(o.completion_mode=="wait"||o.completion_mode=="spin","completion-mode must be wait or spin");
    require(o.count<=UINT32_MAX/24-14,"count too large"); return o;
}
static const std::array<const char*,7> names={"points","targets","normals","out_0","out_1","points_indices","params"};
// CPU contract guard, not a general SPIR-V validator. Full semantic validation is
// still the compiler/driver's job; reject a different entry/workgroup/binding ABI.
static uint32_t validate_spirv(const Bytes& bytes) {
    require(bytes.size()>=20&&bytes.size()%4==0,"invalid SPIR-V byte count");
    std::vector<uint32_t> w(bytes.size()/4);std::memcpy(w.data(),bytes.data(),bytes.size());
    require(w[0]==0x07230203&&w[1]>=0x10000&&w[1]<=0x10500,"expected little-endian SPIR-V <=1.5");
    std::map<uint32_t,uint32_t> bindings,sets;uint32_t entry=0,local_x=0;
    for(size_t i=5;i<w.size();) {
        uint32_t length=w[i]>>16,opcode=w[i]&65535;
        require(length&&length<=w.size()-i,"malformed SPIR-V instruction length");
        if(opcode==17)require(length==2&&w[i+1]==1,"only Shader capability supported by this point-plane adapter");
        if(opcode==15) {
            require(length>=4,"malformed SPIR-V entry point");
            const char* name=reinterpret_cast<const char*>(w.data()+i+3);
            const void* end=std::memchr(name,0,(length-3)*4);
            require(end,"unterminated SPIR-V entry name");
            if(std::string(name)=="main") {require(w[i+1]==5&&!entry,"expected one compute main entry");entry=w[i+2];}
        }
        if(opcode==71&&length==4) {
            if(w[i+2]==33)require(bindings.emplace(w[i+1],w[i+3]).second,"duplicate SPIR-V binding decoration");
            if(w[i+2]==34)require(sets.emplace(w[i+1],w[i+3]).second,"duplicate SPIR-V descriptor-set decoration");
        }
        i+=length;
    }
    require(entry&&bindings.size()==7&&sets.size()==7,"expected main and seven descriptor bindings");
    std::array<bool,7> seen{};
    for(const auto& pair:bindings) {
        auto set=sets.find(pair.first);require(set!=sets.end()&&set->second==0&&pair.second<7&&!seen[pair.second],"expected unique bindings 0..6 in set 0");seen[pair.second]=true;
    }
    for(size_t i=5;i<w.size();) {
        uint32_t length=w[i]>>16,opcode=w[i]&65535;
        if(opcode==16&&length>=3&&w[i+1]==entry&&w[i+2]==17) {
            require(length==6&&(w[i+3]==256||w[i+3]==1024)&&w[i+4]==1&&w[i+5]==1,"expected numthreads(256 or 1024,1,1)");
            require(!local_x,"duplicate literal LocalSize execution mode");local_x=w[i+3];
        }i+=length;
    }
    require(local_x,"expected literal LocalSize(256 or 1024,1,1)");
    return local_x;
}
struct Fixture {
    std::array<Bytes,7> arrays;
    std::array<Bytes,2> spirv;
    std::array<uint32_t,2> local_x{};
    explicit Fixture(const Options& o) {
        uint64_t alloc=uint64_t(o.count)+7, points=std::max(1u,o.count/2);
        std::array<uint64_t,6> sizes={points*16,alloc*16,alloc*16,(alloc+7)*4,(6*alloc+7)*4,uint64_t(o.count)*4};
        for(size_t i=0;i<7;++i) {
            arrays[i]=read_bytes(o.fixture/(std::string(names[i])+".bin"));
            if(i<6) require(arrays[i].size()==sizes[i],std::string("wrong fixture byte count: ")+names[i]);
        }
        require(arrays[6].size()==24||arrays[6].size()==28||arrays[6].size()==32,"params must contain 6-8 uint32 values");
        std::array<uint32_t,8> p{}; std::memcpy(p.data(),arrays[6].data(),arrays[6].size());
        require(p[0]==o.count && p[1]==points,"params count/point allocation mismatch");
        for(size_t i=2;i<6;++i) require(p[i]==alloc,"params allocation mismatch");
        require(p[6]==0&&p[7]==0,"reserved params must be zero"); arrays[6].resize(32,0);
        for(size_t i=0;i<o.count;++i) {
            uint32_t index; std::memcpy(&index,arrays[5].data()+4*i,4);
            require(index<points,"fixture point index out of bounds");
        }
        spirv[0]=read_bytes(o.baseline); spirv[1]=read_bytes(o.candidate);
        for(size_t i=0;i<spirv.size();++i)local_x[i]=validate_spirv(spirv[i]);
        require(local_x[0]==1024,"baseline must retain numthreads(1024,1,1)");
    }
};
struct RunLock {
    int fd=-1; fs::path ownership;
    explicit RunLock(const Options& o):ownership(o.output.string()+".ownership.json") {
        fd=open("/tmp/sintergpu-gpu.lock",O_CREAT|O_RDWR|O_CLOEXEC,0600);
        require(fd>=0,"cannot open GPU lock");
        if(flock(fd,LOCK_EX|LOCK_NB)!=0) {close(fd);fd=-1;throw std::runtime_error("GPU lock is held; no GPU work started");}
        record("active");
    }
    void record(const char* status) const {
        char host[256]{}; gethostname(host,sizeof(host)-1);
        std::ofstream out(ownership);
        require(bool(out),"cannot record GPU lock ownership");
        out<<"{\"lock\":\"/tmp/sintergpu-gpu.lock\",\"pid\":"<<getpid()<<",\"hostname\":"<<quote(host)
            <<",\"status\":"<<quote(status)<<",\"scope\":\"all Vulkan instance/device work\"}\n";
    }
    ~RunLock() {if(fd>=0) {try {record("released");} catch(...) {} flock(fd,LOCK_UN);close(fd);}}
};
struct Buffer {VkBuffer handle=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;VkDeviceSize size=0,allocated=0;uint32_t type=0;VkMemoryPropertyFlags flags=0;void* mapped=nullptr;};
struct Sample {const char* boundary;uint32_t block,position,variant;double wall_ns=0,gpu_ns=0,thread_cpu_ns=0;};
struct Harness {
    const Options& o; const Fixture& fixture;
    VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};VkPhysicalDeviceMemoryProperties memory{};uint32_t family=0,timestamp_bits=0;
    VkQueue queue=VK_NULL_HANDLE;std::array<Buffer,7> shader{},upload{};std::array<Buffer,2> readback{};
    VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;
    VkDescriptorSet set=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;std::array<VkPipeline,2> pipelines{};
    VkCommandPool command_pool=VK_NULL_HANDLE;VkCommandBuffer cmd=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;
    VkQueryPool queries=VK_NULL_HANDLE;std::array<Bytes,2> host_outputs;
    explicit Harness(const Options& op,const Fixture& f):o(op),fixture(f) {}
    ~Harness() {
        if(device) {
            vkDeviceWaitIdle(device);
            if(queries)vkDestroyQueryPool(device,queries,nullptr);
            if(fence)vkDestroyFence(device,fence,nullptr);
            if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);
            for(auto pipeline:pipelines)if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
            if(layout)vkDestroyPipelineLayout(device,layout,nullptr);
            if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
            if(descriptors)vkDestroyDescriptorSetLayout(device,descriptors,nullptr);
            for(auto& b:shader)destroy(b);
            for(auto& b:upload)destroy(b);
            for(auto& b:readback)destroy(b);
            vkDestroyDevice(device,nullptr);
        }
        if(instance)vkDestroyInstance(instance,nullptr);
    }
    void destroy(Buffer& b) {if(b.mapped)vkUnmapMemory(device,b.memory);if(b.handle)vkDestroyBuffer(device,b.handle,nullptr);if(b.memory)vkFreeMemory(device,b.memory,nullptr);}
    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="sintergpu point-plane native A/B";app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
        check(vkCreateInstance(&ici,nullptr,&instance),"vkCreateInstance");
        uint32_t n=0;check(vkEnumeratePhysicalDevices(instance,&n,nullptr),"enumerate physical device count");
        std::vector<VkPhysicalDevice> devices(n);check(vkEnumeratePhysicalDevices(instance,&n,devices.data()),"enumerate physical devices");
        for(auto d:devices) {
            VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(d,&p);
            if(p.vendorID==o.vendor&&p.deviceID==o.device) {
                require(!physical,"multiple matching adapters; select an unambiguous ICD/device environment");
                require(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU||p.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,"software/non-GPU adapter rejected");physical=d;props=p;
            }
        }
        require(physical,"requested physical GPU not found");
        for(auto local_x:fixture.local_x) {
            require(props.limits.maxComputeWorkGroupInvocations>=local_x&&props.limits.maxComputeWorkGroupSize[0]>=local_x,"shader local size unsupported");
            require((uint64_t(o.count)+local_x-1)/local_x<=props.limits.maxComputeWorkGroupCount[0],"dispatch count exceeds device limits");
        }
        require(props.limits.timestampPeriod>0,"invalid timestamp period");
        uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&qn,nullptr);
        std::vector<VkQueueFamilyProperties> q(qn);vkGetPhysicalDeviceQueueFamilyProperties(physical,&qn,q.data());
        bool found=false;for(uint32_t i=0;i<qn;++i)if(q[i].queueCount&&(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&q[i].timestampValidBits){family=i;timestamp_bits=q[i].timestampValidBits;found=true;break;}
        require(found,"no compute queue with timestamps");
        // Slang emits SPIR-V 1.5; Vulkan 1.2 is the minimum profile for this adapter.
        require(props.apiVersion>=VK_API_VERSION_1_2,"Vulkan 1.2 required");
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkPhysicalDeviceFeatures enabled{}; // Point-plane requires only core Shader capability.
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.pEnabledFeatures=&enabled;
        check(vkCreateDevice(physical,&di,nullptr,&device),"vkCreateDevice");vkGetDeviceQueue(device,family,0,&queue);
        vkGetPhysicalDeviceMemoryProperties(physical,&memory);
        for(size_t i=0;i<7;++i) {
            require(i==6||fixture.arrays[i].size()<=props.limits.maxStorageBufferRange,"buffer exceeds storage descriptor range");
            shader[i]=make_buffer(fixture.arrays[i].size(),VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|(i==6?VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,0);
            upload[i]=make_buffer(fixture.arrays[i].size(),VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        for(size_t i=0;i<2;++i) {
            readback[i]=make_buffer(fixture.arrays[i+3].size(),VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            host_outputs[i].resize(fixture.arrays[i+3].size());
        }
        std::array<VkDescriptorSetLayoutBinding,7> bindings{};
        for(uint32_t i=0;i<7;++i)bindings[i]={i,i==6?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dli.bindingCount=7;dli.pBindings=bindings.data();
        check(vkCreateDescriptorSetLayout(device,&dli,nullptr,&descriptors),"create descriptor layout");
        std::array<VkDescriptorPoolSize,2> ps={{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=2;dpi.pPoolSizes=ps.data();
        check(vkCreateDescriptorPool(device,&dpi,nullptr,&descriptor_pool),"create descriptor pool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=descriptor_pool;dai.descriptorSetCount=1;dai.pSetLayouts=&descriptors;
        check(vkAllocateDescriptorSets(device,&dai,&set),"allocate descriptor set");
        std::array<VkDescriptorBufferInfo,7> bi{};std::array<VkWriteDescriptorSet,7> writes{};
        for(uint32_t i=0;i<7;++i){bi[i]={shader[i].handle,0,shader[i].size};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,i,0,1,bindings[i].descriptorType,nullptr,&bi[i],nullptr};}
        vkUpdateDescriptorSets(device,7,writes.data(),0,nullptr);
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&descriptors;
        check(vkCreatePipelineLayout(device,&pli,nullptr,&layout),"create pipeline layout");
        for(size_t i=0;i<2;++i) {
            std::vector<uint32_t> code(fixture.spirv[i].size()/4);std::memcpy(code.data(),fixture.spirv[i].data(),fixture.spirv[i].size());
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smi.codeSize=fixture.spirv[i].size();smi.pCode=code.data();VkShaderModule module=VK_NULL_HANDLE;
            check(vkCreateShaderModule(device,&smi,nullptr,&module),"create shader module");
            VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pi.layout=layout;pi.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
            VkResult result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pi,nullptr,&pipelines[i]);vkDestroyShaderModule(device,module,nullptr);check(result,"create compute pipeline");
        }
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&cpi,nullptr,&command_pool),"create command pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=command_pool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&cai,&cmd),"allocate command buffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fi,nullptr,&fence),"create fence");
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qpi.queryType=VK_QUERY_TYPE_TIMESTAMP;qpi.queryCount=2;
        check(vkCreateQueryPool(device,&qpi,nullptr,&queries),"create timestamp queries");
    }
    Buffer make_buffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags required,VkMemoryPropertyFlags preferred) {
        Buffer b;b.size=size;VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=size;ci.usage=usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device,&ci,nullptr,&b.handle),"create buffer");
        try {
            VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,b.handle,&req);b.allocated=req.size;
            bool found=false;for(uint32_t pass=0;pass<2&&!found;++pass)for(uint32_t i=0;i<memory.memoryTypeCount;++i) {
                auto flags=memory.memoryTypes[i].propertyFlags;
                if((req.memoryTypeBits&(1u<<i))&&(flags&required)==required&&(pass||(flags&preferred)==preferred)) {b.type=i;b.flags=flags;found=true;break;}
            }
            require(found,"required memory unavailable (host staging explicitly requires HOST_CACHED)");
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=b.type;
            check(vkAllocateMemory(device,&ai,nullptr,&b.memory),"allocate buffer memory");check(vkBindBufferMemory(device,b.handle,b.memory,0),"bind buffer memory");
            if(required&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)check(vkMapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"map staging buffer");
        } catch(...) {destroy(b);throw;}return b;
    }
    void sync_host(const Buffer& b,bool flush) {
        if(b.flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)return;
        // Offset 0 and VK_WHOLE_SIZE cover the full allocation, satisfying atom alignment even at its tail.
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=b.memory;r.offset=0;r.size=VK_WHOLE_SIZE;
        check(flush?vkFlushMappedMemoryRanges(device,1,&r):vkInvalidateMappedMemoryRanges(device,1,&r),flush?"flush staging":"invalidate staging");
    }
    void barrier(VkPipelineStageFlags from,VkPipelineStageFlags to,VkAccessFlags src,VkAccessFlags dst) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=src;b.dstAccessMask=dst;
        vkCmdPipelineBarrier(cmd,from,to,0,1,&b,0,nullptr,0,nullptr);
    }
    Sample invoke(uint32_t variant,const char* boundary,uint32_t block=0,uint32_t position=0) {
        bool transfer=std::strcmp(boundary,"transfer_inclusive")==0,timestamp=std::strcmp(boundary,"gpu_timestamp")==0;
        auto cpu_start=thread_cpu_ns();
        auto start=Clock::now();
        if(transfer)for(size_t i=0;i<7;++i) {std::memcpy(upload[i].mapped,fixture.arrays[i].data(),fixture.arrays[i].size());sync_host(upload[i],true);}
        check(vkResetFences(device,1,&fence),"reset fence");check(vkResetCommandBuffer(cmd,0),"reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd,&begin),"begin command buffer");
        if(timestamp)vkCmdResetQueryPool(cmd,queries,0,2);
        if(transfer) {
            barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
            // Queue submission makes preceding host writes available to transfer reads.
            for(size_t i=0;i<7;++i) {VkBufferCopy copy{0,0,shader[i].size};vkCmdCopyBuffer(cmd,upload[i].handle,shader[i].handle,1,&copy);}
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        } else barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[variant]);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        if(timestamp)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,0);
        uint32_t local_x=fixture.local_x[variant];
        vkCmdDispatch(cmd,(o.count+local_x-1)/local_x,1,1);
        if(timestamp)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,1);
        if(transfer) {
            barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
            for(size_t i=0;i<2;++i) {VkBufferCopy copy{0,0,readback[i].size};vkCmdCopyBuffer(cmd,shader[i+3].handle,readback[i].handle,1,&copy);}
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        }
        check(vkEndCommandBuffer(cmd),"end command buffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;
        check(vkQueueSubmit(queue,1,&submit,fence),"queue submit");
        if(o.completion_mode=="wait")check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"fence completion");
        else {
            VkResult status;
            do {status=vkGetFenceStatus(device,fence);} while(status==VK_NOT_READY);
            check(status,"spin fence completion");
        }
        if(transfer)for(size_t i=0;i<2;++i) {sync_host(readback[i],false);std::memcpy(host_outputs[i].data(),readback[i].mapped,host_outputs[i].size());}
        auto stop=Clock::now();
        auto cpu_stop=thread_cpu_ns();
        Sample s{boundary,block,position,variant,std::chrono::duration<double,std::nano>(stop-start).count(),0,double(cpu_stop-cpu_start)};
        if(timestamp) {
            std::array<uint64_t,2> ticks{};check(vkGetQueryPoolResults(device,queries,0,2,sizeof(ticks),ticks.data(),sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"timestamp results");
            uint64_t mask=timestamp_bits==64?UINT64_MAX:((uint64_t(1)<<timestamp_bits)-1);
            s.gpu_ns=double((ticks[1]-ticks[0])&mask)*props.limits.timestampPeriod;
        }return s;
    }
    void correctness(const char* phase) {
        invoke(0,"transfer_inclusive"); auto baseline=host_outputs;
        for(size_t i=0;i<2;++i)write_bytes(o.output.string()+"."+phase+".baseline."+names[i+3]+".bin",baseline[i]);
        invoke(1,"transfer_inclusive");
        for(size_t i=0;i<2;++i)write_bytes(o.output.string()+"."+phase+".candidate."+names[i+3]+".bin",host_outputs[i]);
        for(size_t i=0;i<2;++i)require(baseline[i]==host_outputs[i],std::string(phase)+" baseline/candidate full-byte differential mismatch in "+names[i+3]);
        // Independent poison check; the Python float64 oracle remains a separate required gate.
        for(size_t out=0;out<2;++out)for(size_t i=0;i<host_outputs[out].size()/4;++i) {
            bool live=out==0?i<o.count:(i<size_t(o.count)*4||(i>=size_t(o.count+7)*4&&i<size_t(o.count+7)*4+size_t(o.count)*2));
            if(!live)require(std::memcmp(host_outputs[out].data()+i*4,fixture.arrays[out+3].data()+i*4,4)==0,std::string(phase)+" poisoned padding changed");
        }
    }
    void report(const std::vector<Sample>& samples) const {
        std::ofstream out(o.output);require(bool(out),"cannot write report");out<<std::setprecision(17);
        out<<"{\"status\":\"passed\",\"schema\":\"sintergpu.native.point_plane.ab.v1\",\"profile\":\"native_caller_owned_vulkan\",\"count\":"<<o.count
           <<",\"completion_mode\":"<<quote(o.completion_mode)
           <<",\"launch\":{\"baseline_local_size\":["<<fixture.local_x[0]<<",1,1],\"candidate_local_size\":["<<fixture.local_x[1]<<",1,1],\"baseline_groups_x\":"<<(o.count+fixture.local_x[0]-1)/fixture.local_x[0]<<",\"candidate_groups_x\":"<<(o.count+fixture.local_x[1]-1)/fixture.local_x[1]<<"}"
           <<",\"blocks\":"<<o.blocks<<",\"warmups_per_boundary_per_variant\":"<<o.warmups
           <<",\"correctness\":{\"full_byte_differential_before_after\":true,\"padding_before_after\":true,\"independent_oracle\":\"required externally\"}"
           <<",\"hardware\":{\"name\":"<<quote(props.deviceName)<<",\"vendor_id\":"<<props.vendorID<<",\"device_id\":"<<props.deviceID
           <<",\"device_type\":"<<props.deviceType<<",\"driver_version_raw\":"<<props.driverVersion<<",\"api_version_raw\":"<<props.apiVersion
           <<",\"queue_family\":"<<family<<",\"timestamp_valid_bits\":"<<timestamp_bits<<",\"timestamp_period_ns\":"<<props.limits.timestampPeriod
           <<",\"noncoherent_atom_size\":"<<props.limits.nonCoherentAtomSize<<",\"pipeline_cache_uuid\":\"";
        for(auto byte:props.pipelineCacheUUID)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);
        out<<std::dec<<"\"},\"host\":{\"pid\":"<<getpid()<<",\"cpu_affinity\":[";
        cpu_set_t cpus;CPU_ZERO(&cpus);require(sched_getaffinity(0,sizeof(cpus),&cpus)==0,"cannot read CPU affinity");bool comma=false;
        for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&cpus)){if(comma)out<<',';out<<i;comma=true;}
        out<<"],\"clock\":\"std::chrono::steady_clock\",\"lock\":\"/tmp/sintergpu-gpu.lock\"},\"identity\":{\"algorithm\":\"fnv1a64 (non-cryptographic)\",\"baseline_spirv\":"<<quote(identity(fixture.spirv[0]))<<",\"candidate_spirv\":"<<quote(identity(fixture.spirv[1]));
        fs::path source=__FILE__;if(fs::exists(source))out<<",\"native_source_at_report_time\":"<<quote(identity(read_bytes(source)));
        out<<",\"native_executable\":"<<quote(identity(read_bytes("/proc/self/exe")))<<",\"compiler\":"<<quote(__VERSION__);
        out<<",\"native_file\":"<<quote(source.string())<<",\"baseline_path\":"<<quote(o.baseline.string())<<",\"candidate_path\":"<<quote(o.candidate.string())<<"},\"buffers\":[";
        uint64_t upload_bytes=0,readback_bytes=0;
        for(size_t i=0;i<7;++i) {
            if(i)out<<',';
            upload_bytes+=shader[i].size;
            out<<"{\"name\":"<<quote(names[i])<<",\"bytes\":"<<shader[i].size<<",\"initial_fnv1a64\":"<<quote(identity(fixture.arrays[i]))
               <<",\"device_memory_type\":"<<shader[i].type<<",\"device_memory_flags\":"<<shader[i].flags<<",\"device_allocation_bytes\":"<<shader[i].allocated
               <<",\"upload_memory_type\":"<<upload[i].type<<",\"upload_memory_flags\":"<<upload[i].flags<<",\"upload_allocation_bytes\":"<<upload[i].allocated;
            if(i==3||i==4){auto& b=readback[i-3];readback_bytes+=b.size;out<<",\"readback_memory_type\":"<<b.type<<",\"readback_memory_flags\":"<<b.flags<<",\"readback_allocation_bytes\":"<<b.allocated;}out<<'}';
        }
        out<<"],\"upload_bytes_including_32byte_params\":"<<upload_bytes<<",\"readback_bytes\":"<<readback_bytes
           <<",\"semantics\":{\"gpu_timestamp\":\"timestamps at COMPUTE_SHADER immediately surrounding one dispatch; result fetch after wall timer\",\"resident_completion\":\"one record+submit+fence completion; shared retained buffers and descriptors; no uploads/readback/query commands\",\"transfer_inclusive\":\"CPU memcpy all 7 initial buffers to retained HOST_CACHED staging + flush + record/copies/barriers/dispatch/full 2-output copies + submit/fence + invalidate + CPU memcpy full outputs\",\"order\":\"ABBA independently for each boundary; baseline,candidate,candidate,baseline; every invocation retained; no mean batching\",\"allocations\":\"all Vulkan and host output/sample allocation outside timing\",\"thread_cpu_ns\":\"CLOCK_THREAD_CPUTIME_ID enclosing wall interval; includes thread CPU clock and wall clock overhead, excludes timestamp result fetch; same completion policy for both variants\"},\"samples\":[";
        for(size_t i=0;i<samples.size();++i){const auto& s=samples[i];if(i)out<<',';out<<"{\"boundary\":"<<quote(s.boundary)<<",\"block\":"<<s.block<<",\"position\":"<<s.position<<",\"variant\":"<<quote(s.variant?"candidate":"baseline")<<",\"wall_ns\":"<<s.wall_ns<<",\"thread_cpu_ns\":"<<s.thread_cpu_ns<<",\"gpu_ns\":";if(std::strcmp(s.boundary,"gpu_timestamp")==0)out<<s.gpu_ns;else out<<"null";out<<'}';}
        out<<"]}\n";require(bool(out),"report write failed");
    }
};
int main(int argc,char** argv) {
    try {
        auto o=options(argc,argv);Fixture fixture(o);
        if(o.validate_only){std::cout<<"CPU fixture/SPIR-V ABI validation passed; baseline local size "<<fixture.local_x[0]<<", candidate local size "<<fixture.local_x[1]<<", completion mode "<<o.completion_mode<<"; no Vulkan calls made\n";return 0;}
        if(!o.output.parent_path().empty())fs::create_directories(o.output.parent_path());
        RunLock lock(o);Harness h(o,fixture);h.init();h.correctness("before");
        std::vector<Sample> samples;samples.reserve(size_t(o.blocks)*12);
        constexpr std::array<uint32_t,4> order={0,1,1,0};
        for(const char* boundary:{"gpu_timestamp","resident_completion","transfer_inclusive"}) {
            for(uint32_t i=0;i<o.warmups;++i)for(uint32_t v=0;v<2;++v)h.invoke(v,boundary);
            for(uint32_t block=0;block<o.blocks;++block)for(uint32_t pos=0;pos<4;++pos)samples.push_back(h.invoke(order[pos],boundary,block,pos));
        }
        h.correctness("after");h.report(samples);std::cout<<"Wrote "<<o.output<<" with "<<samples.size()<<" individual ABBA invocations\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"point_plane_ab: "<<e.what()<<'\n';return 1;}
}
