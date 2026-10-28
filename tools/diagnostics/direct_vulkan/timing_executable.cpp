// SPDX-License-Identifier: Apache-2.0
// Unreviewed integration under construction; production execution blocked.
#define main sinter_unused_parity_main
#include "point_plane.cpp"
#undef main
#include "timing_resources.h"
#include "timing_pipelines.h"
#include "timing_readback.h"
#include "timing_oracle.h"
#include "timing_tick_bound.h"
#include "timing_drain.h"
#include "timing_journal.h"
#include "vulkan_point_plane_timing.h"
#include <sstream>
#include <iomanip>
int main(int argc,char** argv) try {
#ifdef SINTERGPU_CPU_TEST_ONLY
 sintergpu_test_only_fake_vulkan_marker();
#endif
#if !defined(SINTERGPU_CPU_TEST_ONLY) && !defined(SINTERGPU_TRUSTED_HOST_TIMING)
 throw std::runtime_error("timing executable blocked pending independent review");
#endif
 if(argc!=9)throw std::runtime_error("usage: UUID driver donor.spv candidate.spv fixture output lockfd samples");
 verifyLease(parse32(argv[7]));if(parse32(argv[8])!=50)throw std::runtime_error("exactly 50 samples required");
 auto expected=uuid(argv[1]);uint32_t driver=parse32(argv[2]);
 if(std::string(argv[1])!="96d2dba095fd5c00e8492e0610f24e06" || driver!=2497233024u)throw std::runtime_error("target mismatch");
 auto old=read(argv[3]),candidate=read(argv[4]),input=read(argv[5]);
#ifndef SINTERGPU_CPU_TEST_ONLY
 if(hash(old)!="b34c971681eba7c1c7d9a885a13223fc3d481ab72fa39f2b9bf0ddbda2f00f76"||hash(candidate)!="8e56c1349d3a3fdf0ce9c88ee008f977818abbfae368adb4216d74720691afc6"||hash(input)!="e8d3cf128e6f3c547433f2bc457efc26ce0e24147592dc76bcbe4aee726178f7")throw std::runtime_error("fixed input identity");
#else
 if(hash(input)!="e8d3cf128e6f3c547433f2bc457efc26ce0e24147592dc76bcbe4aee726178f7")throw std::runtime_error("fixed input identity");
#endif
 std::array<std::vector<uint32_t>,2> words{};for(size_t i=0;i<2;++i){const auto& x=i?candidate:old;if(x.empty()||x.size()%4)throw std::runtime_error("shader size");words[i].resize(x.size()/4);memcpy(words[i].data(),x.data(),x.size());}
 Context c;VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.apiVersion=VK_API_VERSION_1_1;VkInstanceCreateInfo ic{};ic.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ic.pApplicationInfo=&app;check(vkCreateInstance(&ic,nullptr,&c.instance));
 uint32_t n=0;check(vkEnumeratePhysicalDevices(c.instance,&n,nullptr));if(!n||n>128)throw std::runtime_error("physical count");std::vector<VkPhysicalDevice> devices(n);check(vkEnumeratePhysicalDevices(c.instance,&n,devices.data()));VkPhysicalDevice physical=VK_NULL_HANDLE;VkPhysicalDeviceProperties props{};
 for(auto p:devices){VkPhysicalDeviceIDProperties id{};id.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;VkPhysicalDeviceProperties2 p2{};p2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;p2.pNext=&id;vkGetPhysicalDeviceProperties2(p,&p2);if(!memcmp(id.deviceUUID,expected.data(),16)){if(physical)throw std::runtime_error("duplicate UUID");physical=p;props=p2.properties;}}
 if(!physical||props.driverVersion!=driver||props.apiVersion<VK_API_VERSION_1_1||props.vendorID!=0x10de||props.deviceID!=0x2782)throw std::runtime_error("device identity");
 auto plan=sinterFixedPointPlaneAllocationPlan(props.limits);if(props.limits.maxComputeWorkGroupInvocations<1024||props.limits.maxComputeWorkGroupSize[0]<1024||props.limits.maxComputeWorkGroupCount[0]<5)throw std::runtime_error("compute limits");
 uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);if(!nq||nq>128)throw std::runtime_error("queue count");std::vector<VkQueueFamilyProperties> qp(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qp.data());uint32_t qi=0;while(qi<nq&&(!(qp[qi].queueFlags&VK_QUEUE_COMPUTE_BIT)||!qp[qi].queueCount||!qp[qi].timestampValidBits))++qi;if(qi==nq)throw std::runtime_error("timestamp queue");
 float priority=1;VkDeviceQueueCreateInfo qci{};qci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;qci.queueFamilyIndex=qi;qci.queueCount=1;qci.pQueuePriorities=&priority;VkDeviceCreateInfo dc{};dc.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qci;check(vkCreateDevice(physical,&dc,nullptr,&c.device));VkQueue queue=VK_NULL_HANDLE;vkGetDeviceQueue(c.device,qi,0,&queue);if(!queue)throw std::runtime_error("queue unavailable");VkPhysicalDeviceMemoryProperties memory{};vkGetPhysicalDeviceMemoryProperties(physical,&memory);
 SinterTimingJournal journal(argv[6]);std::ostringstream header;header<<"# unqualified engineering diagnostics; ticks and host nanoseconds\n";header<<"# timestamp_valid_bits="<<qp[qi].timestampValidBits<<" period_ns="<<std::setprecision(std::numeric_limits<float>::max_digits10)<<props.limits.timestampPeriod<<" queue_family="<<qi<<"\n";journal.append(header.str());SinterTimingJournal raw((std::string(argv[6])+".raw").c_str());
 {
 SinterResourceAPI resourceApi{vkCreateBuffer,vkDestroyBuffer,vkGetBufferMemoryRequirements,vkAllocateMemory,vkFreeMemory,vkBindBufferMemory,vkMapMemory,vkUnmapMemory};SinterTimingResources res(c.device,memory,plan,resourceApi);
 std::array<VkDescriptorSetLayoutBinding,7> bindings{};for(uint32_t i=0;i<7;++i){bindings[i].binding=i;bindings[i].descriptorCount=1;bindings[i].descriptorType=i==6?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
 VkDescriptorSetLayoutCreateInfo li{};li.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;li.bindingCount=7;li.pBindings=bindings.data();check(vkCreateDescriptorSetLayout(c.device,&li,nullptr,&c.layout));VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&c.layout;check(vkCreatePipelineLayout(c.device,&pli,nullptr,&c.pipelineLayout));
 SinterPipelineAPI pipelineApi{vkCreateShaderModule,vkDestroyShaderModule,vkCreateComputePipelines,vkDestroyPipeline};SinterTimingPipelines pipelines(c.device,c.pipelineLayout,words,pipelineApi);

 VkDescriptorPoolSize ps[2]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};VkDescriptorPoolCreateInfo dp{};dp.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=ps;check(vkCreateDescriptorPool(c.device,&dp,nullptr,&c.descriptors));VkDescriptorSetAllocateInfo dai{};dai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;dai.descriptorPool=c.descriptors;dai.descriptorSetCount=1;dai.pSetLayouts=&c.layout;VkDescriptorSet ds=VK_NULL_HANDLE;check(vkAllocateDescriptorSets(c.device,&dai,&ds));
 std::array<VkDescriptorBufferInfo,7> info{};std::array<VkWriteDescriptorSet,7> writes{};for(uint32_t i=0;i<7;++i){info[i]={res.at(i,1).buffer,0,res.at(i,1).size};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=ds;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=bindings[i].descriptorType;writes[i].pBufferInfo=&info[i];}vkUpdateDescriptorSets(c.device,7,writes.data(),0,nullptr);
 VkCommandPoolCreateInfo cp{};cp.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;cp.queueFamilyIndex=qi;check(vkCreateCommandPool(c.device,&cp,nullptr,&c.pool));VkCommandBufferAllocateInfo ca{};ca.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ca.commandPool=c.pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer cmd=VK_NULL_HANDLE;check(vkAllocateCommandBuffers(c.device,&ca,&cmd));
 SinterTimestampPair timestamps(c.device,qp[qi].timestampValidBits,props.limits.timestampPeriod,{vkCreateQueryPool,vkDestroyQueryPool,vkCmdResetQueryPool,vkCmdWriteTimestamp,vkGetQueryPoolResults});SinterTimingDispatch api{vkResetCommandBuffer,vkBeginCommandBuffer,vkEndCommandBuffer,vkQueueSubmit,vkQueueWaitIdle,vkCmdCopyBuffer,{vkCmdBindPipeline,vkCmdBindDescriptorSets,vkCmdDispatch,vkCmdPipelineBarrier}};SinterPointPlaneTiming timer(queue,cmd,api,timestamps);SinterTimingDrain drain{c.device,vkDeviceWaitIdle};SinterReadbackAPI readApi{vkResetCommandBuffer,vkBeginCommandBuffer,vkEndCommandBuffer,vkQueueSubmit,vkQueueWaitIdle,vkCmdCopyBuffer,vkCmdPipelineBarrier};
 constexpr std::array<size_t,6> sizes={8192,16512,16512,4156,24796,4100};uint32_t params[8]={1025,512,1032,1032,1032,1032,0,0};std::array<SinterTimingStaging,7> staging{};std::array<std::vector<uint8_t>,7> outputs{};size_t offset=0;for(size_t i=0;i<7;++i){auto size=res.at(i,1).size;outputs[i].resize(size);const void* source=i==6?(const void*)params:(const void*)(input.data()+offset);if(i<6)offset+=sizes[i];staging[i]={{res.at(i,0).buffer,res.at(i,1).buffer,res.at(i,2).buffer,size},res.at(i,0).mapped,res.at(i,2).mapped,source,outputs[i].data()};}
 auto reset=[&](){for(auto& s:staging)memcpy(s.mappedUpload,s.initialBytes,s.gpu.size);sinterFixtureTransfer(cmd,queue,res,readApi,true);};
 auto fetch=[&](){sinterFixtureTransfer(cmd,queue,res,readApi,false);std::vector<uint8_t> out;for(size_t i=0;i<6;++i){auto* p=(const uint8_t*)res.at(i,2).mapped;out.insert(out.end(),p,p+sizes[i]);}if(memcmp(res.at(6,2).mapped,params,sizeof(params)))throw std::runtime_error("parameter changed");sinterValidatePointPlaneFixture(input,out);return out;};
 uint32_t samples=parse32(argv[8]);for(uint32_t run=0;run<1;++run)for(uint32_t sample=0;sample<samples+5;++sample){std::vector<uint8_t> baseline;for(uint32_t slot=0;slot<3;++slot){uint32_t v=slot==1?1:0;SinterPointPlaneCommand command{pipelines.at(v),c.pipelineLayout,ds,v?5u:2u};reset();auto t=timer.resident(command);sinterTickInterval(t.rawTicks[0],t.rawTicks[3],qp[qi].timestampValidBits,props.limits.timestampPeriod,t.hostCompletionUpperBoundNs);sinterTickInterval(t.rawTicks[1],t.rawTicks[2],qp[qi].timestampValidBits,props.limits.timestampPeriod,t.hostCompletionUpperBoundNs);auto resident=fetch();std::array<uint8_t,32> residentParams{};memcpy(residentParams.data(),res.at(6,2).mapped,residentParams.size());timer.markResidentValidated();reset();auto transfer=timer.transfer(command,staging);std::vector<uint8_t> x;for(size_t i=0;i<6;++i)x.insert(x.end(),outputs[i].begin(),outputs[i].end());if(memcmp(outputs[6].data(),params,sizeof(params)))throw std::runtime_error("transfer params changed");sinterValidatePointPlaneFixture(input,x);if(x!=resident)throw std::runtime_error("resident transfer mismatch");if(slot==0)baseline=resident;else if(baseline!=resident)throw std::runtime_error("paired mismatch");timer.markTransferValidated();if(sample>=5){raw.append(std::string((const char*)resident.data(),resident.size()));raw.append(std::string((const char*)residentParams.data(),residentParams.size()));raw.append(std::string((const char*)x.data(),x.size()));raw.append(std::string((const char*)outputs[6].data(),outputs[6].size()));std::ostringstream row;row<<run<<' '<<(sample-5)<<' '<<slot<<' '<<t.rawTicks[0]<<' '<<t.rawTicks[1]<<' '<<t.rawTicks[2]<<' '<<t.rawTicks[3]<<' '<<t.hostCompletionUpperBoundNs<<' '<<transfer<<' '<<hash(resident)<<'\n';journal.append(row.str());}}}
 check(vkDeviceWaitIdle(c.device));
 }
 check(c.close());verifyLease(parse32(argv[7]));raw.finish();journal.finish();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
