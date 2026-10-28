// SPDX-License-Identifier: Apache-2.0
// Empty-query diagnostic. Caller must hold the lock and verify build, ICD and source.
#include <vulkan/vulkan.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(call) do { VkResult r_=(call); if(r_!=VK_SUCCESS) { fprintf(stderr,"%s failed: %d\n",#call,(int)r_); goto fail; } } while(0)
static int number(const char *s,unsigned long *out) {
    char *end=NULL; if(!*s || *s=='-' || *s=='+') return 0;
    unsigned long v=strtoul(s,&end,0); if(!end || *end) return 0; *out=v; return 1;
}
static void uuid_text(const uint8_t *u,char *s) {
    snprintf(s,37,"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
}
int main(int argc,char **argv) {
    VkInstance instance=VK_NULL_HANDLE; VkDevice device=VK_NULL_HANDLE; VkQueryPool queries=VK_NULL_HANDLE;
    VkCommandPool pool=VK_NULL_HANDLE; VkCommandBuffer cmd=VK_NULL_HANDLE; uint64_t *ticks=NULL;
    unsigned long expected_driver=0, samples=0; int status=1;
    if(argc!=4 || !number(argv[2],&expected_driver) || expected_driver>UINT32_MAX || !number(argv[3],&samples) || samples<50 || samples>1024) {
        fprintf(stderr,"usage: probe expected-lowercase-vulkan-uuid expected-driver-version-integer samples-50-to-1024\n"); return 2;
    }
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="sintergpu-timestamp-diagnostic",.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ici,NULL,&instance));
    uint32_t n=0; CHECK(vkEnumeratePhysicalDevices(instance,&n,NULL));
    if(n!=1) { fprintf(stderr,"exactly one selected physical device required\n"); goto fail; }
    VkPhysicalDevice physical; CHECK(vkEnumeratePhysicalDevices(instance,&n,&physical));
    VkPhysicalDeviceIDProperties id={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    id.pNext=&driver;
    VkPhysicalDeviceProperties2 properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&id};
    vkGetPhysicalDeviceProperties2(physical,&properties);
    char uuid[37]; uuid_text(id.deviceUUID,uuid);
    if(strcmp(uuid,argv[1]) || properties.properties.driverVersion!=(uint32_t)expected_driver ||
       properties.properties.apiVersion<VK_API_VERSION_1_2 || properties.properties.limits.timestampPeriod<=0 ||
       !properties.properties.limits.timestampComputeAndGraphics) {
        fprintf(stderr,"UUID, driver, Vulkan version or timestamp capability mismatch\n"); goto fail;
    }
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(physical,&qn,NULL);
    if(qn==0 || qn>256) { fprintf(stderr,"unsupported queue family count\n"); goto fail; }
    VkQueueFamilyProperties families[256]; vkGetPhysicalDeviceQueueFamilyProperties(physical,&qn,families);
    uint32_t family=UINT32_MAX;
    for(uint32_t i=0;i<qn;i++) if((families[i].queueFlags&VK_QUEUE_COMPUTE_BIT) && families[i].queueCount && families[i].timestampValidBits) { family=i; break; }
    if(family==UINT32_MAX || families[family].timestampValidBits>64) { fprintf(stderr,"compute timestamp queue unavailable\n"); goto fail; }
    float priority=1.0f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    CHECK(vkCreateDevice(physical,&dci,NULL,&device));
    VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    VkQueryPoolCreateInfo qi={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=(uint32_t)(2*samples)};
    CHECK(vkCreateQueryPool(device,&qi,NULL,&queries));
    VkCommandPoolCreateInfo pi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};
    CHECK(vkCreateCommandPool(device,&pi,NULL,&pool));
    VkCommandBufferAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK(vkAllocateCommandBuffers(device,&ai,&cmd));
    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(cmd,&bi)); vkCmdResetQueryPool(cmd,queries,0,(uint32_t)(2*samples));
    for(uint32_t i=0;i<(uint32_t)samples;i++) {
        vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,2*i);
        vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,2*i+1);
    }
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
    CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue));
    ticks=calloc(2*samples,sizeof(*ticks)); if(!ticks) { fprintf(stderr,"allocation failed\n"); goto fail; }
    CHECK(vkGetQueryPoolResults(device,queries,0,(uint32_t)(2*samples),2*samples*sizeof(*ticks),ticks,sizeof(*ticks),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
    printf("{\"device_uuid\":\"%s\",\"vendor_id\":%u,\"device_id\":%u,\"driver_version\":%u,\"driver_id\":%u,\"api_version\":%u,\"timestamp_period_ns\":%.9g,\"queue_family\":%u,\"timestamp_valid_bits\":%u,\"query_pairs\":[",uuid,properties.properties.vendorID,properties.properties.deviceID,properties.properties.driverVersion,(unsigned)driver.driverID,properties.properties.apiVersion,(double)properties.properties.limits.timestampPeriod,family,families[family].timestampValidBits);
    for(unsigned long i=0;i<samples;i++) printf("%s[%" PRIu64 ",%" PRIu64 "]",i?",":"",ticks[2*i],ticks[2*i+1]);
    puts("]}"); status=0;
fail:
    free(ticks);
    if(device) { if(pool) vkDestroyCommandPool(device,pool,NULL); if(queries) vkDestroyQueryPool(device,queries,NULL); vkDestroyDevice(device,NULL); }
    if(instance) vkDestroyInstance(instance,NULL);
    return status;
}
