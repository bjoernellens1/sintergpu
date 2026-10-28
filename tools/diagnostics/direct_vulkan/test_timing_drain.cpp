// SPDX-License-Identifier: Apache-2.0
#include "timing_drain.h"
#include <cassert>
#include <initializer_list>
#include <sys/wait.h>
#include <unistd.h>
static int channel=-1;static bool fail=false;
static VKAPI_ATTR VkResult VKAPI_CALL idle(VkDevice){char x='W';assert(write(channel,&x,1)==1);return fail?VK_ERROR_DEVICE_LOST:VK_SUCCESS;}
struct Query {~Query(){char x='Q';assert(write(channel,&x,1)==1);}};
int main(){for(bool failure:{false,true}){int fds[2];assert(pipe(fds)==0);pid_t child=fork();assert(child>=0);if(!child){close(fds[0]);channel=fds[1];fail=failure;{Query q;SinterTimingDrain d{(VkDevice)1,idle};}close(channel);_exit(0);}close(fds[1]);char data[3]{};int n=read(fds[0],data,3);assert(n>0);int more=read(fds[0],data+n,3-n);if(more>0)n+=more;close(fds[0]);int status=0;assert(waitpid(child,&status,0)==child);assert(WIFEXITED(status));assert(WEXITSTATUS(status)==(failure?2:0));assert(data[0]=='W');assert(n==(failure?1:2));if(!failure)assert(data[1]=='Q');}}
