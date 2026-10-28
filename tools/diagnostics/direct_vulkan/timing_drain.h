// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vulkan/vulkan.h>
#include <cstdlib>
// On failure, do not unwind and destroy resources that may still be in use.
// The parent must record the abnormal child exit as failed evidence.
struct SinterTimingDrain { VkDevice device; PFN_vkDeviceWaitIdle wait; ~SinterTimingDrain(){if(device&&(!wait||wait(device)!=VK_SUCCESS))std::_Exit(2);} };
