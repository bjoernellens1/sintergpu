# SPDX-License-Identifier: Apache-2.0
"""Only run using the isolated instrumented SlangPy build; no Vulkan device."""
import slangpy as s
assert '/sintergpu-allocation-correlation-lab/' in s.__file__, s.__file__
d = s.Device(type=s.DeviceType.cpu)
b = d.create_buffer(size=64)
assert b.vulkan_buffer_device_correlation_v1 is None
d.close()
assert b.vulkan_buffer_device_correlation_v1 is None
print('CPU backend and closed buffer reject Vulkan correlation')
