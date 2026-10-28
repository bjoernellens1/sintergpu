"""Run only with the isolated instrumented SlangPy build and its libsgl."""
import sys

import slangpy as s

assert '/sintergpu-rhi-receipt-lab/' in s.__file__, s.__file__
assert s.DeviceType.cpu is not None
assert len(sys.argv) == 1

d = s.Device(type=s.DeviceType.cpu)
b = d.create_buffer(size=64)
assert b.vulkan_allocation_observation_v1 is None
d.close()
assert b.vulkan_allocation_observation_v1 is None
print('CPU backend and closed SGL buffer reject Vulkan allocation observation')
