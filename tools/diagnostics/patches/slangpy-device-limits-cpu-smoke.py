# SPDX-License-Identifier: Apache-2.0
"""Use the isolated schema-2 runtime; never initializes Vulkan."""
import slangpy as s

names = ('device_limits_known', 'max_storage_buffer_range', 'max_uniform_buffer_range',
         'min_storage_buffer_offset_alignment', 'min_uniform_buffer_offset_alignment',
         'max_compute_work_group_count', 'max_compute_work_group_size',
         'max_compute_work_group_invocations', 'dispatch_addresses_known',
         'get_instance_proc_addr_address', 'get_physical_device_properties_address')
assert all(hasattr(s.VulkanDeviceCreationReceipt, name) for name in names)
d = s.Device(type=s.DeviceType.cpu)
assert d.vulkan_creation_receipt is None
d.close()
assert d.vulkan_creation_receipt is None
print('schema-2 fields bound; CPU device has no Vulkan receipt before/after close')
