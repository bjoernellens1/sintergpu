# SPDX-License-Identifier: Apache-2.0
"""Requires isolated patched runtime; creates only a CPU device."""
import slangpy as s
assert hasattr(s.CommandBuffer, 'vulkan_recorded_dispatches_v1')
d = s.Device(type=s.DeviceType.cpu)
command = d.create_command_encoder().finish()
assert command.vulkan_recorded_dispatches_v1 is None
d.close()
assert command.vulkan_recorded_dispatches_v1 is None
print('CPU command buffer has no Vulkan dispatch observation before/after close')
