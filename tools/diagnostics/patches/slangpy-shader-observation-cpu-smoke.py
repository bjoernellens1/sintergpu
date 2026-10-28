# SPDX-License-Identifier: Apache-2.0
"""Requires isolated patched runtime; uses only SlangPy's CPU device."""
import slangpy as s
assert hasattr(s.ShaderProgram, 'vulkan_shader_bytes_observation_v1')
d = s.Device(type=s.DeviceType.cpu)
module = d.load_module_from_source('cpu_observation_smoke', '[shader("compute")][numthreads(1,1,1)] void main(uint3 tid:SV_DispatchThreadID) {}')
program = d.link_program([module], [module.entry_point('main')])
kernel = d.create_compute_kernel(program)
assert program.vulkan_shader_bytes_observation_v1 is None
d.close()
assert program.vulkan_shader_bytes_observation_v1 is None
print('CPU program has no Vulkan shader bytes before/after close')
