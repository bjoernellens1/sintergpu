import os
import unittest


@unittest.skipUnless(os.getenv('SINTERGPU_GPU_TESTS') == '1', 'GPU execution explicitly requested only')
class AutodiffProbeTests(unittest.TestCase):
    def test_compositing_gradient_matches_analytic_reference(self):
        import numpy as np
        import slangpy as spy
        from sintergpu.adapters.slangpy import SlangPyAdapter
        device = spy.Device(type=spy.DeviceType.vulkan)
        adapter = SlangPyAdapter(device, allow_experimental=True)
        kernel = adapter.kernel('xsplat_vk.backward_probe.backwardProbeMain')
        output = device.create_buffer(element_count=1,struct_size=32,usage=spy.BufferUsage.unordered_access)
        kernel.dispatch(thread_count=[1,1,1],vars={'output':output})
        actual = output.to_numpy().view(np.float32)[:5]
        dot = .8*.7+.4*(-.2)+.2*.5
        expected = [.72*.31*.7,.72*.31*(-.2),.72*.31*.5,.72*(dot-.13),.31*dot+.69*.13]
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
