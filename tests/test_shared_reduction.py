import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.getenv('SINTERGPU_GPU_TESTS') == '1', 'GPU execution explicitly requested only')
class SharedReductionTests(unittest.TestCase):
    def test_exact_fixed_tree_sum5_and_min1(self):
        import numpy as np
        import slangpy as spy
        import sintergpu
        helper = (Path(sintergpu.__file__).parent/'kernels/common/block_reduce.slang').read_text()
        device = spy.Device(type=spy.DeviceType.vulkan)
        for scalar, channels, combine in [('float',5,'SINTERGPU_ADD'),('uint',1,'min')]:
            with self.subTest(scalar):
                count = 253
                dtype = np.float32 if scalar == 'float' else np.uint32
                data = np.random.default_rng(17).uniform(-10,10,(channels,256)).astype(dtype) if scalar == 'float' else np.random.default_rng(17).integers(0,1000,(channels,256),dtype=dtype)
                data[:,count:] = 0 if scalar == 'float' else 1000000
                golden = data.copy()
                for stride in (128,64,32,16,8,4,2,1):
                    golden[:,:stride] = golden[:,:stride]+golden[:,stride:2*stride] if scalar == 'float' else np.minimum(golden[:,:stride],golden[:,stride:2*stride])
                source = helper + f'''
StructuredBuffer<{scalar}> input;
RWStructuredBuffer<{scalar}> output;
groupshared {scalar} sharedValues[{channels*256}];
[shader("compute")][numthreads(256,1,1)]
void main(uint lane:SV_GroupIndex) {{
 for(uint c=0;c<{channels};++c) sharedValues[c*256+lane]=input[c*256+lane];
 GroupMemoryBarrierWithGroupSync();
 SINTERGPU_REDUCE_256(sharedValues,lane,{channels},{combine})
 if(lane==0) for(uint c=0;c<{channels};++c) output[c]=sharedValues[c*256];
}}
'''
                module = device.load_module_from_source('shared_reduce_'+scalar,source)
                program = device.link_program([module],[module.entry_point('main')])
                kernel = device.create_compute_kernel(program)
                usage = spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                input_buffer = device.create_buffer(element_count=data.size,struct_size=4,usage=usage,data=data.reshape(-1))
                output = device.create_buffer(element_count=channels,struct_size=4,usage=usage)
                kernel.dispatch(thread_count=[256,1,1],vars={'input':input_buffer,'output':output})
                actual = output.to_numpy().view(dtype)
                np.testing.assert_array_equal(actual,golden[:,0])


if __name__ == '__main__':
    unittest.main()
