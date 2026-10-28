"""Projection division must not amplify reciprocal approximation in normals."""
from pathlib import Path
import os
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
GPU=pytest.mark.skipif(os.environ.get('SINTERGPU_GPU_TESTS')!='1',reason='explicit GPU opt-in after source clearance')


def test_seeded_geometry_numpy_matches_frozen_torch():
    torch=pytest.importorskip('torch')
    from sintergpu.references.vkl.geometry import depth_geometry
    from benchmarks import vkl_timings as m
    for c in m.cases():
        if c['kind']!='geometry':continue
        p=c['params'];b,_,h,w=c['shape'];a=c['arrays']
        tensor=lambda name,shape:torch.from_numpy(a[name].reshape(shape))
        expected=depth_geometry(tensor('depth',(b,1,h,w)),tensor('intrinsics',(b,3,3)),tensor('pose',(b,4,4)) if p['useWorld'] else None,tensor('confidence',(b,1,h,w)) if p['useConfidence'] else None)
        actual=m.reference_call(c)
        for name,value in expected.items():np.testing.assert_allclose(actual[name+'Out'][:value.numel()].reshape(value.shape),value.numpy(),atol=5e-7,rtol=8e-6,equal_nan=True)


@GPU
def test_seeded_geometry_preserves_rounded_rays_and_existing_normal_tolerance():
    import slangpy as spy
    from sintergpu.adapters.slangpy import SlangPyAdapter
    from benchmarks import vkl_timings as m
    d=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        adapter=SlangPyAdapter(d,allow_experimental=True);usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        for c in m.cases():
            if c['kind']!='geometry':continue
            expected=m.reference_call(c)
            buffers={k:d.create_buffer(element_count=a.size,struct_size=4,usage=usage,data=a) for k,a in c['arrays'].items()}
            adapter.kernel(c['operation']).dispatch(c['thread_count'],vars=dict(params=c['params'],**buffers));d.wait()
            actual={k:buffers[k].to_numpy().view(c['arrays'][k].dtype).reshape(-1) for k in c['outputs']}
            np.testing.assert_array_equal(actual['rays_cameraOut'],expected['rays_cameraOut'])
            m.check_parity(c,actual,expected)
    finally:d.close()


def test_resource_independent_division_helper_exists_without_int64_requirement():
    p=ROOT/'src/sintergpu/kernels/common/float32_divide.slang'
    assert p.is_file(),'correctly-rounded scalar32 division helper missing'
    assert 'uint64' not in p.read_text() and 'double' not in p.read_text()


@GPU
def test_division_helper_matches_ieee_binary32_random_and_edge_bits(tmp_path):
    import slangpy as spy
    p=ROOT/'src/sintergpu/kernels/common/float32_divide.slang'
    assert p.is_file(),'correctly-rounded division helper missing'
    rng=np.random.default_rng(27510)
    edge=np.array([0,0x80000000,1,2,3,5,7,0x80000001,0x7fffff,0x800000,0x800001,0x80800000,0x3f000000,0x3f800000,0x3fffffff,0x40000000,0x40800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0x7f800001],np.uint32)
    a=np.concatenate((rng.integers(0,2**32,65537,dtype=np.uint32),np.repeat(edge,len(edge))))
    b=np.concatenate((rng.integers(0,2**32,65537,dtype=np.uint32),np.tile(edge,len(edge))))
    source=tmp_path/'divide-probe.slang';source.write_text('#include "'+p.as_posix()+'"\nStructuredBuffer<uint> a,b;RWStructuredBuffer<uint> output;uint count;[shader("compute")][numthreads(256,1,1)]void probe(uint3 id:SV_DispatchThreadID){if(id.x<count)output[id.x]=sintergpuF32DivideBits(a[id.x],b[id.x]);}')
    d=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        module=d.load_module(str(source));kernel=d.create_compute_kernel(d.link_program([module],[module.entry_point('probe')]))
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        buffers={k:d.create_buffer(element_count=v.size,struct_size=4,usage=usage,data=v) for k,v in dict(a=a,b=b,output=np.full(a.size+13,0xdeadbeef,np.uint32)).items()}
        kernel.dispatch([a.size,1,1],vars=dict(count=a.size,**buffers));d.wait();actual=buffers['output'].to_numpy().view(np.uint32)
        with np.errstate(all='ignore'):expected=np.divide(a.view(np.float32),b.view(np.float32)).view(np.uint32)
        nan=(expected&0x7fffffff)>0x7f800000
        np.testing.assert_array_equal(actual[:a.size][~nan],expected[~nan])
        assert np.all((actual[:a.size][nan]&0x7fffffff)>0x7f800000)
        np.testing.assert_array_equal(actual[a.size:],np.full(13,0xdeadbeef,np.uint32))
    finally:d.close()


def test_point_multiply_helper_explicitly_requests_noncontraction():
    p=ROOT/'src/sintergpu/kernels/common/float32_multiply.slang'
    assert p.is_file(),'explicit noncontracted point multiplication helper missing'
    assert 'NoContraction' in p.read_text()


@GPU
def test_point_multiply_is_rounded_before_following_subtraction(tmp_path):
    import slangpy as spy
    p=ROOT/'src/sintergpu/kernels/common/float32_multiply.slang'
    assert p.is_file(),'explicit noncontracted point multiplication helper missing'
    a=np.array([0x3f800001,0xbf800001,0x3f800001,0xbf800001],np.uint32).view(np.float32)
    b=np.array([0x3f7ffffe,0x3f7ffffe,0xbf7ffffe,0xbf7ffffe],np.uint32).view(np.float32)
    c=np.array([1,-1,-1,1],np.float32)
    expected=a*b-c
    assert np.all(expected==0)
    source=tmp_path/'multiply-probe.slang';source.write_text('#include "'+p.as_posix()+'"\nStructuredBuffer<float> a,b,c;RWStructuredBuffer<float> output;[shader("compute")][numthreads(1,1,1)]void probe(uint3 id:SV_DispatchThreadID){if(id.x<4)output[id.x]=sintergpuF32MulNoContract(a[id.x],b[id.x])-c[id.x];}')
    d=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        mod=d.load_module(str(source));kernel=d.create_compute_kernel(d.link_program([mod],[mod.entry_point('probe')]))
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        buffers={k:d.create_buffer(element_count=v.size,struct_size=4,usage=usage,data=v) for k,v in dict(a=a,b=b,c=c,output=np.full(17,73,np.float32)).items()}
        kernel.dispatch([4,1,1],vars=buffers);d.wait();actual=buffers['output'].to_numpy().view(np.float32)
        np.testing.assert_array_equal(actual[:4],expected)
        np.testing.assert_array_equal(actual[4:],np.full(13,73,np.float32))
    finally:d.close()


@GPU
def test_fused_depth_preserves_controlled_frozen_rounding_before_geometry():
    import slangpy as spy
    torch=pytest.importorskip('torch')
    from sintergpu.references.vkl.depth_fuse import fuse_depth
    from sintergpu.adapters.slangpy import SlangPyAdapter
    from benchmarks import vkl_timings as m
    device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        adapter=SlangPyAdapter(device,allow_experimental=True);usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        for c in m.cases():
            if c['kind']!='fusion' or c['params']['relative']:continue
            p=c['params'];shape=(p['sources'],)+c['shape'];a=c['arrays']
            ds=torch.from_numpy(a['depths'].reshape(shape));vs=torch.from_numpy(a['valids'].reshape(shape)!=0);cs=torch.from_numpy(a['confidences'].reshape(shape))
            expected=fuse_depth(list(ds),list(vs),list(cs),consistency_threshold=p['threshold'])
            buffers={k:device.create_buffer(element_count=v.size,struct_size=4,usage=usage,data=v) for k,v in a.items()}
            adapter.kernel(c['operation']).dispatch(c['thread_count'],vars=dict(params=p,**buffers));device.wait()
            actual=buffers['depthOut'].to_numpy().view(np.float32)
            np.testing.assert_array_equal(actual[:expected['depth'].numel()],expected['depth'].numpy().reshape(-1))
            np.testing.assert_array_equal(actual[expected['depth'].numel():],a['depthOut'][expected['depth'].numel():])
    finally:device.close()
