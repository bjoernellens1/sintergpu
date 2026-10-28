"""Zero-overhead point-plane profile: numerical and emitted-instruction gates."""
import os
from pathlib import Path
import importlib.util

import numpy as np
import pytest

CANDIDATE=Path(__file__).resolve().parents[1]


def support():
    path=CANDIDATE/'benchmarks/inlined_point_support.py'
    assert path.is_file(), 'inlined point-plane profile not implemented'
    spec=importlib.util.spec_from_file_location('inlined_point_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def gpu():
    if os.getenv('SINTERGPU_GPU_TESTS')!='1':pytest.skip('explicit GPU opt-in')
    import slangpy as spy
    adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    target=os.getenv('SINTERGPU_TARGET','radeon8060s')
    vendor,device_id=(0x1002,0x1586) if target=='radeon8060s' else (0x10de,0x2782)
    adapter=next(a for a in adapters if a.vendor_id==vendor and a.device_id==device_id)
    device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapter.luid,
        compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:yield device
    finally:device.close()


@pytest.mark.parametrize('profile',['aosoa','scalar'])
@pytest.mark.parametrize('count',[1,63,1023,1024,1025,4099,65539])
@pytest.mark.parametrize('candidate_local_size',[256,1024])
def test_inlined_profile_exact_frozen_outputs(gpu,profile,count,candidate_local_size):
    module=support()
    case=module.point_case(gpu,count,profile,candidate_local_size=candidate_local_size)
    actual={v:case['read'](v) for v in ('baseline','candidate')}
    for name in actual['baseline']:
        np.testing.assert_array_equal(actual['baseline'][name].view(np.uint32),actual['candidate'][name].view(np.uint32))
    module.check_point_oracle(case,actual['candidate'])


@pytest.mark.parametrize('profile',['aosoa','scalar'])
@pytest.mark.parametrize('candidate_local_size',[256,1024])
def test_inlined_profile_preserves_exceptional_values(gpu,profile,candidate_local_size):
    case=support().point_case(gpu,65,profile,candidate_local_size=candidate_local_size)
    case['arrays']['normals'][1,:3]=[np.nan,1,2]
    case['arrays']['targets'][2,:3]=[np.inf,1,2]
    case['arrays']['normals'][3,:3]=[-0.,0.,-0.]
    case['arrays']['points'][case['arrays']['points_indices'][4],:3]=np.finfo(np.float32).max
    case['arrays']['normals'][4,:3]=[2,2,2]
    actual={v:case['read'](v) for v in ('baseline','candidate')}
    for name in actual['baseline']:
        np.testing.assert_array_equal(actual['baseline'][name].view(np.uint32),actual['candidate'][name].view(np.uint32))
    assert np.isnan(actual['candidate']['out_0'][1])
    assert not np.isfinite(actual['candidate']['out_0'][2])
    assert not np.isfinite(actual['candidate']['out_0'][4])
