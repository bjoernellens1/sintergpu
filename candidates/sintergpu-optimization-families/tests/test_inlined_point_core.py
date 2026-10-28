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


@pytest.mark.parametrize('profile',['aosoa','scalar'])
@pytest.mark.parametrize('count',[1,63,1023,1024,1025,4099,65539])
@pytest.mark.parametrize('candidate_local_size',[256,1024])
def test_inlined_profile_exact_frozen_outputs(profile,count,candidate_local_size):
    if os.getenv('SINTERGPU_GPU_TESTS')!='1':pytest.skip('explicit GPU opt-in')
    import slangpy as spy
    module=support()
    adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    target=os.getenv('SINTERGPU_TARGET','radeon8060s')
    vendor,device_id=(0x1002,0x1586) if target=='radeon8060s' else (0x10de,0x2782)
    adapter=next(a for a in adapters if a.vendor_id==vendor and a.device_id==device_id)
    device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapter.luid,
        compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        case=module.point_case(device,count,profile,candidate_local_size=candidate_local_size)
        actual={v:case['read'](v) for v in ('baseline','candidate')}
        for name in actual['baseline']:
            np.testing.assert_array_equal(actual['baseline'][name].view(np.uint32),actual['candidate'][name].view(np.uint32))
        module.check_point_oracle(case,actual['candidate'])
    finally:device.close()
