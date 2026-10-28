"""Real imports must preserve both frozen numerical schedules and padded outputs."""
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_installed_point_plane_module_is_available_without_gpu():
    import sintergpu
    assert sintergpu.module_path('optimization.point_plane').is_file()


def test_normalizer_retains_precision_and_arithmetic():
    from tools.check_point_plane_module import normalized_instructions
    a='%x = OpTypeFloat 32\nOpDecorate %x RelaxedPrecision\n'
    b='%other = OpTypeFloat 32\nOpDecorate %other RelaxedPrecision\n'
    assert normalized_instructions(a)==normalized_instructions(b)
    assert normalized_instructions(a)!=normalized_instructions('%x = OpTypeFloat 64\nOpDecorate %x RelaxedPrecision\n')
    assert normalized_instructions(a)!=normalized_instructions('%x = OpTypeFloat 32\n')


def test_real_import_matches_baseline_executable_and_resource_layout(tmp_path):
    compiler=os.getenv('SINTERGPU_SLANGC') or shutil.which('slangc')
    if not compiler:pytest.skip('explicit pinned standalone compiler required')
    from tools.check_point_plane_module import verify_compiler
    result=verify_compiler(Path(compiler),tmp_path)
    assert len(result['comparisons'])==4
    assert all(r['resource_layout_equal'] and r['equal_except_declared_local_size'] for r in result['comparisons'])
    assert result['example_compiled']


@pytest.fixture
def gpu():
    if os.getenv('SINTERGPU_GPU_TESTS')!='1':pytest.skip('explicit GPU opt-in')
    import slangpy as spy
    from sintergpu import module_path
    target=os.getenv('SINTERGPU_TARGET','radeon8060s')
    assert target in ('radeon8060s','rtx4070')
    vendor,device=(0x1002,0x1586) if target=='radeon8060s' else (0x10de,0x2782)
    adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    adapter=next(a for a in adapters if a.vendor_id==vendor and a.device_id==device)
    instance=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapter.luid,
        compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise,
            'include_paths':[str(module_path('optimization.point_plane').parent.parent)]})
    try:yield instance
    finally:instance.close()


@pytest.mark.parametrize('profile',['aosoa','scalar'])
@pytest.mark.parametrize('count',[1,63,1023,1024,1025,4099,65539])
@pytest.mark.parametrize('local_size',[256,1024])
def test_imported_core_exact_frozen_outputs_and_oracle(gpu,profile,count,local_size):
    from benchmarks.point_plane_support import point_case,check_point_oracle
    case=point_case(gpu,count,profile,local_size=local_size)
    actual={v:case['read'](v) for v in ('baseline','candidate')}
    for name in case['outputs']:
        np.testing.assert_array_equal(actual['baseline'][name].view(np.uint32),actual['candidate'][name].view(np.uint32))
    check_point_oracle(case,actual['candidate'])


@pytest.mark.parametrize('profile',['aosoa','scalar'])
@pytest.mark.parametrize('local_size',[256,1024])
def test_imported_core_preserves_exceptional_values(gpu,profile,local_size):
    from benchmarks.point_plane_support import point_case
    case=point_case(gpu,65,profile,local_size=local_size)
    case['arrays']['normals'][1,:3]=[np.nan,1,2]
    case['arrays']['targets'][2,:3]=[np.inf,1,2]
    case['arrays']['normals'][3,:3]=[-0.,0.,-0.]
    case['arrays']['points'][case['arrays']['points_indices'][4],:3]=np.finfo(np.float32).max
    case['arrays']['normals'][4,:3]=[2,2,2]
    actual={v:case['read'](v) for v in ('baseline','candidate')}
    for name in case['outputs']:
        np.testing.assert_array_equal(actual['baseline'][name].view(np.uint32),actual['candidate'][name].view(np.uint32))
    assert np.isnan(actual['candidate']['out_0'][1])
    assert not np.isfinite(actual['candidate']['out_0'][2])
    assert not np.isfinite(actual['candidate']['out_0'][4])


def test_independent_rotation_first_oracle_matches_finite_differences():
    from benchmarks.point_plane_support import point_case,point_oracle
    case=point_case(None,17,'aosoa')
    arrays=case['arrays'];p=arrays['points'][arrays['points_indices'],:3].astype(np.float64)
    q=arrays['targets'][:17,:3].astype(np.float64);n=arrays['normals'][:17,:3].astype(np.float64)
    _,jac=point_oracle(case)
    step=1e-5
    for axis in range(6):
        delta=np.zeros(6);delta[axis]=step
        plus=p+np.cross(delta[:3],p)+delta[3:]
        minus=p-np.cross(delta[:3],p)-delta[3:]
        numeric=(np.sum(n*(plus-q),axis=1)-np.sum(n*(minus-q),axis=1))/(2*step)
        np.testing.assert_allclose(jac[:,axis],numeric,atol=2e-9,rtol=2e-9)
