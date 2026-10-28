"""Candidate-only numerical gates; no catalog registration or promotion."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

CANDIDATE = Path(__file__).resolve().parents[1]
ROOT = CANDIDATE.parents[1]
KERNELS = ROOT / 'src/sintergpu/kernels'


def test_candidate_device_functions_are_present():
    for name in ('point_plane', 'small_cholesky'):
        assert (CANDIDATE / 'kernels/optimization' / (name + '.slang')).is_file(), name


def test_prevalidated_solver_core_is_a_separate_profile():
    text=(CANDIDATE/'kernels/optimization/small_cholesky.slang').read_text()
    assert 'sinterSmallCholeskyCore' in text, 'missing core for the incumbent prevalidated-dimension profile'


def symbolic():
    path = CANDIDATE / 'python/point_plane_symbolic.py'
    assert path.is_file(), 'shared symbolic expression factory is missing'
    spec = importlib.util.spec_from_file_location('candidate_symbolic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_symbolic_exports_match_independent_finite_differences():
    sf = pytest.importorskip('symforce.symbolic')
    factory = symbolic()
    rng = np.random.default_rng(31)
    for _ in range(16):
        p, q, normal = rng.normal(size=(3, 3))
        residual, jacobian = factory.point_plane_expression(sf, sf.V3(p), sf.V3(q), sf.V3(normal))
        values = np.array([float(residual)] + [float(x) for x in jacobian])
        expected = np.r_[normal @ (p - q), np.cross(p, normal), normal]
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=1e-12)
        # Independent central difference of a left infinitesimal rigid action.
        steps = np.eye(6) * 1e-5
        def evaluate(xi):
            return normal @ (p + np.cross(xi[:3], p) + xi[3:] - q)
        fd = np.array([(evaluate(step) - evaluate(-step)) / 2e-5 for step in steps])
        np.testing.assert_allclose(values[1:], fd, atol=2e-10, rtol=2e-10)
        scalars = factory.point_plane_scalars(sf, *p, *q, *normal)
        np.testing.assert_allclose([float(x) for x in scalars], values, atol=1e-12, rtol=1e-12)


@pytest.fixture(scope='module')
def gpu():
    if os.environ.get('SINTERGPU_GPU_TESTS') != '1':
        pytest.skip('GPU tests explicitly requested only')
    import slangpy as spy
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    if os.getenv('SINTERGPU_TARGET','radeon8060s')=='rtx4070':
        adapter=next(a for a in adapters if a.vendor_id==0x10de and a.name.endswith('GeForce RTX 4070 Ti'))
    else:
        adapter=next(a for a in adapters if a.vendor_id==0x1002 and a.device_id==0x1586)
    device = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=adapter.luid,
        compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    yield device
    device.close()


def support():
    path = CANDIDATE / 'benchmarks/family_support.py'
    assert path.is_file(), 'candidate test/benchmark harness missing'
    spec = importlib.util.spec_from_file_location('family_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('profile', ['aosoa', 'scalar'])
@pytest.mark.parametrize('count', [1, 63, 1023, 1024, 1025, 4099])
def test_point_plane_exact_frozen_schedule_and_oracle(gpu, profile, count):
    harness = support()
    case = harness.point_case(gpu, count, profile)
    baseline = case['read']('baseline')
    candidate = case['read']('candidate')
    for key in baseline:
        # Include poisoned padding, nonfinite classes and signed zeros bitwise.
        np.testing.assert_array_equal(candidate[key].view(np.uint32), baseline[key].view(np.uint32))
    harness.check_point_oracle(case, candidate)


@pytest.mark.parametrize('dimension', [6, 12, 18])
@pytest.mark.parametrize('kind', ['spd', 'singular', 'indefinite', 'nonfinite', 'rhs-nonfinite', 'few-observations', 'damping', 'upper-poison'])
def test_cholesky_exact_frozen_loop_and_numpy_solve(gpu, dimension, kind):
    harness = support()
    case = harness.solve_case(gpu, dimension, kind)
    baseline = case['read']('baseline')
    candidate = case['read']('candidate')
    for key in baseline:
        np.testing.assert_array_equal(candidate[key].view(np.uint32), baseline[key].view(np.uint32))
    harness.check_solve_oracle(case, candidate)


@pytest.mark.parametrize('kind', ['spd','few-observations','damping'])
def test_schur6_observation_guard_is_distinct_from_caspar6(gpu,kind):
    harness=support();case=harness.solve_case(gpu,6,kind,schur=True)
    baseline=case['read']('baseline');candidate=case['read']('candidate')
    for key in baseline:
        np.testing.assert_array_equal(candidate[key].view(np.uint32),baseline[key].view(np.uint32))
    harness.check_solve_oracle(case,candidate)


@pytest.mark.parametrize('profile',['aosoa','scalar'])
def test_point_plane_nonfinite_and_signed_zero_are_not_filtered(gpu,profile):
    harness=support();case=harness.point_case(gpu,65,profile)
    case['arrays']['normals'][1,:3]=[np.nan,1,2]
    case['arrays']['targets'][2,:3]=[np.inf,1,2]
    case['arrays']['normals'][3,:3]=[-0.,0.,-0.]
    case['arrays']['points'][case['arrays']['points_indices'][4],:3]=np.finfo(np.float32).max
    case['arrays']['normals'][4,:3]=[2,2,2]
    for variant in ('baseline','candidate'):case['upload'](variant)
    baseline=case['read']('baseline');candidate=case['read']('candidate')
    for key in baseline:
        np.testing.assert_array_equal(candidate[key].view(np.uint32),baseline[key].view(np.uint32))
    assert np.isnan(candidate['out_0'][1])
    assert not np.isfinite(candidate['out_0'][2])
    assert not np.isfinite(candidate['out_0'][4])


@pytest.mark.parametrize('dimension',[0,7])
def test_cholesky_rejects_out_of_capacity_without_reading_terms(gpu,dimension):
    harness=support()
    helper=(CANDIDATE/'kernels/optimization/small_cholesky.slang').read_text()
    source=helper+f'''
StructuredBuffer<float> input;
RWStructuredBuffer<float> output;
RWStructuredBuffer<uint> status;
[shader("compute")][numthreads(1,1,1)]
void main() {{float x[6];bool valid=sinterSmallCholesky<6>(input,0,{dimension},0,true,x);
 for(uint i=0;i<6;++i)output[i]=x[i];status[0]=valid?0:1;}}
'''
    case=harness.make_case(gpu,dict(candidate=source),dict(input=np.ones(1,np.float32),
        output=np.ones(13,np.float32),status=np.zeros(1,np.uint32)),('output','status'),{},[1,1,1])
    actual=case['read']('candidate')
    np.testing.assert_array_equal(actual['output'][:6],0)
    np.testing.assert_array_equal(actual['output'][6:],1)
    np.testing.assert_array_equal(actual['status'],1)


@pytest.mark.parametrize('dimension',[6,18])
@pytest.mark.parametrize('kind',['spd','singular','rhs-nonfinite'])
def test_checked_cholesky_direct_success_and_failure_clear(gpu,dimension,kind):
    harness=support();case=harness.solve_case(gpu,dimension,kind)
    # Export the checked helper's x directly: no adapter zeroing can mask a bug.
    source=case['sources']['candidate'].replace(f'sinterSmallCholeskyCore<{dimension}>(inputTerms',
        f'sinterSmallCholesky<{dimension}>(inputTerms')
    source=source.replace('delta[i]=valid?x[i]:0.0;', 'delta[i]=x[i];')
    case['kernels']['candidate']=harness.compile_kernel(gpu,source)
    actual=case['read']('candidate')
    harness.check_solve_oracle(case,actual)


@pytest.mark.parametrize('count', [1, 63, 64, 65, 1025, 4099])
@pytest.mark.parametrize('tree', ['workgroup', 'subgroup'])
def test_fused_point_plane_preserves_full_gn_outputs(gpu, count, tree):
    harness = support()
    case = harness.fusion_case(gpu, count, tree)
    baseline = case['read']('baseline')
    candidate = case['read']('candidate')
    for key in baseline:
        np.testing.assert_array_equal(candidate[key].view(np.uint32), baseline[key].view(np.uint32))
    harness.check_fusion_oracle(case, candidate)
