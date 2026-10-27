"""Independent binary feature parity with the frozen VKL implementations."""
import importlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def numeric():
    try:return importlib.import_module('sintergpu.references.features_numeric')
    except ImportError:pytest.fail('independent binary feature references missing')


def frozen(name):
    path=ROOT/'references/donors/vkl/src/vkl/kernels/reference'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('frozen_feature_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def brief_case(channels=3,custom=False):
    rng=np.random.default_rng(21);rgb=rng.random((2,channels,37,39),dtype=np.float32)
    points=np.array([[[18,18],[0,0],[-1,9],[2**62,2**62],[38,36],[17,16]]]*2,np.int64)
    valid=np.ones((2,6),bool);valid[1,0]=False
    domain=np.ones((2,1,37,39),bool);domain[0,0,1,1]=False;rgb[1,:,0,0]=np.nan
    pairs=np.array([[[0,0],[1,0]],[[-1,-1],[2,2]],[[2,0],[-2,0]]],np.int64) if custom else None
    return rgb,points,valid,domain,pairs


def match_case(bits=35,points=19,landmarks=17):
    rng=np.random.default_rng(53);a=rng.integers(0,2,(2,landmarks,bits)).astype(bool);b=rng.integers(0,2,(2,points,bits)).astype(bool)
    xy=rng.uniform(-2,4,(2,landmarks,2)).astype(np.float32);centers=rng.integers(-2,5,(2,points,2),dtype=np.int64)
    av=np.ones(a.shape[:2],bool);bv=np.ones(b.shape[:2],bool)
    xy[0,0]=[np.nan,0];av[-1,-1]=False;bv[0,0]=False
    return a,b,xy,centers,av,bv


@pytest.mark.parametrize('channels',[1,3])
@pytest.mark.parametrize('custom',[False,True])
def test_brief_numpy_matches_frozen(channels,custom):
    torch=pytest.importorskip('torch');rgb,p,v,domain,pairs=brief_case(channels,custom)
    kwargs=dict(pixels_valid=v,rgb_valid=domain,pairs=pairs,num_bits=259)
    actual=numeric().brief_describe(rgb,p,**kwargs)
    expected=frozen('brief').brief_describe(torch.from_numpy(rgb),torch.from_numpy(p),**{k:torch.from_numpy(x) if isinstance(x,np.ndarray) else x for k,x in kwargs.items()})
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key].numpy(),err_msg=key)


@pytest.mark.parametrize('mutual',[False,True])
@pytest.mark.parametrize('chunk',[None,1,7,100])
def test_binary_matching_numpy_matches_frozen(mutual,chunk):
    torch=pytest.importorskip('torch');a,b,xy,cp,av,bv=match_case()
    kwargs=dict(radius_px=3.5,max_hamming_distance=35,mutual=mutual,landmark_valid=av,candidate_valid=bv,candidate_chunk_size=chunk)
    actual=numeric().binary_local_match(a,b,xy,cp,**kwargs)
    expected=frozen('local_match').binary_local_match(*map(torch.from_numpy,(a,b,xy,cp)),**{k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in kwargs.items()})
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key].numpy(),err_msg=key)


def test_radius_and_distance_ties_mutual_ownership_and_invalid_outputs():
    a=np.zeros((1,3,4),bool);b=np.zeros((1,2,4),bool);p=np.zeros((1,3,2),np.float32);c=np.array([[[3,4],[0,0]]],np.int64)
    for mutual in (False,True):
        r=numeric().binary_local_match(a,b,p,c,radius_px=5,max_hamming_distance=0,mutual=mutual,candidate_chunk_size=1)
        np.testing.assert_array_equal(r['matched_index'],[[0,-1,-1]] if mutual else [[0,0,0]])
        np.testing.assert_array_equal(r['distance'],0)
    c[:]=np.iinfo(np.int64).max
    r=numeric().binary_local_match(a,b,p,c,radius_px=1,max_hamming_distance=4)
    np.testing.assert_array_equal(r['matched_index'],-1);np.testing.assert_array_equal(r['valid'],False)


def test_setup_plans_are_bounded_typed_and_chunked_without_runtime_imports():
    try:m=importlib.import_module('sintergpu.features')
    except ImportError:pytest.fail('resource-independent feature setup missing')
    kw=dict(max_workgroups_x=65535,max_storage_buffer_range=1<<27)
    brief=m.brief_layout(2,3,37,39,6,259,**kw)
    assert brief['elements']['pixels']==24 and brief['dtypes']['pixels']=='int64'
    plan=m.binary_match_plan(2,17,19,35,radius_px=3.5,max_hamming_distance=35,chunk_size=7,**kw)
    assert plan['elements']['pairDistance']==2*17*7
    assert [s['params']['chunkStart'] for s in plan['stages'] if s['entrypoint']=='matchPairs']==[0,7,14]
    assert [s['params']['chunkCount'] for s in plan['stages'] if s['entrypoint']=='matchPairs']==[7,7,5]
    for changes in (dict(max_workgroups_x=1),dict(max_storage_buffer_range=16)):
        with pytest.raises(ValueError):m.brief_layout(2,3,37,39,6,259,**(kw|changes))
    with pytest.raises(ValueError):m.binary_match_plan(1,3,4,8,radius_px=0,max_hamming_distance=8,**kw)
    with pytest.raises(ValueError):m.binary_match_plan(1,3,4,8,radius_px=1,max_hamming_distance=9,**kw)
    assert m.brief_layout(1,1,1,1,0,1,**kw)['stages']==[]


def test_nonmutual_plan_does_not_require_unused_reverse_dispatch_and_empty_recording_is_noop():
    from sintergpu.features import binary_match_plan,brief_layout,record_feature_plan
    plan=binary_match_plan(1,1,1000,1,radius_px=1,max_hamming_distance=1,mutual=False,max_workgroups_x=4,max_storage_buffer_range=1<<27)
    assert all(s['entrypoint']!='reverseMinimum' for s in plan['stages'])
    record_feature_plan({}, {},brief_layout(1,1,1,1,0,1,max_workgroups_x=1,max_storage_buffer_range=16),command_encoder=object())


def test_new_feature_shader_entries_exist():
    brief=ROOT/'src/sintergpu/kernels/vkl/brief.slang';match=ROOT/'src/sintergpu/kernels/vkl/binary_match.slang'
    assert brief.is_file() and match.is_file(),'feature compute implementations missing'
    for entry in ('packBinary','initializeMatch','matchPairs','rowMinimum','reverseMinimum','finalizeMatch'):
        assert 'void '+entry+'(' in match.read_text()


def test_feature_catalog_declares_native_int64_requirement():
    from sintergpu.catalog import get_operation
    for op in ('vkl.brief.briefDescribe','vkl.binary_match.matchPairs'):
        assert 'shaderInt64' in get_operation(op)['requirements']


@pytest.fixture
def gpu_runner():
    if __import__('os').environ.get('SINTERGPU_GPU_TESTS')!='1':pytest.skip('explicit GPU opt-in after independent source clearance')
    from benchmarks.vkl_features import FeatureRunner
    r=FeatureRunner()
    try:yield r
    finally:r.close()


@pytest.mark.parametrize('channels',[1,3])
@pytest.mark.parametrize('custom',[False,True])
def test_gpu_brief_complete_outputs(gpu_runner,channels,custom):
    from sintergpu.features import brief_layout
    rgb,p,v,domain,pairs=brief_case(channels,custom)
    bits=259 if pairs is None else len(pairs)
    layout=brief_layout(2,channels,37,39,6,bits,default_pairs=pairs is None,use_pixels_valid=True,use_rgb_valid=True,**gpu_runner.limits)
    inputs=dict(rgb=rgb,pixels=p,pixelsValid=v.astype(np.uint32),rgbValid=domain.astype(np.uint32))
    if pairs is not None:inputs['pairs']=pairs
    prepared=gpu_runner.prepare(layout,inputs);gpu_runner.execute(prepared)
    actual=gpu_runner.read(prepared,('descriptorOut','briefValidOut'));expected=numeric().brief_describe(rgb,p,pixels_valid=v,rgb_valid=domain,pairs=pairs,num_bits=bits)
    np.testing.assert_array_equal(actual['descriptorOut'].reshape(expected['descriptor'].shape),expected['descriptor'])
    np.testing.assert_array_equal(actual['briefValidOut'].reshape(expected['valid'].shape),expected['valid'])


@pytest.mark.parametrize('mutual',[False,True])
@pytest.mark.parametrize('chunk',[None,1,7,100])
def test_gpu_matching_all_outputs_and_chunk_invariance(gpu_runner,mutual,chunk):
    from sintergpu.features import binary_match_plan
    a,b,xy,cp,av,bv=match_case(257)
    plan=binary_match_plan(2,17,19,257,radius_px=3.5,max_hamming_distance=257,chunk_size=chunk,mutual=mutual,use_landmark_valid=True,use_candidate_valid=True,**gpu_runner.limits)
    prepared=gpu_runner.prepare(plan,dict(landmarkDescriptor=a.astype(np.uint32),candidateDescriptor=b.astype(np.uint32),projected=xy,candidateXY=cp,landmarkValid=av.astype(np.uint32),candidateValid=bv.astype(np.uint32)))
    gpu_runner.execute(prepared);actual=gpu_runner.read(prepared,('matchedIndexOut','distanceOut','matchValidOut'))
    expected=numeric().binary_local_match(a,b,xy,cp,radius_px=3.5,max_hamming_distance=257,mutual=mutual,landmark_valid=av,candidate_valid=bv,candidate_chunk_size=chunk)
    for name,key in [('matchedIndexOut','matched_index'),('distanceOut','distance'),('matchValidOut','valid')]:np.testing.assert_array_equal(actual[name].reshape(2,17),expected[key])


def test_gpu_bit_arithmetic_matches_independent_float32_operations(gpu_runner):
    r=gpu_runner;spy=r.spy;rng=np.random.default_rng(381)
    edge=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345],np.uint32)
    a=np.concatenate((rng.integers(0,2**32,65537,dtype=np.uint32),np.repeat(edge,len(edge))))
    b=np.concatenate((rng.integers(0,2**32,65537,dtype=np.uint32),np.tile(edge,len(edge))))
    common=(ROOT/'src/sintergpu/kernels/common/float32_bits.slang').read_text()
    source=common+'''\nStructuredBuffer<uint> a,b;RWStructuredBuffer<uint> result;uint count;
[shader("compute")][numthreads(256,1,1)] void probe(uint3 id:SV_DispatchThreadID) { uint i=id.x;if(i>=count)return;result[i*3]=sintergpuF32AddBits(a[i],b[i]);result[i*3+1]=sintergpuF32MulBits(a[i],b[i]);result[i*3+2]=uint(sintergpuF32LessBits(a[i],b[i])); }'''
    module=r.device.load_module_from_source('binary32_probe',source);kernel=r.device.create_compute_kernel(r.device.link_program([module],[module.entry_point('probe')]))
    usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
    make=lambda x:r.device.create_buffer(element_count=x.size,struct_size=4,usage=usage,data=x)
    output=make(np.full(a.size*3,0xdeadbeef,np.uint32))
    kernel.dispatch(thread_count=[a.size,1,1],vars=dict(a=make(a),b=make(b),result=output,count=a.size));r.device.wait()
    actual=output.to_numpy().view(np.uint32).reshape(-1,3)
    with np.errstate(all='ignore'):expected=[np.add(a.view(np.float32),b.view(np.float32),dtype=np.float32),np.multiply(a.view(np.float32),b.view(np.float32),dtype=np.float32),a.view(np.float32)<b.view(np.float32)]
    for i,e in enumerate(expected[:2]):
        got=actual[:,i].view(np.float32);np.testing.assert_array_equal(np.isnan(got),np.isnan(e));np.testing.assert_array_equal(actual[~np.isnan(e),i],e.view(np.uint32)[~np.isnan(e)])
    np.testing.assert_array_equal(actual[:,2],expected[2])


@pytest.mark.parametrize('chunk',[None,129])
def test_gpu_matching_reductions_cross_256_threads(gpu_runner,chunk):
    from sintergpu.features import binary_match_plan
    a=np.zeros((1,257,35),bool);b=np.zeros((1,259,35),bool);a[0,-1]=True;b[0,-1]=True
    xy=np.zeros((1,257,2),np.float32);cp=np.zeros((1,259,2),np.int64)
    plan=binary_match_plan(1,257,259,35,radius_px=1,max_hamming_distance=0,chunk_size=chunk,**gpu_runner.limits)
    p=gpu_runner.prepare(plan,dict(landmarkDescriptor=a.astype(np.uint32),candidateDescriptor=b.astype(np.uint32),projected=xy,candidateXY=cp));gpu_runner.execute(p)
    actual=gpu_runner.read(p,('matchedIndexOut','distanceOut','matchValidOut'));expected=numeric().binary_local_match(a,b,xy,cp,radius_px=1,max_hamming_distance=0,candidate_chunk_size=chunk)
    for name,key in [('matchedIndexOut','matched_index'),('distanceOut','distance'),('matchValidOut','valid')]:np.testing.assert_array_equal(actual[name].reshape(1,257),expected[key])


@pytest.mark.parametrize('channels',[1,3])
def test_gpu_brief_subnormals_and_signed_zero_follow_frozen_comparisons(gpu_runner,channels):
    from sintergpu.features import brief_layout
    rgb=np.zeros((1,channels,1,4),np.float32);rgb[0,-1,0]=np.array([10,20,0x80000000,0],np.uint32).view(np.float32)
    pixels=np.array([[[0,0]]],np.int64);pairs=np.array([[[0,0],[1,0]],[[2,0],[3,0]]],np.int64)
    plan=brief_layout(1,channels,1,4,1,2,default_pairs=False,**gpu_runner.limits)
    p=gpu_runner.prepare(plan,dict(rgb=rgb,pixels=pixels,pairs=pairs));gpu_runner.execute(p)
    actual=gpu_runner.read(p,('descriptorOut','briefValidOut'));expected=numeric().brief_describe(rgb,pixels,pairs=pairs)
    np.testing.assert_array_equal(actual['descriptorOut'].reshape(1,1,2),expected['descriptor']);np.testing.assert_array_equal(actual['briefValidOut'],1)


@pytest.mark.parametrize('radius,offset',[(1e-20,1.01e-20),(1e308,3e38)])
def test_gpu_matching_subnormal_and_overflow_radius_gates(gpu_runner,radius,offset):
    from sintergpu.features import binary_match_plan
    a=np.zeros((1,2,1),bool);b=np.zeros((1,1,1),bool);xy=np.array([[[offset,0],[0,0]]],np.float32);cp=np.zeros((1,1,2),np.int64)
    plan=binary_match_plan(1,2,1,1,radius_px=radius,max_hamming_distance=0,mutual=False,**gpu_runner.limits)
    p=gpu_runner.prepare(plan,dict(landmarkDescriptor=a.astype(np.uint32),candidateDescriptor=b.astype(np.uint32),projected=xy,candidateXY=cp));gpu_runner.execute(p)
    actual=gpu_runner.read(p,('matchedIndexOut','distanceOut','matchValidOut'));expected=numeric().binary_local_match(a,b,xy,cp,radius_px=radius,max_hamming_distance=0,mutual=False)
    for name,key in [('matchedIndexOut','matched_index'),('distanceOut','distance'),('matchValidOut','valid')]:np.testing.assert_array_equal(actual[name].reshape(1,2),expected[key])


def test_gpu_extreme_int64_coordinates_and_offsets(gpu_runner):
    from sintergpu.features import brief_layout,binary_match_plan
    torch=pytest.importorskip('torch');minimum=np.iinfo(np.int64).min;maximum=np.iinfo(np.int64).max
    rgb=np.zeros((1,1,2,2),np.float32);points=np.array([[[0,0],[maximum,minimum]]],np.int64)
    pairs=np.array([[[minimum,maximum],[0,0]]],np.int64)
    plan=brief_layout(1,1,2,2,2,1,default_pairs=False,**gpu_runner.limits)
    p=gpu_runner.prepare(plan,dict(rgb=rgb,pixels=points,pairs=pairs));gpu_runner.execute(p)
    expected=frozen('brief').brief_describe(torch.from_numpy(rgb),torch.from_numpy(points),pairs=torch.from_numpy(pairs))
    actual=gpu_runner.read(p,('descriptorOut','briefValidOut'))
    np.testing.assert_array_equal(actual['descriptorOut'].reshape(1,2,1),expected['descriptor'].numpy());np.testing.assert_array_equal(actual['briefValidOut'].reshape(1,2),expected['valid'].numpy())
    a=np.zeros((1,2,1),bool);b=np.zeros((1,3,1),bool)
    xy=np.array([[[maximum,maximum],[minimum,minimum]]],np.float32);cp=np.array([[[maximum,maximum],[minimum,minimum],[0,0]]],np.int64)
    plan=binary_match_plan(1,2,3,1,radius_px=1,max_hamming_distance=0,**gpu_runner.limits)
    p=gpu_runner.prepare(plan,dict(landmarkDescriptor=a.astype(np.uint32),candidateDescriptor=b.astype(np.uint32),projected=xy,candidateXY=cp));gpu_runner.execute(p)
    expected=frozen('local_match').binary_local_match(*map(torch.from_numpy,(a,b,xy,cp)),radius_px=1,max_hamming_distance=0)
    actual=gpu_runner.read(p,('matchedIndexOut','distanceOut','matchValidOut'))
    for name,key in [('matchedIndexOut','matched_index'),('distanceOut','distance'),('matchValidOut','valid')]:np.testing.assert_array_equal(actual[name].reshape(1,2),expected[key].numpy())


def test_gpu_all_invalid_enabled_masks_and_output_tails(gpu_runner):
    from sintergpu.features import binary_match_plan
    a,b,xy,cp,av,bv=match_case();av[:]=False;bv[:]=False
    plan=binary_match_plan(2,17,19,35,radius_px=3,max_hamming_distance=35,chunk_size=7,use_landmark_valid=True,use_candidate_valid=True,**gpu_runner.limits)
    inputs=dict(landmarkDescriptor=a.astype(np.uint32),candidateDescriptor=b.astype(np.uint32),projected=xy,candidateXY=cp,landmarkValid=av.astype(np.uint32),candidateValid=bv.astype(np.uint32))
    for name in ('matchedIndexOut','distanceOut','matchValidOut','pairDistance'):
        inputs[name]=np.full(plan['elements'][name]+5,-77 if plan['dtypes'][name]=='int64' else 0xdeadbeef,np.dtype(plan['dtypes'][name]))
    p=gpu_runner.prepare(plan,inputs);gpu_runner.execute(p);actual=gpu_runner.read(p,('matchedIndexOut','distanceOut','matchValidOut','pairDistance'))
    for name,value in [('matchedIndexOut',-1),('distanceOut',0),('matchValidOut',0)]:np.testing.assert_array_equal(actual[name][:34],value)
    for name in actual:np.testing.assert_array_equal(actual[name][-5:],inputs[name][-5:])
