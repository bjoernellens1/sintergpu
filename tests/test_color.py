"""Independent camera-format/sampling contracts; GPU execution is opt-in."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1]/'src/sintergpu/references/color.py'
spec=importlib.util.spec_from_file_location('color',MODULE)
c=importlib.util.module_from_spec(spec)
if MODULE.exists(): spec.loader.exec_module(c)


def test_video_black_white_padded_and_preserved_padding():
    assert hasattr(c,'nv12_to_rgb'), 'portable NV12 decode is missing'
    y=np.array([16,235,77,77,235,16,77,77],dtype=np.uint8)
    uv=np.array([128,128,77,77],dtype=np.uint8)
    output=np.full(16,99,np.uint8)
    actual=c.nv12_to_rgb(y,uv,width=2,height=2,y_pitch=4,uv_pitch=4,rgb_pitch=8,output=output)
    np.testing.assert_array_equal(actual.reshape(2,8)[:,:6],[[0,0,0,255,255,255],[255,255,255,0,0,0]])
    assert np.all(actual.reshape(2,8)[:,6:]==99)


@pytest.mark.parametrize('matrix',['bt601','bt709'])
@pytest.mark.parametrize('range_',['limited','full'])
def test_rgb_box_encode_known_gray(matrix,range_):
    assert hasattr(c,'rgb_to_nv12'), 'centered NV12 encode is missing'
    tags=c.ColorSpec(matrix,range_)
    rgb=np.full(12,255,np.uint8)
    y,uv=c.rgb_to_nv12(rgb,width=2,height=2,tags=tags)
    assert np.all(y[:4]==(235 if range_=='limited' else 255))
    np.testing.assert_array_equal(uv[:2],[128,128])


def test_matrix_is_not_hardcoded_and_yuyv_pair_layout():
    assert hasattr(c,'yuyv_to_rgb'), 'packed YUYV decode is missing'
    data=np.array([90,20,140,220],np.uint8)
    a=c.yuyv_to_rgb(data,width=2,height=1,tags=c.ColorSpec('bt601','full'))
    b=c.yuyv_to_rgb(data,width=2,height=1,tags=c.ColorSpec('bt709','full'))
    assert not np.array_equal(a,b)
    assert not np.array_equal(a[:3],a[3:6])


def test_remap_borders_nonfinite_odd_output_and_round_ties():
    assert hasattr(c,'remap_rgb'), 'bilinear packed RGB remap is missing'
    rgb=np.array([0,1,2,1,2,3,99,99],np.uint8)
    maps=np.array([[[-1,0],[.5,0],[9,0]],[[np.nan,0],[0,np.inf],[1,0]],[[0,0],[1,0],[0,0]]],np.float32)
    out=c.remap_rgb(rgb,maps,width=2,height=1)
    pixels=out[:27].reshape(3,3,3)
    np.testing.assert_array_equal(pixels[0],[[0,1,2],[0,2,2],[1,2,3]])
    np.testing.assert_array_equal(pixels[1,:2],0)


def test_fusion_rounds_decode_before_interpolation():
    assert hasattr(c,'fused_yuyv_remap'), 'fused camera remap is missing'
    packed=np.random.default_rng(9).integers(0,256,8*6,dtype=np.uint8)
    maps=np.random.default_rng(10).uniform(-1,7,(5,7,2)).astype(np.float32)
    tags=c.ColorSpec('bt709','limited')
    decoded=c.yuyv_to_rgb(packed,width=8,height=3,tags=tags)
    separate=c.remap_rgb(decoded,maps,width=8,height=3)
    fused=c.fused_yuyv_remap(packed,maps,width=8,height=3,tags=tags)
    np.testing.assert_array_equal(fused,separate)


@pytest.mark.parametrize('operation,kwargs',[
    ('nv12_decode',dict(width=3,height=2)),('nv12_decode',dict(width=2,height=3)),
    ('yuyv_decode',dict(width=3,height=2)),('yuyv_decode',dict(width=2,height=2,source_pitch=3)),
    ('fused_remap',dict(width=2,height=2)),('yuyv_decode',dict(width=2,height=2,source_capacity=7)),
])
def test_host_validation_rejects_invalid_dispatch(operation,kwargs):
    assert hasattr(c,'validate_request'), 'dispatch validation is missing'
    with pytest.raises(ValueError): c.validate_request(operation,**kwargs)


def test_exact_hip_profile_is_not_silently_substituted():
    assert hasattr(c,'ColorSpec'), 'explicit numerical profiles are missing'
    with pytest.raises(ValueError,match='HIP'):
        c.ColorSpec(profile='hip-exact')


def test_known_bt601_limited_red_and_centered_box_chroma():
    data=np.array([81,90,81,240],np.uint8)
    rgb=c.yuyv_to_rgb(data,width=2,height=1)
    np.testing.assert_array_equal(rgb[:6].reshape(2,3),[[254,0,0],[254,0,0]])
    pixels=np.array([255,0,0, 0,0,255, 0,0,255, 0,0,255],np.uint8)
    _,uv=c.rgb_to_nv12(pixels,width=2,height=2,tags=c.ColorSpec('bt601','full'))
    top_only=(np.array([255,0,0]*4,np.uint8))
    _,uv_top=c.rgb_to_nv12(top_only,width=2,height=2,tags=c.ColorSpec('bt601','full'))
    assert not np.array_equal(uv[:2],uv_top[:2])
    np.testing.assert_array_equal(uv[:2],[213,144])


def test_nv12_nearest_oracle_matches_retained_independent_numpy():
    import sys
    path=Path(__file__).resolve().parents[1]/'references/donors/eligsir-sim/src/eligsir_sim/video/colour.py'
    donor_spec=importlib.util.spec_from_file_location('retained_colour',path)
    donor=importlib.util.module_from_spec(donor_spec); sys.modules['retained_colour']=donor; donor_spec.loader.exec_module(donor)
    rng=np.random.default_rng(33); y=rng.integers(0,256,(4,6),dtype=np.uint8); uv=rng.integers(0,256,(2,3,2),dtype=np.uint8)
    for matrix in ('bt601','bt709'):
        for range_ in ('limited','full'):
            expected=donor.nv12_to_rgb_numpy(y,uv,donor.ColourTags(matrix=matrix,range=range_),out_dtype='uint8',chroma_upsample='nearest')
            actual=c.nv12_to_rgb(y.reshape(-1),uv.reshape(-1),width=6,height=4,tags=c.ColorSpec(matrix,range_))
            np.testing.assert_array_equal(actual[:72].reshape(4,6,3),expected)


def load_benchmark():
    path=Path(__file__).resolve().parents[1]/'benchmarks/color_pipeline.py'
    spec=importlib.util.spec_from_file_location('color_pipeline_benchmark',path)
    benchmark=importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)
    return benchmark


def test_fusion_gate_requires_gain_ci_and_tail_preservation():
    b=load_benchmark()
    assert b.fusion_gate(np.ones(100),np.full(100,.9))['passed']
    assert not b.fusion_gate(np.ones(100),np.full(100,.96))['passed']
    bad_tail=np.full(100,.8); bad_tail[90:]=1.2
    assert not b.fusion_gate(np.ones(100),bad_tail)['passed']


@pytest.mark.skipif(__import__('os').environ.get('SINTERGPU_GPU_TESTS')!='1',reason='GPU opt-in plus independent performance clearance required')
def test_gpu_camera_profiles_and_fused_exactness():
    b=load_benchmark(); runner=b.GPURunner()
    rng=np.random.default_rng(103)
    width,height=6,4; source_pitch=15; rgb_pitch=23
    packed=rng.integers(0,256,60,dtype=np.uint8)
    maps=rng.uniform(-2,8,(5,7,2)).astype(np.float32)
    maps[0,0]=[np.nan,0]; maps[0,1]=[0,np.inf]; maps[0,2]=[1e30,-1e30]
    seed=np.full(116,173,np.uint8)
    def check(actual,expected,pitch,w,h,initial):
        live=np.zeros(actual.size,bool)
        for row in range(h): live[row*pitch:row*pitch+w*3]=True
        assert np.max(np.abs(actual[live].astype(int)-expected[live].astype(int)))<=1
        np.testing.assert_array_equal(actual[~live],initial[~live])
    for matrix in ('bt601','bt709'):
        for range_ in ('limited','full'):
            tags=c.ColorSpec(matrix,range_)
            p=c.validate_request('yuyv_decode',width=width,height=height,source_pitch=source_pitch,output_pitch=rgb_pitch,source_capacity=60,output_capacity=92,tags=tags)
            decoded=runner.execute('yuyvDecode',packed,p)
            expected=c.yuyv_to_rgb(packed,width=width,height=height,source_pitch=source_pitch,rgb_pitch=rgb_pitch,tags=tags,output=np.full(92,173,np.uint8))
            check(decoded,expected,rgb_pitch,width,height,np.full(92,173,np.uint8))
            remap_p=c.validate_request('remap',width=width,height=height,output_width=7,output_height=5,source_pitch=rgb_pitch,output_pitch=23,source_capacity=92,output_capacity=116,calibration=maps,tags=tags)
            separate=runner.execute('remapBilinear',decoded,remap_p,maps=maps,output_seed=seed)
            fused_p=c.validate_request('fused_remap',width=width,height=height,output_width=7,output_height=5,source_pitch=source_pitch,output_pitch=23,source_capacity=60,output_capacity=116,calibration=maps,tags=tags)
            fused=runner.execute('fusedYUYVRemap',packed,fused_p,maps=maps,output_seed=seed)
            np.testing.assert_array_equal(separate,fused)
            oracle=c.fused_yuyv_remap(packed,maps,width=width,height=height,source_pitch=source_pitch,rgb_pitch=23,tags=tags,output=seed.copy())
            check(fused,oracle,23,7,5,seed)
            y=rng.integers(0,256,32,dtype=np.uint8); uv=rng.integers(0,256,16,dtype=np.uint8)
            p=c.validate_request('nv12_decode',width=width,height=height,source_pitch=8,uv_pitch=8,output_pitch=23,source_capacity=32,uv_capacity=16,output_capacity=92,tags=tags)
            decoded_nv12=runner.execute('nv12Decode',y,p,uv=uv)
            expected_nv12=c.nv12_to_rgb(y,uv,width=width,height=height,y_pitch=8,uv_pitch=8,rgb_pitch=23,tags=tags,output=np.full(92,173,np.uint8))
            check(decoded_nv12,expected_nv12,23,width,height,np.full(92,173,np.uint8))
            p=c.validate_request('nv12_encode',width=width,height=height,source_pitch=23,output_pitch=9,output_uv_pitch=9,source_capacity=92,output_capacity=36,output_uv_capacity=20,tags=tags)
            actual_y,actual_uv=runner.execute('nv12Encode',decoded,p)
            expected_y,expected_uv=c.rgb_to_nv12(decoded,width=width,height=height,source_pitch=23,y_pitch=9,uv_pitch=9,tags=tags,y_output=np.full(36,173,np.uint8),uv_output=np.full(20,181,np.uint8))
            for actual,expected,rows in ((actual_y,expected_y,4),(actual_uv,expected_uv,2)):
                live=np.zeros(len(actual),bool)
                for row in range(rows): live[row*9:row*9+width]=True
                assert np.max(np.abs(actual[live].astype(int)-expected[live].astype(int)))<=1
                np.testing.assert_array_equal(actual[~live],expected[~live])
    # Odd resize output; one-pixel input axes and half-pixel border rules.
    source=np.array([0,1,2,1,2,3,99,99],np.uint8)
    p=c.validate_request('resize',width=2,height=1,output_width=3,output_height=3,source_capacity=8,output_capacity=28)
    actual=runner.execute('resizeBilinear',source,p)
    expected=c.resize_rgb(source,width=2,height=1,output_width=3,output_height=3,output=np.full(28,173,np.uint8))
    check(actual,expected,9,3,3,np.full(28,173,np.uint8))


def test_decode_rejects_output_extent_smaller_than_camera():
    with pytest.raises(ValueError,match='extent'):
        c.validate_request('yuyv_decode',width=6,height=4,output_width=2,output_height=2)


def test_dispatch_validation_requires_all_borrowed_capacities():
    with pytest.raises(ValueError,match='capacities'):
        c.validate_request('yuyv_decode',width=6,height=4,for_dispatch=True)
