"""Frozen-donor differential contracts for geometry and depth fusion ports."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def numeric():
    path=ROOT/'src/sintergpu/references/geometry_depth_numeric.py'
    assert path.is_file(), 'independent geometry/depth oracle has not been implemented'
    spec=importlib.util.spec_from_file_location('geometry_depth_numeric',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def donor(name):
    pytest.importorskip('torch')
    path=ROOT/'references/donors/vkl/src/vkl/kernels/reference'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('frozen_vkl_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def compare(actual, expected, *, rtol=3e-6, atol=3e-7):
    assert actual.keys()==expected.keys()
    for key, value in expected.items():
        if hasattr(value,'numpy'): value=value.numpy()
        if isinstance(value,str): assert actual[key]==value
        elif value.dtype.kind in 'bui': np.testing.assert_array_equal(actual[key],value,err_msg=key)
        else: np.testing.assert_allclose(actual[key],value,rtol=rtol,atol=atol,equal_nan=True,err_msg=key)


def geometry_case(shape=(2,1,17,19)):
    b,_,h,w=shape
    rng=np.random.default_rng(1)
    depth=np.ones(shape,np.float32)
    depth+=rng.uniform(-.03,.03,shape).astype(np.float32)
    depth.reshape(-1)[:min(6,depth.size)]=[0,-1,np.nan,np.inf,1.25,1][:min(6,depth.size)]
    depth[...,w//2:]=2
    intr=np.tile(np.array([[21,0,w/2],[0,18,h/2],[0,0,1]],np.float32),(b,1,1))
    pose=np.tile(np.eye(4,dtype=np.float32),(b,1,1))
    pose[:,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    pose[:,:3,3]=[2,-3,.25]
    confidence=rng.random(shape,dtype=np.float32)
    confidence.reshape(-1)[0]=-0.0
    confidence.reshape(-1)[-1]=np.nan
    if confidence.size>=3:confidence.reshape(-1)[-3:-1]=[-np.inf,np.inf]
    return depth,intr,pose,confidence


def fusion_case(sources=3, count=269):
    rng=np.random.default_rng(11)
    depth=rng.uniform(.5,2,(sources,2,1,1,count)).astype(np.float32)
    depth[:,:,:,:,:9]=1
    if sources>1:
        depth[1,:,:,:,:3]=[1.3,1.02,np.nan]
    depth.reshape(-1)[7:10]=[np.inf,0,-1]
    valid=rng.random(depth.shape)>.15
    valid[:,:,:,:,:6]=True
    valid[:,:,:,:,6]=False
    confidence=rng.random(depth.shape,dtype=np.float32)
    confidence[:,:,:,:,:6]=.5
    confidence[...,8]=0
    return depth,valid,confidence


@pytest.mark.parametrize('shape',[(2,1,17,19),(1,1,1,1),(1,1,1,7),(1,1,7,1)])
@pytest.mark.parametrize('optional',[False,True])
def test_numpy_geometry_against_frozen_donor(shape,optional):
    torch = pytest.importorskip("torch")
    args=geometry_case(shape)
    expected=donor('geometry').depth_geometry(torch.from_numpy(args[0]),torch.from_numpy(args[1]),
            torch.from_numpy(args[2]) if optional else None,torch.from_numpy(args[3]) if optional else None)
    actual=numeric().depth_geometry(args[0],args[1],args[2] if optional else None,args[3] if optional else None)
    compare(actual,expected)


def test_numpy_world_transform_does_not_remask_zero_or_nonfinite_points():
    torch = pytest.importorskip("torch")
    xyz=np.array([0,1,np.nan,0,2,3,0,4,5],np.float32).reshape(1,3,1,3)
    pose=np.eye(4,dtype=np.float32)[None];pose[0,:3,3]=[7,8,9]
    expected=donor('geometry').world_transform(torch.from_numpy(xyz),torch.from_numpy(pose))
    compare(numeric().world_transform(xyz,pose),expected)


@pytest.mark.parametrize('sources',[1,2,3,9])
@pytest.mark.parametrize('relative',[False,True])
def test_numpy_depth_fusion_against_frozen_donor(sources,relative):
    torch = pytest.importorskip("torch")
    d,v,c=fusion_case(sources)
    statuses=['RELATIVE' if relative else 'METRIC_ESTIMATED']*sources
    expected=donor('depth_fuse').fuse_depth([torch.from_numpy(x) for x in d],[torch.from_numpy(x) for x in v],[torch.from_numpy(x) for x in c],metric_status=statuses,consistency_threshold=.25)
    compare(numeric().fuse_depth(d,v,c,relative=relative,consistency_threshold=.25),expected)


def temporal_case(sources=4):
    d,v,c=fusion_case(sources)
    rng=np.random.default_rng(42)
    visible=rng.random(d.shape)>.1;consistent=rng.random(d.shape)>.1
    trusted=rng.random(d.shape)>.8
    # Visibility/consistency never apply to target source zero.
    visible[0]=False;consistent[0]=False
    trusted[...,1]=False
    return d,v,c,visible,consistent,trusted


def torch_temporal(args, min_support=2, min_temporal=1):
    torch = pytest.importorskip("torch")
    d,v,c,vis,con,trust=map(torch.from_numpy,args)
    return donor('temporal_depth_fuse').temporal_depth_fuse(d[0],list(d[1:]),v[0],list(v[1:]),c[0],list(c[1:]),
        visible=list(vis[1:]),consistent=list(con[1:]),trusted=list(trust[1:]),target_trusted=trust[0],
        consistency_threshold=.25,min_support=min_support,min_temporal_support=min_temporal)


@pytest.mark.parametrize('sources',[2,4,9])
@pytest.mark.parametrize('min_support,min_temporal',[(1,0),(2,1),(4,2)])
def test_numpy_temporal_fusion_full_outputs(sources,min_support,min_temporal):
    args=temporal_case(sources)
    expected=torch_temporal(args,min_support,min_temporal)
    compare(numeric().temporal_depth_fuse(*args,consistency_threshold=.25,min_support=min_support,min_temporal_support=min_temporal),expected)


@pytest.mark.parametrize('module,entry',[('geometry','depthGeometry'),('geometry','worldTransform'),('depth_fusion','fuseDepth'),('depth_fusion','temporalDepthFuse')])
def test_dispatchable_ports_exist(module,entry):
    shader=ROOT/f'src/sintergpu/kernels/vkl/{module}.slang'
    assert shader.is_file(),f'{module} Slang port missing'
    assert f'void {entry}(' in shader.read_text()


@pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS')!='1',reason='explicit GPU qualification only')
class TestGPU:
    @pytest.fixture(scope='class')
    def device(self):
        import slangpy as spy
        return spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})

    def dispatch(self,device,module,entry,count,params,inputs,expected):
        import slangpy as spy
        path=ROOT/f'src/sintergpu/kernels/vkl/{module}.slang'
        assert path.is_file(),f'{module} Slang port missing'
        mod=device.load_module(str(path));kernel=device.create_compute_kernel(device.link_program([mod],[mod.entry_point(entry)]))
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        bindings={'params':params};out={}
        for key,arr in inputs.items():
            arr=np.ascontiguousarray(arr).reshape(-1)
            bindings[key]=device.create_buffer(element_count=arr.size,struct_size=4,usage=usage,data=arr)
        for key,arr in expected.items():
            dtype=np.uint32 if arr.dtype.kind=='b' else np.int32 if arr.dtype.kind in 'iu' else np.float32
            # Integer outputs explicitly use 32-bit typed storage; values preserve donor int64 semantics.
            padded=np.full(arr.size+13,73,dtype=dtype)
            out[key]=(device.create_buffer(element_count=padded.size,struct_size=4,usage=usage,data=padded),dtype,arr.shape)
            bindings[key]=out[key][0]
        kernel.dispatch([count,1,1],vars=bindings)
        result={}
        for key,(buffer,dtype,shape) in out.items():
            a=buffer.to_numpy().view(dtype)
            assert np.all(a[np.prod(shape):]==73),key+' output capacity overwritten'
            result[key]=a[:np.prod(shape)].reshape(shape)
        compare(result,expected,rtol=8e-6,atol=5e-7)
        return result

    @pytest.mark.parametrize('optional',[False,True])
    @pytest.mark.parametrize('shape',[(2,1,17,19),(1,1,1,1)])
    def test_geometry(self,device,optional,shape):
        torch = pytest.importorskip("torch")
        depth,intr,pose,conf=geometry_case(shape);b,_,h,w=shape
        ref=donor('geometry').depth_geometry(torch.from_numpy(depth),torch.from_numpy(intr),torch.from_numpy(pose) if optional else None,torch.from_numpy(conf) if optional else None)
        expected={k+'Out':v.numpy() for k,v in ref.items()}
        expected['xyz_worldOut']=ref['xyz_world'].numpy() if optional else np.full((b,3,h,w),73,np.float32)
        result=self.dispatch(device,'geometry','depthGeometry',depth.size,
            dict(batches=b,height=h,width=w,useWorld=int(optional),useConfidence=int(optional)),
            dict(depth=depth,intrinsics=intr,pose=pose,confidence=conf),expected)
        if optional:
            finite=np.isfinite(expected['geometry_confidenceOut'])
            np.testing.assert_array_equal(np.signbit(result['geometry_confidenceOut'][finite]),np.signbit(expected['geometry_confidenceOut'][finite]))

    def test_world_transform(self,device):
        torch = pytest.importorskip("torch")
        xyz=np.random.default_rng(6).normal(size=(2,3,17,19)).astype(np.float32)
        xyz[:,:,0,0]=0;xyz[0,0,0,1]=np.nan
        _,_,pose,_=geometry_case()
        ref=donor('geometry').world_transform(torch.from_numpy(xyz),torch.from_numpy(pose))
        self.dispatch(device,'geometry','worldTransform',2*17*19,dict(batches=2,height=17,width=19,useWorld=0,useConfidence=0),dict(xyz=xyz,pose=pose),{'xyz_worldOut':ref['xyz_world'].numpy()})

    @pytest.mark.parametrize('sources',[1,2,3,9])
    @pytest.mark.parametrize('relative',[False,True])
    def test_fuse_depth(self,device,sources,relative):
        torch = pytest.importorskip("torch")
        d,v,c=fusion_case(sources)
        ref=donor('depth_fuse').fuse_depth([torch.from_numpy(x) for x in d],[torch.from_numpy(x) for x in v],[torch.from_numpy(x) for x in c],metric_status=['RELATIVE' if relative else 'METRIC_ANCHORED']*sources,consistency_threshold=.25)
        expected={k+'Out':x.numpy() for k,x in ref.items() if k!='metric_status'}
        self.dispatch(device,'depth_fusion','fuseDepth',d[0].size,
            dict(count=d[0].size,sources=sources,relative=int(relative),minSupport=0,minTemporalSupport=0,threshold=.25),
            dict(depths=d,valids=v.astype(np.uint32)*7,confidences=c),expected)

    @pytest.mark.parametrize('sources',[2,4,9])
    @pytest.mark.parametrize('min_support,min_temporal',[(1,0),(2,1),(4,2)])
    def test_temporal(self,device,sources,min_support,min_temporal):
        args=temporal_case(sources)
        ref=torch_temporal(args,min_support,min_temporal)
        expected={k+'Out':v.numpy() for k,v in ref.items()}
        d,v,c,vis,con,trust=args
        self.dispatch(device,'depth_fusion','temporalDepthFuse',d[0].size,
            dict(count=d[0].size,sources=sources,relative=0,minSupport=min_support,minTemporalSupport=min_temporal,threshold=.25),
            dict(depths=d,valids=v.astype(np.uint32),confidences=c,visibility=vis.astype(np.uint32),consistency=con.astype(np.uint32),trusted=trust.astype(np.uint32)),expected)

    def test_borrowed_encoder_fusion_geometry_pipeline(self,device):
        torch = pytest.importorskip("torch")
        import slangpy as spy
        from sintergpu.adapters.slangpy import SlangPyAdapter
        from sintergpu.vkl import geometry_layout,fusion_layout,validate_capacities
        d,v,c=fusion_case(3,17*19)
        d=d.reshape(3,2,1,17,19);v=v.reshape(d.shape);c=c.reshape(d.shape)
        _,intr,pose,_=geometry_case()
        fused=donor('depth_fuse').fuse_depth([torch.from_numpy(x) for x in d],[torch.from_numpy(x) for x in v],[torch.from_numpy(x) for x in c])
        expected=donor('geometry').depth_geometry(fused['depth'],torch.from_numpy(intr),torch.from_numpy(pose),fused['confidence'])
        limit=int(device.info.limits.max_compute_dispatch_thread_groups[0])
        fl=fusion_layout(2,17,19,3,max_workgroups_x=limit)
        gl=geometry_layout(2,17,19,world=True,confidence=True,max_workgroups_x=limit)
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        def buffer(array):
            array=np.ascontiguousarray(array).reshape(-1)
            return device.create_buffer(element_count=array.size,struct_size=4,usage=usage,data=array)
        fb={}
        inputs=dict(depths=d,valids=v.astype(np.uint32),confidences=c)
        for name,size in fl['elements'].items():
            fb[name]=buffer(inputs[name] if name in inputs else np.zeros(size,np.int32 if name=='winning_sourceOut' else np.uint32 if name=='validOut' else np.float32))
        gb=dict(depth=fb['depthOut'],confidence=fb['confidenceOut'],intrinsics=buffer(intr),pose=buffer(pose))
        for name,size in gl['elements'].items():
            if name not in gb:gb[name]=buffer(np.zeros(size,np.uint32 if name=='validOut' else np.float32))
        for layout,buffers in [(fl,fb),(gl,gb)]:validate_capacities(layout,{k:b.size//4 for k,b in buffers.items()})
        adapter=SlangPyAdapter(device,allow_experimental=True)
        ids=['vkl.depth_fusion.fuseDepth','vkl.geometry.depthGeometry']
        for op in ids:adapter.prepare(op)
        encoder=device.create_command_encoder()
        for op,layout,buffers in [(ids[0],fl,fb),(ids[1],gl,gb)]:
            adapter.record(op,thread_count=layout['thread_count'],bindings=dict(params=layout['params'],**buffers),command_encoder=encoder)
        device.submit_command_buffer(encoder.finish());device.wait_for_idle()
        actual={key:gb[key+'Out'].to_numpy().view(np.uint32 if key=='valid' else np.float32).reshape(tuple(value.shape)) for key,value in expected.items()}
        compare(actual,expected,rtol=8e-6,atol=5e-7)

    @pytest.mark.parametrize('constant',[.5,1.,2.,4.])
    def test_fusion_constant_depth_is_an_exact_fixed_point(self,device,constant):
        torch = pytest.importorskip("torch")
        d=np.full((3,2,1,1,269),constant,np.float32)
        v=np.ones(d.shape,bool);c=np.random.default_rng(201).uniform(.125,1,d.shape).astype(np.float32)
        ref=donor('depth_fuse').fuse_depth([torch.from_numpy(x) for x in d],[torch.from_numpy(x) for x in v],[torch.from_numpy(x) for x in c])
        expected={k+'Out':x.numpy() for k,x in ref.items() if k!='metric_status'}
        result=self.dispatch(device,'depth_fusion','fuseDepth',d[0].size,
            dict(count=d[0].size,sources=3,relative=0,minSupport=2,minTemporalSupport=1,threshold=.25),
            dict(depths=d,valids=v.astype(np.uint32),confidences=c),expected)
        np.testing.assert_array_equal(expected['depthOut'],np.float32(constant))
        np.testing.assert_array_equal(result['depthOut'],expected['depthOut'])
