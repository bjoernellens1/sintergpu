"""Independent resampling/provenance comparisons with frozen VKL Torch code."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def frozen():
    pytest.importorskip('torch')
    name='_sintergpu_frozen_vkl_resampling'
    if name not in sys.modules:
        p=types.ModuleType(name);p.__path__=[str(ROOT/'references/donors/vkl/src/vkl/kernels/reference')];sys.modules[name]=p
    return importlib.import_module(name+'.depth_densify')


def oracle():
    p=ROOT/'src/sintergpu/references/depth_resampling_numeric.py'
    assert p.is_file(),'independent resampling oracle missing'
    s=importlib.util.spec_from_file_location('depth_resampling_numeric',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def data(channels=3):
    rng=np.random.default_rng(571)
    depth=rng.uniform(.8,1.2,(2,1,5,7)).astype(np.float32)
    valid=rng.random(depth.shape)>.4
    depth.reshape(-1)[:4]=[0,-1,np.nan,np.inf];valid.reshape(-1)[:4]=True
    conf=rng.random(depth.shape,dtype=np.float32);conf.reshape(-1)[4]=0
    rgb=rng.random((2,channels,5,7),dtype=np.float32)
    rgb[0,:,2,3]=np.nan
    return depth,valid,conf,rgb


def compare(actual,expected):
    assert actual.keys()==expected.keys()
    for key,e in expected.items():
        e=e.numpy()
        if e.dtype.kind in 'bui':np.testing.assert_array_equal(actual[key],e,err_msg=key)
        else:np.testing.assert_allclose(actual[key],e,rtol=8e-6,atol=5e-7,equal_nan=True,err_msg=key)


@pytest.mark.parametrize('size',[(1,1),(17,19),(31,23),(7,3),(5,7)])
def test_numpy_nearest_lift_preserves_propagated_provenance(size):
    torch=pytest.importorskip('torch');d,v,c,_=data()
    expected=frozen().depth_upsample(torch.from_numpy(d),torch.from_numpy(v),torch.from_numpy(c),size=size)
    compare(oracle().depth_upsample(d,v,c,size=size),expected)


@pytest.mark.parametrize('channels',[1,3])
@pytest.mark.parametrize('steps',[1,2,4])
@pytest.mark.parametrize('direct',[False,True])
def test_numpy_densify_all_outputs_and_initial_direct_mask(channels,steps,direct):
    torch=pytest.importorskip('torch');d,v,c,rgb=data(channels)
    direct_mask=(np.indices(d.shape).sum(0)%3==0) if direct else None
    kwargs=dict(max_rgb_difference=.75,max_relative_spread=.6,min_support=1,steps=steps)
    expected=frozen().depth_densify(torch.from_numpy(d),torch.from_numpy(v),torch.from_numpy(c),torch.from_numpy(rgb),direct_valid=torch.from_numpy(direct_mask) if direct else None,**kwargs)
    compare(oracle().depth_densify(d,v,c,rgb,direct_valid=direct_mask,**kwargs),expected)


def test_flat_hole_keeps_direct_and_propagated_labels_distinct():
    d=np.ones((1,1,3,3),np.float32);d[0,0,1,1]=np.nan
    v=np.ones(d.shape,bool);c=np.ones(d.shape,np.float32);rgb=np.zeros((1,3,3,3),np.float32)
    actual=oracle().depth_densify(d,v,c,rgb,steps=2)
    assert actual['direct_valid'][0,0,1,1]==False
    assert actual['propagated_valid'][0,0,1,1] and actual['provenance'][0,0,1,1]==2
    np.testing.assert_array_equal(actual['depth'],1)
    assert actual['support'][0,0,1,1]==9
    np.testing.assert_allclose(actual['confidence'][0,0,1,1],np.float32(8/9))


@pytest.mark.parametrize('scale',[2,4,8])
def test_downsample_reuses_pyramid_semantics_and_direct_labels(scale):
    torch=pytest.importorskip('torch');d,v,c,_=data()
    expected=frozen().depth_downsample(torch.from_numpy(d),torch.from_numpy(v),torch.from_numpy(c),scale=scale)
    compare(oracle().depth_downsample(d,v,c,scale=scale),expected)


def test_downsample_rejects_unavailable_pyramid_level():
    d=np.ones((1,1,1,1),np.float32);v=np.ones(d.shape,bool)
    with pytest.raises(ValueError,match='available'):oracle().depth_downsample(d,v,d,scale=2)


def test_resampling_layouts_and_multistep_plan_are_resource_independent():
    from sintergpu.vkl import depth_lift_layout,depth_densify_plan,depth_pyramid_plan
    lift=depth_lift_layout(2,5,7,17,19,max_workgroups_x=65535)
    assert lift['thread_count']==(2*17*19,1,1)
    assert lift['elements']['depth']==70 and lift['elements']['depthOut']==646
    stages=depth_densify_plan(2,5,7,channels=3,steps=4,direct=True,max_workgroups_x=65535)
    assert len(stages)==4 and [x['params']['firstStep'] for x in stages]==[1,0,0,0]
    assert all(x['params']['useDirect']==1 for x in stages)
    assert stages[0]['elements']['rgb']==210
    for channels in [0,2,4,True]:
        with pytest.raises(ValueError):depth_densify_plan(1,5,7,channels=channels,max_workgroups_x=65535)
    default=depth_densify_plan(1,5,7,steps=2,max_workgroups_x=65535)
    assert default[0]['elements']['direct_valid']==1 and default[1]['elements']['direct_valid']==35
    assert [x['params']['useDirect'] for x in default]==[0,1]
    pyramid=depth_pyramid_plan(2,5,7,max_workgroups_x=65535)
    assert [x['level'] for x in pyramid]==[1,2,4,8]
    assert [x['shape'] for x in pyramid]==[(2,1,5,7),(2,1,3,4),(2,1,2,2),(2,1,1,1)]
    assert depth_pyramid_plan(1,1,1,max_workgroups_x=65535)[0]['level']==1
    with pytest.raises(ValueError,match='available'):depth_pyramid_plan(1,1,1,scale=2,max_workgroups_x=65535)
    for f,args in [(depth_lift_layout,(2,5,7,17,19)),(depth_densify_plan,(2,17,19)),(depth_pyramid_plan,(2,17,19))]:
        with pytest.raises(ValueError):f(*args,max_workgroups_x=1)
    with pytest.raises(ValueError):depth_densify_plan(2,2**30,1,channels=3,max_workgroups_x=2**32-1)
    with pytest.raises(ValueError):depth_lift_layout(1,1,2**24+1,1,1,max_workgroups_x=65535)


class TestDepthResamplingGPU:
    @pytest.fixture(scope='class')
    def device(self):
        import os
        if os.environ.get('SINTERGPU_GPU_TESTS')!='1':pytest.skip('explicit GPU opt-in required')
        spy=pytest.importorskip('slangpy')
        d=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        yield d
        d.close()

    @staticmethod
    def buffer(device,a):
        import slangpy as spy
        a=np.ascontiguousarray(a).reshape(-1)
        return device.create_buffer(element_count=max(1,a.size),struct_size=4,
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access,data=a if a.size else np.zeros(1,np.float32))

    @staticmethod
    def kernel(device,entry,filename='depth_resampling'):
        p=ROOT/f'src/sintergpu/kernels/vkl/{filename}.slang'
        module=device.load_module(str(p));return device.create_compute_kernel(device.link_program([module],[module.entry_point(entry)]))

    def outputs(self,device,shape,names):
        count=int(np.prod(shape));out={}
        for name in names:
            dtype=np.float32 if name in ('depth','confidence') else np.int32 if name=='support' else np.uint32
            out[name]=(self.buffer(device,np.full(count+13,73,dtype)),dtype)
        return out

    @staticmethod
    def read(out,shape):
        count=int(np.prod(shape));result={}
        for name,(buffer,dtype) in out.items():
            a=buffer.to_numpy().view(dtype);assert np.all(a[count:]==73),name+' padding overwritten'
            result[name]=a[:count].reshape(shape)
        return result

    @staticmethod
    def close(actual,expected):
        for name,a in expected.items():
            if a.dtype.kind in 'bui':np.testing.assert_array_equal(actual[name],a,err_msg=name)
            else:np.testing.assert_allclose(actual[name],a,rtol=8e-6,atol=5e-7,equal_nan=True,err_msg=name)

    @pytest.mark.parametrize('size',[(1,1),(17,19),(31,23),(7,3),(5,7)])
    def test_nearest_lift(self,device,size):
        from sintergpu.vkl import depth_lift_layout
        d,v,c,_=data();expected=oracle().depth_upsample(d,v,c,size=size)
        plan=depth_lift_layout(2,5,7,*size,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]))
        bindings={k:self.buffer(device,a) for k,a in dict(depth=d,valid=v.astype(np.uint32)*7,confidence=c).items()}
        out=self.outputs(device,expected['depth'].shape,expected)
        self.kernel(device,plan['entrypoint']).dispatch(list(plan['thread_count']),vars={'params':plan['params'],**bindings,**{k+'Out':a[0] for k,a in out.items()}})
        self.close(self.read(out,expected['depth'].shape),expected)

    @pytest.mark.parametrize('limits',[(.75,.6),(float('nan'),.6),(.75,float('nan')),(float('inf'),float('inf'))])
    @pytest.mark.parametrize('channels',[1,3])
    @pytest.mark.parametrize('steps',[1,2,4])
    @pytest.mark.parametrize('direct',[False,True])
    def test_densify_complete_borrowed_encoder_pipeline(self,device,channels,steps,direct,limits):
        from sintergpu.vkl import depth_densify_plan
        d,v,c,rgb=data(channels)
        d,v,c=[np.tile(a,(1,1,4,3)) for a in (d,v,c)];rgb=np.tile(rgb,(1,1,4,3));mask=(np.indices(d.shape).sum(0)%3==0)
        kwargs=dict(max_rgb_difference=limits[0],max_relative_spread=limits[1],min_support=1,steps=steps)
        expected=oracle().depth_densify(d,v,c,rgb,direct_valid=mask if direct else None,**kwargs)
        plans=depth_densify_plan(2,d.shape[-2],d.shape[-1],channels=channels,direct=direct,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]),**kwargs)
        states=[self.outputs(device,d.shape,expected) for _ in range(2)]
        current={k:self.buffer(device,a) for k,a in dict(depth=d,valid=v.astype(np.uint32)*7,confidence=c,direct_valid=mask.astype(np.uint32) if direct else np.zeros(1,np.uint32)).items()}
        rgb_buffer=self.buffer(device,rgb);kernel=self.kernel(device,'densifyDepth');encoder=device.create_command_encoder()
        for n,plan in enumerate(plans):
            out=states[n%2]
            kernel.dispatch(list(plan['thread_count']),vars={'params':plan['params'],**current,'rgb':rgb_buffer,**{k+'Out':a[0] for k,a in out.items()}},command_encoder=encoder)
            current={k:out[k][0] for k in ('depth','valid','confidence','direct_valid')}
        device.submit_command_buffer(encoder.finish())
        self.close(self.read(states[(steps-1)%2],d.shape),expected)

    @pytest.mark.parametrize('scale',[2,4,8])
    def test_downsample_composes_released_level_source(self,device,scale):
        from sintergpu.vkl import depth_downsample_plan
        d,v,c,_=data();expected=oracle().depth_downsample(d,v,c,scale=scale)
        graph=depth_downsample_plan(2,5,7,scale=scale,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]))
        inputs={k:self.buffer(device,a) for k,a in dict(depth=d,valid=v.astype(np.uint32),confidence=c).items()}
        kernels={name:self.kernel(device,name,'depth_pyramid' if name=='depthPyramidLevel' else 'depth_resampling') for name in ['initializeDepth','depthPyramidLevel','labelDirectDepth']}
        encoder=device.create_command_encoder();states=[]
        for plan in graph['stages']:
            names=[k.removesuffix('Out') for k in plan['elements'] if k.endswith('Out')]
            out=self.outputs(device,plan['shape'],names);states.append(out)
            bind=inputs if plan['entrypoint']!='labelDirectDepth' else {'valid':inputs['valid']}
            kernels[plan['entrypoint']].dispatch(list(plan['thread_count']),vars={'params':plan['params'],**bind,**{k+'Out':a[0] for k,a in out.items()}},command_encoder=encoder)
            if plan['entrypoint']!='labelDirectDepth':inputs={k:out[k][0] for k in ('depth','valid','confidence')}
        device.submit_command_buffer(encoder.finish())
        outputs={name:states[ref['stage']][ref['binding'].removesuffix('Out')] for name,ref in graph['outputs'].items()}
        self.close(self.read(outputs,graph['shape']),expected)

    @pytest.mark.parametrize('rgb_limit,spread_limit,filled',[(.5,.25,True),(float(np.nextafter(np.float32(.5),np.float32(0))),.25,False),(.5,float(np.nextafter(np.float32(.25),np.float32(0))),False)])
    def test_inclusive_rgb_and_spread_boundaries(self,device,rgb_limit,spread_limit,filled):
        from sintergpu.vkl import depth_densify_plan
        torch=pytest.importorskip('torch')
        d=np.ones((1,1,3,3),np.float32);d[0,0,0,0]=1.25;d[0,0,1,1]=np.nan
        v=np.ones(d.shape,bool);c=np.ones(d.shape,np.float32);rgb=np.zeros((1,3,3,3),np.float32);rgb[:,:,:,2]=.5
        kwargs=dict(max_rgb_difference=rgb_limit,max_relative_spread=spread_limit,min_support=8,steps=1)
        e=frozen().depth_densify(torch.from_numpy(d),torch.from_numpy(v),torch.from_numpy(c),torch.from_numpy(rgb),**kwargs)
        expected={k:a.numpy() for k,a in e.items()};assert bool(expected['valid'][0,0,1,1])==filled
        plans=depth_densify_plan(1,3,3,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]),**kwargs)
        inputs={k:self.buffer(device,a) for k,a in dict(depth=d,valid=v.astype(np.uint32),confidence=c,rgb=rgb,direct_valid=np.zeros(1,np.uint32)).items()}
        out=self.outputs(device,d.shape,expected);plan=plans[0]
        self.kernel(device,'densifyDepth').dispatch(list(plan['thread_count']),vars={'params':plan['params'],**inputs,**{k+'Out':a[0] for k,a in out.items()}})
        self.close(self.read(out,d.shape),expected)

    @pytest.mark.parametrize('flat',[.5,1.,2.,4.])
    @pytest.mark.parametrize('shape',[(2,1,17,19),(1,1,1,1),(1,1,1,7),(1,1,7,1)])
    @pytest.mark.parametrize('optional',[False,True])
    def test_full_pyramid_all_levels_and_optional_confidence(self,device,shape,optional,flat):
        torch=pytest.importorskip('torch')
        from sintergpu.vkl import depth_pyramid_plan
        rng=np.random.default_rng(34);d=np.full(shape,flat,np.float32);v=rng.random(shape)>.2;c=rng.random(shape,dtype=np.float32)
        if d.size>1:d.reshape(-1)[0]=np.nan
        expected=frozen().depth_pyramid(depth=torch.from_numpy(d),valid=torch.from_numpy(v),confidence=torch.from_numpy(c) if optional else None)
        plans=depth_pyramid_plan(shape[0],shape[-2],shape[-1],confidence=optional,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]))
        inputs={k:self.buffer(device,a) for k,a in dict(depth=d,valid=v.astype(np.uint32)*7,confidence=c if optional else np.zeros(1,np.float32)).items()}
        kernels={name:self.kernel(device,name,'depth_pyramid' if name=='depthPyramidLevel' else 'depth_resampling') for name in ['initializeDepth','depthPyramidLevel']}
        encoder=device.create_command_encoder();states=[]
        for plan in plans:
            out=self.outputs(device,plan['shape'],('depth','valid','confidence'));states.append(out)
            kernels[plan['entrypoint']].dispatch(list(plan['thread_count']),vars={'params':plan['params'],**inputs,**{k+'Out':a[0] for k,a in out.items()}},command_encoder=encoder)
            inputs={k:out[k][0] for k in ('depth','valid','confidence')}
        device.submit_command_buffer(encoder.finish())
        for plan,out in zip(plans,states):
            actual=self.read(out,plan['shape']);e={k:v.numpy() for k,v in expected['/'+str(plan['level'])].items()};self.close(actual,e)
            # Exact flat-depth fixed point across every level, not just tolerance.
            np.testing.assert_array_equal(actual['depth'],e['depth'])


def resampling_guard(text):
    import re
    code=re.sub(r'//[^\n]*','',text)
    assert not re.search(r'\b(while|do|groupshared|Interlocked\w*|GroupMemoryBarrier\w*)\b',code)
    assert len(re.findall(r'\[numthreads\(256,1,1\)\]',code))==4
    assert code.count('if(i>=params.batches*params.height*params.width)return;')==2
    assert code.count('if(i>=params.batches*plane)return;')==2
    assert not re.search(r'for\s*\([^)]*<\s*params\.(batches|height|width|outputHeight|outputWidth)',code)
    dense=code.split('void densifyDepth',1)[1]
    for name in ('depthOut','validOut','confidenceOut','direct_validOut','propagated_validOut','provenanceOut','supportOut'):
        assert len(re.findall(r'\b'+name+r'\[i\]\s*=',dense))==1
    assert re.findall(r'for\s*\(([^)]*)\)',dense)==['uint k=0;k<9;k++','int dy=-1;dy<=1;dy++','int dx=-1;dx<=1;dx++','uint c=0;c<params.channels;c++','uint pass=0;pass<8;pass++','uint k=0;k<8-pass;k++']
    assert 'values[9]' in dense and 'pass<8' in dense and 'k<8-pass' in dense
    assert 'params.firstStep' in dense and 'else direct=direct_valid[i]!=0' in dense


def test_resampling_hot_path_guard_and_mutations():
    p=ROOT/'src/sintergpu/kernels/vkl/depth_resampling.slang'
    assert p.is_file()
    text=p.read_text();resampling_guard(text)
    for old,new in [('supportOut[i]=int(support);',''),('pass<8','pass<params.height'),
                    ('if(i>=params.batches*plane)return;',''),('else direct=direct_valid[i]!=0','else direct=centerValid'),('dy<=1','dy<=2'),('dx<=1','dx<=params.width')]:
        assert old in text
        with pytest.raises(AssertionError):resampling_guard(text.replace(old,new))


@pytest.mark.parametrize('rgb_limit,spread_limit',[(float('nan'),.6),(float('inf'),.6),(.75,float('nan')),(float('inf'),float('inf'))])
def test_nonfinite_thresholds_match_frozen_nonnegative_validation(rgb_limit,spread_limit):
    torch=pytest.importorskip('torch');d,v,c,rgb=data()
    kwargs=dict(max_rgb_difference=rgb_limit,max_relative_spread=spread_limit,min_support=1,steps=2)
    expected=frozen().depth_densify(torch.from_numpy(d),torch.from_numpy(v),torch.from_numpy(c),torch.from_numpy(rgb),**kwargs)
    compare(oracle().depth_densify(d,v,c,rgb,**kwargs),expected)


def test_downsample_plan_exposes_aliases_without_copying_depth_outputs():
    from sintergpu.vkl import depth_downsample_plan
    plan=depth_downsample_plan(2,5,7,scale=4,max_workgroups_x=65535)
    assert plan['shape']==(2,1,2,2) and len(plan['stages'])==4
    assert plan['stages'][-1]['entrypoint']=='labelDirectDepth'
    for name,ref in plan['outputs'].items():
        assert ref['binding'] in plan['stages'][ref['stage']]['elements']
    assert plan['outputs']['depth']==dict(stage=2,binding='depthOut')
    assert plan['outputs']['provenance']==dict(stage=3,binding='provenanceOut')
    assert 'depthOut' not in plan['stages'][-1]['elements']


def test_float32_numpy_thresholds_are_normalized_to_plain_parameters():
    from sintergpu.vkl import depth_densify_plan
    plan=depth_densify_plan(1,5,7,max_workgroups_x=65535,max_rgb_difference=np.float32(.15),max_relative_spread=np.float32(.1))
    assert type(plan[0]['params']['maxRGBDifference']) is float
    assert plan[0]['params']['maxRGBDifference']==float(np.float32(.15))
