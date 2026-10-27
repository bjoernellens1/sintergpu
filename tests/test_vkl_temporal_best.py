"""All numerical outputs of frozen VKL best-observation reduction."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    assert path.is_file(), f'{name} has not been implemented'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def case(seed,neighbors):
    rng=np.random.default_rng(seed)
    quality=rng.uniform(-2,2,(11,neighbors)).astype(np.float32)
    target=rng.uniform(-2,2,11).astype(np.float32)
    equiv=rng.random(quality.shape)>.3
    confidence=rng.random(quality.shape,dtype=np.float32)
    # Frozen max propagates NaN, and any nonfinite winning candidate makes the
    # target win, even when another equivalent candidate is finite.
    equiv[:7]=True
    quality[0]=np.nan;quality[1]=np.inf;quality[2]=-np.inf
    target[3]=np.nan;target[4]=-np.inf;target[5]=np.inf
    quality[6]=.5;target[6]=.5
    equiv[7]=False
    confidence[4]=0
    return equiv,target,quality,confidence


@pytest.mark.parametrize('neighbors',[1,2,7,33])
@pytest.mark.parametrize('seed',[1,17,92])
@pytest.mark.parametrize('optional',[False,True])
def test_numpy_temporal_best_against_frozen(seed,neighbors,optional):
    torch=pytest.importorskip('torch')
    equiv,target,quality,confidence=case(seed,neighbors)
    kwargs={'pairwise_quality':quality*.75,'pairwise_confidence':confidence} if optional else {}
    donor=load(ROOT/'references/donors/vkl/src/vkl/kernels/reference/temporal_best.py','frozen_best')
    expected=donor.temporal_best_observation(torch.from_numpy(equiv),torch.from_numpy(target),torch.from_numpy(quality),**{k:torch.from_numpy(v) for k,v in kwargs.items()})
    oracle=load(ROOT/'src/sintergpu/references/temporal_best_numeric.py','best_numeric')
    actual=oracle.temporal_best_observation(equiv,target,quality,**kwargs)
    assert actual.keys()==expected.keys()
    for k,v in expected.items():
        if k=='best_index':np.testing.assert_array_equal(actual[k],v.numpy())
        else:np.testing.assert_allclose(actual[k],v.numpy(),rtol=3e-6,atol=3e-7,equal_nan=True,err_msg=k)


def test_shader_and_layout_contract():
    from sintergpu.vkl import temporal_best_layout
    layout=temporal_best_layout(269,33,max_workgroups_x=65535,quality=True,confidence=True)
    assert layout['thread_count']==(269,1,1)
    assert layout['elements']['neighbor_quality']==1
    assert layout['elements']['best_indexOut']==269
    assert layout['params']==dict(batches=269,neighbors=33,useQuality=1,useConfidence=1)
    for batches,neighbors,limit in [(2**31,3,2**32-1),(2,2**31,65535),(257,1,1)]:
        with pytest.raises(ValueError):temporal_best_layout(batches,neighbors,max_workgroups_x=limit)
    p=ROOT/'src/sintergpu/kernels/vkl/temporal_best.slang'
    assert p.is_file()
    text=p.read_text()
    assert 'if(i>=params.batches)return;' in text
    assert 'groupshared' not in text and 'Interlocked' not in text
    import re
    code=re.sub(r'//[^\n]*','',text)
    assert re.findall(r'for\s*\(([^)]*)\)',code)==['uint n=0;n<params.neighbors;n++']
    assert not re.search(r'\b(while|do|GroupMemoryBarrier\w*)\b',code)
    for name in ('best_indexOut','best_qualityOut','replaceabilityOut','replacement_confidenceOut'):
        assert len(re.findall(r'\b'+name+r'\[i\]\s*=',code))==1


@pytest.mark.skipif(__import__('os').environ.get('SINTERGPU_GPU_TESTS')!='1',reason='explicit GPU opt-in required')
@pytest.mark.parametrize('neighbors',[1,2,7,33])
@pytest.mark.parametrize('quality_enabled',[False,True])
@pytest.mark.parametrize('confidence_enabled',[False,True])
def test_gpu_temporal_best_full_outputs(neighbors,quality_enabled,confidence_enabled):
    spy=pytest.importorskip('slangpy')
    # Standalone fixture owns allocation/submission; production shader owns none.
    device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        eq,t,q,c=case(92,neighbors)
        # Repeat to cross a256thread group and exercise nonmultiple tails.
        eq=np.tile(eq,(25,1));t=np.tile(t,25);q=np.tile(q,(25,1));c=np.tile(c,(25,1))
        pq=q*np.float32(.75)
        oracle=load(ROOT/'src/sintergpu/references/temporal_best_numeric.py','best_numeric')
        kwargs={}
        if quality_enabled:kwargs['pairwise_quality']=pq
        if confidence_enabled:kwargs['pairwise_confidence']=c
        expected=oracle.temporal_best_observation(eq,t,q,**kwargs)
        from sintergpu.vkl import temporal_best_layout
        plan=temporal_best_layout(t.size,neighbors,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups[0]),quality=quality_enabled,confidence=confidence_enabled)
        module=device.load_module(str(ROOT/'src/sintergpu/kernels/vkl/temporal_best.slang'))
        kernel=device.create_compute_kernel(device.link_program([module],[module.entry_point('temporalBest')]))
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        inputs=dict(equivalence=eq.astype(np.uint32)*7,target_quality=t,neighbor_quality=np.zeros(1,np.float32) if quality_enabled else q,
                    pairwise_quality=pq if quality_enabled else np.zeros(1,np.float32),
                    pairwise_confidence=c if confidence_enabled else np.zeros(1,np.float32))
        bindings={'params':plan['params']};outputs={}
        for name,a in inputs.items():
            a=np.ascontiguousarray(a).reshape(-1);bindings[name]=device.create_buffer(element_count=a.size,struct_size=4,usage=usage,data=a)
        for name,a in expected.items():
            dtype=np.int32 if name=='best_index' else np.float32
            padded=np.full(a.size+13,73,dtype=dtype)
            outputs[name]=(device.create_buffer(element_count=padded.size,struct_size=4,usage=usage,data=padded),dtype)
            bindings[name+'Out']=outputs[name][0]
        encoder=device.create_command_encoder()
        kernel.dispatch(thread_count=list(plan['thread_count']),vars=bindings,command_encoder=encoder)
        device.submit_command_buffer(encoder.finish())
        for name,(buffer,dtype) in outputs.items():
            actual=buffer.to_numpy().view(dtype)
            assert np.all(actual[t.size:]==73)
            if name=='best_index':np.testing.assert_array_equal(actual[:t.size],expected[name])
            else:np.testing.assert_allclose(actual[:t.size],expected[name],rtol=8e-6,atol=5e-7,equal_nan=True,err_msg=name)
    finally:device.close()
