# SPDX-License-Identifier: Apache-2.0
"""Opt-in VKL stream parity; review and external exclusive GPU lock required."""
import os
import types
import numpy as np
import pytest
from test_depth_pyramid_gpu import CASES, EPS, compare, device as incumbent_device, execute, expected
from test_depth_pyramid_candidate_gpu import assert_exact
from benchmarks.depth_pyramid_gpu_fixture import Fixture, specification
from depth_pyramid_stream_fixture import stream_preflight, stream_fixture, stream_source

OPTIN=pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS')!='1' or os.getenv('SINTERGPU_DEPTH_PYRAMID_STREAM_GPU')!='1',
                        reason='stream GPU opt-in, independent review and external exclusive lock required')
def raw_confidence_cases():
    # Direct shader contract probes, outside donor's usual confidence domain.
    for index,confidence in enumerate((
        [np.nan, .2, .3, .4], [np.inf, .2, .3, .4],
        [-np.inf, .2, .3, .4], [-0., 0., .2, .4],
        [.1, np.nan, np.inf, -np.inf],
    )):
        for distinct in (False,True):
            depth=np.array([1.,1.,1.,2. if distinct else 1.],np.float32).reshape(1,2,2)
            valid=np.ones_like(depth,np.uint32)
            conf=np.array(confidence,np.float32).reshape(depth.shape)
            yield f'vkl-raw-confidence-{index}-distinct-{distinct}', specification('vkl',depth,valid=valid,confidence=conf,threshold=EPS)

def raw_poison_cases():
    """Raw shader probes; confidence values intentionally exceed donor domain."""
    items=(
        ('vkl-raw-poison-masked',[1.,1.,1.,1.],[1,0,0,1],[.25,np.nan,np.inf,.75],(1,2,2)),
        ('vkl-raw-poison-nan-zero-depth',[1.,np.nan,0.,2.],[1,1,1,1],[.25,np.nan,np.inf,.75],(1,2,2)),
        ('vkl-raw-poison-oob',[1.],[1],[.5],(1,1,1)),
        ('vkl-raw-first-min-negative-zero',[1.,1.,2.,3.],[1,1,1,1],[-0.,.75,.2,.4],(1,2,2)),
    )
    for name,depth,valid,conf,shape in items:
        d=np.array(depth,np.float32).reshape(shape)
        v=np.array(valid,np.uint32).reshape(shape)
        c=np.array(conf,np.float32).reshape(shape)
        yield name,specification('vkl',d,valid=v,confidence=c,threshold=EPS)


def poison_oob_confidence(fixture):
    # Poison the allocated guard immediately after the single live sample.
    # Actual out-of-bounds reads are still forbidden; read_checked verifies it remains intact.
    guard=fixture.initial['confidence'][fixture.spec['inputs']['confidence'].size:]
    guard[::2]=np.nan
    guard[1::2]=np.inf
    fixture.buffers['confidence'].copy_from_numpy(fixture.initial['confidence'])


VKL_CASES=[(name,spec) for name,spec in CASES if spec['profile']=='vkl']+list(raw_confidence_cases())+list(raw_poison_cases())


@pytest.fixture(scope='module')
def stream_device():
    stream_preflight()
    generator=incumbent_device.__wrapped__()
    dev=next(generator)
    try: yield dev
    finally: generator.close()


def test_preflight_and_source_identity():
    identities=stream_preflight()
    assert len(identities)>=7
    assert 'depthPyramidVklAccumulate' in stream_source()
    assert sum(name.startswith('vkl-mask-') for name,_ in VKL_CASES)==16
    assert sum(name.startswith('vkl-threshold-') for name,_ in VKL_CASES)==3
    assert any(name.startswith('vkl-shape-') and s['shape'][0]>1 for name,s in VKL_CASES)
    assert sum(name.startswith('vkl-raw-confidence-') for name,_ in VKL_CASES)==10


def test_stream_loads_distinct_module_with_borrowed_bindings(monkeypatch):
    import sys
    events=[]
    class Kernel:
        def dispatch(self,**kwargs): events.append(('dispatch',kwargs))
    class Device:
        info=types.SimpleNamespace(limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535,y=65535),max_compute_threads_per_group=256,max_compute_thread_group_size=types.SimpleNamespace(x=256,y=256)))
        def create_buffer(self,**kw): return object()
        def load_module(self,path): events.append(('incumbent',path)); return types.SimpleNamespace(entry_point=lambda x:x)
        def load_module_from_source(self,name,source): events.append(('source',name,source)); return types.SimpleNamespace(entry_point=lambda x:x)
        def link_program(self,modules,entries): return entries[0]
        def create_compute_kernel(self,program): return Kernel()
    class Encoder:
        def global_barrier(self): events.append(('barrier',))
    monkeypatch.setitem(sys.modules,'slangpy',types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2)))
    d=np.ones((1,4,4),np.float32)
    spec=specification('vkl',d,valid=np.ones_like(d,np.uint32),confidence=d,threshold=EPS)
    fixture=stream_fixture(Device(),spec)
    assert events[-1][0]=='source' and events[-1][2]==stream_source()
    assert events[-1][1].startswith('depth_pyramid_stream_')
    bindings=fixture.bindings; encoder=Encoder(); fixture.record(encoder)
    assert events[-2][0]=='barrier' and events[-1][0]=='dispatch'
    assert events[-1][1]['vars'] is bindings and events[-1][1]['command_encoder'] is encoder


@OPTIN
@pytest.mark.parametrize('case_id,spec', VKL_CASES, ids=[name for name,_ in VKL_CASES])
def test_stream_matches_incumbent_and_reference(stream_device,case_id,spec):
    initial=stream_preflight()
    old_fixture=Fixture(stream_device,spec)
    new_fixture=stream_fixture(stream_device,spec)
    if case_id=='vkl-raw-poison-oob':
        poison_oob_confidence(old_fixture); poison_oob_confidence(new_fixture)
    old=execute(stream_device,old_fixture)
    new=execute(stream_device,new_fixture)
    oracle=expected(spec)
    compare(old,oracle); compare(new,oracle); assert_exact(new,old)
    assert stream_preflight()==initial


@OPTIN
def test_stream_reused_buffers(stream_device):
    initial=stream_preflight()
    d=np.ones((2,5,7),np.float32); v=np.ones_like(d,np.uint32)*7; c=np.full_like(d,.5)
    specs=[specification('vkl',d*x,valid=v*(x!=0),confidence=c,threshold=EPS) for x in (1,0,2)]
    old=Fixture(stream_device,specs[0]); new=stream_fixture(stream_device,specs[0])
    for i,spec in enumerate(specs):
        if i: old.upload(spec['inputs']); new.upload(spec['inputs'])
        a=execute(stream_device,old); b=execute(stream_device,new)
        compare(a,expected(spec)); compare(b,expected(spec)); assert_exact(b,a)
    assert stream_preflight()==initial


def test_source_integrity_rejects_mutation_and_symlink(tmp_path):
    from benchmarks.depth_pyramid_gpu_fixture import ROOT, SOURCES
    from depth_pyramid_stream_fixture import STREAM
    for name in [*SOURCES,STREAM]:
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    assert len(stream_preflight(tmp_path))==7
    target=tmp_path/STREAM
    target.write_bytes(target.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='digest mismatch'): stream_preflight(tmp_path)
    target.unlink(); target.symlink_to(ROOT/STREAM)
    with pytest.raises(ValueError,match='symlinked'): stream_preflight(tmp_path)


def test_stream_rejects_non_vkl_before_device_access():
    with pytest.raises(ValueError,match='only supports VKL'):
        stream_fixture(None,{'profile':'slam'})


def test_raw_confidence_cases_are_distinct_from_donor_domain():
    cases=list(raw_confidence_cases())
    assert len(cases)==10
    assert any(np.isnan(s['inputs']['confidence']).any() for _,s in cases)
    assert any(np.isinf(s['inputs']['confidence']).any() for _,s in cases)
    assert any(np.signbit(s['inputs']['confidence']).any() for _,s in cases)
    for _,spec in cases:
        assert set(expected(spec)) == set(spec['outputs'])


def test_raw_poisoned_lanes_and_first_minimum_signed_zero_coverage():
    cases=dict(raw_poison_cases())
    assert set(cases)=={'vkl-raw-poison-masked','vkl-raw-poison-nan-zero-depth',
                        'vkl-raw-poison-oob','vkl-raw-first-min-negative-zero'}
    masked=cases['vkl-raw-poison-masked']['inputs']
    assert np.isnan(masked['confidence'][masked['valid']==0]).any()
    assert np.isinf(masked['confidence'][masked['valid']==0]).any()
    invalid=cases['vkl-raw-poison-nan-zero-depth']['inputs']
    assert np.isnan(invalid['confidence'][np.isnan(invalid['depth'])]).any()
    assert np.isinf(invalid['confidence'][invalid['depth']==0]).any()
    assert cases['vkl-raw-poison-oob']['inputs']['depth'].shape==(1,1,1)
    value=expected(cases['vkl-raw-first-min-negative-zero'])['confidenceOut']
    assert value.item()==0 and np.signbit(value).item()


def test_oob_confidence_guard_is_poisoned_and_uploaded():
    class Buffer:
        def copy_from_numpy(self,value): self.copy=value.copy()
    initial=np.full(20,np.float32(73)); initial[0]=.5
    buffer=Buffer()
    f=types.SimpleNamespace(initial={'confidence':initial},
        spec={'inputs':{'confidence':np.array([.5],np.float32)}},buffers={'confidence':buffer})
    poison_oob_confidence(f)
    assert f.initial['confidence'][0]==.5
    assert np.isnan(f.initial['confidence'][1::2]).all()
    assert np.isposinf(f.initial['confidence'][2::2]).all()
    np.testing.assert_array_equal(buffer.copy.view(np.uint32),f.initial['confidence'].view(np.uint32))
