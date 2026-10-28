# SPDX-License-Identifier: Apache-2.0
"""Fake-device graph contracts; no GPU execution or qualification."""
import hashlib
import numpy as np
import pytest
from benchmarks.depth_pyramid_resident_graph import ResidentGraph, ROOT

class Buffer:
    def __init__(self,data,events): self.data=data.copy(); self.size=data.nbytes; self.events=events
    def to_numpy(self): self.events.append('read'); return self.data.copy()
class Device:
    def __init__(self): self.events=[]
    def create_buffer(self,*,data,**kwargs): self.events.append('allocate'); return Buffer(data,self.events)
class Kernel:
    def __init__(self,name,events): self.name=name; self.events=events
    def dispatch(self,**kwargs): self.events.append(('dispatch',self.name,kwargs))
class Encoder:
    def __init__(self,events): self.events=events
    def global_barrier(self): self.events.append('barrier')


def source_ids():
    paths=('src/sintergpu/kernels/vkl/depth_resampling.slang','src/sintergpu/kernels/vkl/depth_pyramid.slang')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def setup(shape=(2,5,7),confidence=True):
    device=Device(); kernels={'initializeDepth':Kernel('init',device.events),'depthPyramidLevel':Kernel('level',device.events)}
    d=np.ones(shape,np.float32); v=np.ones(shape,np.uint32)*7; c=np.full(shape,.5,np.float32)
    graph=ResidentGraph(device,kernels,{'depth':d,'valid':v,'confidence':c},usage=3,max_workgroups_x=65535,source_ids=source_ids(),confidence=confidence)
    return graph,device


def test_records_barriers_and_chained_buffers_without_hidden_work():
    graph,device=setup(); device.events.clear(); encoder=Encoder(device.events); graph.record(encoder)
    assert [e if isinstance(e,str) else e[:2] for e in device.events]==['barrier',('dispatch','init'),'barrier',('dispatch','level'),'barrier',('dispatch','level'),'barrier',('dispatch','level')]
    calls=[e[2] for e in device.events if isinstance(e,tuple)]
    assert all(c['command_encoder'] is encoder for c in calls)
    for previous,current in zip(calls,calls[1:]):
        for name in ('depth','valid','confidence'):
            assert current['vars'][name] is previous['vars'][name+'Out']


def test_checks_every_level_input_and_tail():
    graph,device=setup(); seen=[]
    graph.read_checked(lambda level,outputs: seen.append((level,outputs)))
    assert [level for level,_ in seen]==[1,2,4,8]
    graph.levels[2]['confidenceOut'].data[-1]=0
    with pytest.raises(AssertionError,match='tail'): graph.read_checked(lambda *x:None)
    graph.levels[2]['confidenceOut'].data[-1]=73
    graph.inputs['depth'].data[0]=0
    with pytest.raises(AssertionError,match='input'): graph.read_checked(lambda *x:None)


def test_contract_rejects_sources_shapes_and_missing_kernels():
    graph,device=setup((1,1,1)); assert len(graph.plans)==1
    with pytest.raises(ValueError): setup((0,1,1))
    ids=source_ids(); ids[next(iter(ids))]='0'*64
    d=np.ones((1,2,2),np.float32)
    with pytest.raises(ValueError,match='source'): ResidentGraph(device,{}, {'depth':d,'valid':d.astype(np.uint32),'confidence':d},usage=3,max_workgroups_x=65535,source_ids=ids)


def test_array_candidate_requires_shared_include_identity():
    from benchmarks.depth_pyramid_resident_graph import check_sources
    init='src/sintergpu/kernels/vkl/depth_resampling.slang'
    candidate='benchmarks/shaders/depth_pyramid_vkl_candidate.slang'
    core='src/sintergpu/kernels/common/depth_pyramid_core.slang'
    ids={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (init,candidate)}
    with pytest.raises(ValueError,match='include'): check_sources(ids)
    ids[core]=hashlib.sha256((ROOT/core).read_bytes()).hexdigest()
    check_sources(ids)


def test_validator_receives_every_output_and_can_reject_corruption():
    graph,_=setup()
    for level in graph.levels:
        for name,b in level.items(): b.data[:-19]=7
    seen=[]
    def validate(level,outputs):
        seen.append(level)
        assert set(outputs)=={'depthOut','validOut','confidenceOut'}
        for array in outputs.values(): np.testing.assert_array_equal(array,7)
    graph.read_checked(validate)
    assert seen==[1,2,4,8]
    graph.levels[1]['depthOut'].data[0]=3
    with pytest.raises(AssertionError): graph.read_checked(validate)
