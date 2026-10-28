# SPDX-License-Identifier: Apache-2.0
"""CPU-only recording and timing boundary tests."""
import types
import numpy as np
import pytest
from benchmarks.depth_pyramid_resident_timing import collect

class Buffer:
    def __init__(self,data,e): self.data=data.copy(); self.e=e
    def copy_from_numpy(self,a): self.e.append('upload'); self.data=a.copy()
    def to_numpy(self): self.e.append('read'); return self.data.copy()
class Encoder:
    def __init__(self,e): self.e=e
    def write_timestamp(self,pool,index): self.e.append(('timestamp',index))
    def finish(self): self.e.append('finish'); return object()
class Pool:
    def reset(self): pass
    def get_results(self,a,b): return [0,1000]
class Device:
    def __init__(self,e): self.e=e; self.info=types.SimpleNamespace(timestamp_frequency=1000000)
    def create_command_encoder(self): self.e.append('encoder'); return Encoder(self.e)
    def submit_command_buffer(self,c): self.e.append('submit')
    def wait_for_idle(self): self.e.append('wait')
    def create_query_pool(self,t,n): return Pool()
class Graph:
    def __init__(self,e):
        self.e=e; self.original={'depth':np.full(20,73,np.float32)}
        self.inputs={'depth':Buffer(self.original['depth'],e)}
        self.plans=[{'level':1,'shape':(1,1,1,1)}]
        self.output_initial=[{'depthOut':np.full(20,73,np.float32)}]
        self.levels=[{'depthOut':Buffer(self.output_initial[0]['depthOut'],e)}]
    def record(self,encoder): self.e.extend(['barrier','dispatch'])
    def read_checked(self,validator):
        self.e.append('checked_read')
        if not np.array_equal(self.levels[0]['depthOut'].data[-19:],self.output_initial[0]['depthOut'][-19:]): raise AssertionError('tail changed')
        validator(1,{'depthOut':np.array([[[73.]]],np.float32)})


def test_resident_dispatch_boundaries_and_paired_samples(monkeypatch):
    import benchmarks.depth_pyramid_resident_timing as module
    events=[]; clock=lambda: (events.append('clock') or len(events)*1000)
    monkeypatch.setattr(module.time,'perf_counter_ns',clock)
    out=collect(Device(events),object(),Graph(events),Graph(events),lambda level,result:events.append('validate'),lambda:{'source':'same'},samples=50,warmups=0)
    assert len(out['runs'])==3
    assert all(r['order']==['old_before','candidate','old_after']*50 for r in out['runs'])
    i=events.index(('timestamp',0)); j=events.index(('timestamp',1),i)
    assert events[i+1:j]==['barrier','dispatch']
    assert all(len(r['gpu_resident_ms']['candidate'])==50 for r in out['runs'])
    first_read=events.index('read'); assert events[first_read-1]=='wait'
    # transfer clock stops after the final raw readback, before validation
    assert events[first_read+2]=='clock' and events[first_read+3]=='validate'


def test_failure_before_reset():
    events=[]
    def reject(*args): raise AssertionError('oracle')
    with pytest.raises(AssertionError,match='oracle'):
        collect(Device(events),object(),Graph(events),Graph(events),reject,lambda:{'source':'same'},samples=50,warmups=0)
    assert 'upload' not in events


def test_minimum_sample_count():
    with pytest.raises(ValueError,match='50'):
        collect(Device([]),object(),Graph([]),Graph([]),lambda *x:None,lambda:{'source':'same'},samples=49,warmups=0)


def test_raw_all_levels_reject_tail_corruption_before_reset():
    events=[]; old=Graph(events); new=Graph(events)
    new.levels[0]['depthOut'].data[-1]=0
    with pytest.raises(AssertionError,match='tail changed'):
        collect(Device(events),object(),old,new,lambda *a:None,lambda:{'source':'same'},samples=50,warmups=0)
    assert events.count('upload')==0  # invalid priming guard fails before reset


def test_graph_contract_and_source_changes_fail_closed():
    events=[]; old=Graph(events); new=Graph(events)
    new.original['depth'][-1]=0
    with pytest.raises(ValueError,match='input buffer mismatch'):
        collect(Device(events),object(),old,new,lambda *a:None,lambda:{'source':'same'},samples=50,warmups=0)
    old=Graph(events); new=Graph(events); ids=iter([{'source':'a'},{'source':'b'}])
    with pytest.raises(ValueError,match='identity changed'):
        collect(Device(events),object(),old,new,lambda *a:None,lambda:next(ids),samples=50,warmups=0)


def test_skipped_later_write_is_detected_after_poison_reset():
    events=[]
    class Writer(Graph):
        def __init__(self,e): super().__init__(e); self.calls=0
        def record(self,encoder):
            self.calls+=1; self.e.extend(['barrier','dispatch'])
            if self.calls != 4: self.levels[0]['depthOut'].data[0]=5
        def read_checked(self,validator):
            self.e.append('checked_read')
            validator(1,{'depthOut':self.levels[0]['depthOut'].data[:1].reshape(1,1,1).copy()})
    def validate(level,result):
        np.testing.assert_array_equal(result['depthOut'],5)
    with pytest.raises(AssertionError):
        collect(Device(events),object(),Writer(events),Writer(events),validate,lambda:{'source':'same'},samples=50,warmups=0)


def test_validation_then_reset_then_timed_resident_and_transfer(monkeypatch):
    import benchmarks.depth_pyramid_resident_timing as module
    events=[]
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:(events.append('clock') or len(events)*1000))
    collect(Device(events),object(),Graph(events),Graph(events),lambda *a:events.append('validate'),lambda:{'source':'same'},samples=50,warmups=0)
    starts=[i for i,e in enumerate(events) if e==('timestamp',0)]
    measured=starts[2]
    assert events[measured-3:measured]==['wait','clock','encoder']
    assert events[measured-4]=='upload' and events[measured-5]=='validate'
    stop=events.index(('timestamp',1),measured)
    validation=events.index('validate',stop)
    assert events[validation+1:validation+5]==['upload','wait','clock','upload']
