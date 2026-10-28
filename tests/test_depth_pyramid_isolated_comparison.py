"""Contract tests use fake devices; no timings qualify hardware."""
import numpy as np
import pytest
import copy
import types
import hashlib
from benchmarks.depth_pyramid_gpu_fixture import ROOT
from benchmarks.depth_pyramid_isolated_comparison import collect, evaluate

class Buffer:
    def __init__(self, events, name): self.events,self.name=events,name
    def copy_from_numpy(self, value): self.events.append(('copy',self.name)); self.data=value.copy()
    def to_numpy(self): self.events.append(('read_raw',self.name)); return self.data.copy()
class Pool:
    def __init__(self, events): self.events=events
    def reset(self): self.events.append(('reset_query',))
    def get_timestamp_results(self, start, count): return [0., .001]
class Device:
    def __init__(self, events): self.events=events; self.info=types.SimpleNamespace(timestamp_frequency=1000)
    def create_query_pool(self, kind, n): self.events.append(('pool',n)); return Pool(self.events)
    def wait_for_idle(self): self.events.append(('wait',))
class Kernel:
    def __init__(self, events, name): self.events,self.name=events,name
    def dispatch(self, **kw): self.events.append(('dispatch',self.name,tuple(sorted(kw))))
class Fixture:
    def __init__(self, events,name):
        self.name=name; self.spec={'inputs':{'d':np.ones((2,),np.float32)},'outputs':{'out':np.float32},'threads':(1,1,1),'profile':'vkl','params':{'epsilon':.01},'shape':(1,),'groups':(1,1),'entry':'entry','parameter':'params','count':1}
        self.initial={'d':np.ones(21,np.float32),'out':np.ones(20,np.float32)}
        self.buffers={k:Buffer(events,k) for k in self.initial}
        for k,v in self.initial.items(): self.buffers[k].data=v.copy()
        self.bindings={'out':self.buffers['out']}; self.kernel=Kernel(events,name); self.events=events
    def upload(self, inputs): self.events.append(('upload',self.name))
    def read_checked(self): self.events.append(('read',self.name)); return {'out':np.ones(1,np.float32)}

def identity():
    paths=['src/sintergpu/kernels/common/depth_pyramid_core.slang','benchmarks/shaders/depth_pyramid_vkl_candidate.slang','src/sintergpu/kernels/vkl/depth_pyramid.slang']
    return {'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'compiler':'compiler version','compiler_sha256':'c'*64,'runtime':'runtime version','device':'device name','driver':'driver version','device_uuid':'uuid','timestamp_frequency':1000,'control_receipt':{'sha256':'b'*64}}

def setup():
    e=[]; return e, Device(e), {'old':Fixture(e,'old'),'candidate':Fixture(e,'candidate')}

def test_collection_orders_pairs_and_validates_all_samples():
    e,device,fixtures=setup(); validations=[]
    result=collect(device, object(), fixtures, lambda output: validations.append(output), identity, samples=50, warmups=0)
    assert len(result['samples']) == 3
    for run in result['samples']:
        assert run['order'] == ['old_before','candidate','old_after']*50
        for boundary in ('isolated_gpu_ms','host_complete_ms','transfer_inclusive_ms'):
            assert all(len(run[boundary][key])==50 for key in ('old_before','candidate','old_after'))
    assert len(validations)==3*50*3*2
    assert sum(x==('copy','d') for x in e)==3*50*3


def test_bad_input_identity_and_bad_timestamps_fail_closed():
    e,device,fixtures=setup(); fixtures['candidate'].spec['inputs']['d'][0]=2
    with pytest.raises(ValueError, match='input'): collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)
    e,device,fixtures=setup()
    with pytest.raises(ValueError, match='50'): collect(device,object(),fixtures,lambda x:None,lambda:{},samples=49,warmups=0)
    class BadPool(Pool):
        def get_timestamp_results(self,*args): return [1.,1.]
    device.create_query_pool=lambda *args:BadPool(e)
    with pytest.raises(ValueError, match='timestamp'): collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)


def test_evaluation_checks_each_run_and_each_boundary():
    e,device,fixtures=setup(); report=collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)
    for run in report['samples']:
        for boundary in ('isolated_gpu_ms','host_complete_ms','transfer_inclusive_ms'):
            run[boundary]={'old_before':[2.]*50,'candidate':[1.]*50,'old_after':[2.]*50}
    assert evaluate(report,bootstrap_samples=1000)['status']=='statistical_pass'
    report['samples'][2]['isolated_gpu_ms']['candidate']=[3.]*50
    assert evaluate(report,bootstrap_samples=1000)['status']=='inconclusive'


def test_output_and_source_changes_abort_collection():
    e,device,fixtures=setup()
    fixtures['candidate'].read_checked=lambda: {'out':np.zeros(1,np.float32)}
    with pytest.raises(AssertionError,match='measured and transfer'):
        collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)
    e,device,fixtures=setup(); changed=identity(); changed['driver']='other'; identities=iter([identity(),changed])
    with pytest.raises(ValueError,match='source identity changed'):
        collect(device,object(),fixtures,lambda x:None,lambda:next(identities),samples=50,warmups=0)

@pytest.mark.parametrize('field,value', [
    ('profile','slam'),('params',{'epsilon':.02}),('shape',(1,1)),
    ('groups',(2,1)),('entry','other'),('parameter','other'),
    ('threads',(2,1,1)),('count',2),
])
def test_dispatch_contract_mismatch_rejected(field,value):
    _,device,fixtures=setup(); fixtures['candidate'].spec[field]=value
    with pytest.raises(ValueError):
        collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)


def test_report_captures_workload_boundaries_identity_and_controls():
    _,device,fixtures=setup()
    report=collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)
    assert report['observed_identity_unverified'] == identity()
    assert report['workload']['params'] == {'epsilon':.01}
    assert report['workload']['inputs']['d']['sha256'] == hashlib.sha256(np.ones(2,np.float32).tobytes()).hexdigest()
    assert set(report['boundaries']) == {'isolated_gpu_ms','host_complete_ms','transfer_inclusive_ms'}
    assert 'separate transfer dispatch' in report['sequence']
    assert 'unverified' in report['scope']


def test_identity_and_storage_mismatch_rejected():
    _,device,fixtures=setup()
    bad=identity(); del bad['driver']
    with pytest.raises(ValueError,match='identity'): collect(device,object(),fixtures,lambda x:None,lambda:bad,samples=50,warmups=0)
    bad=identity(); bad['timestamp_frequency']=2
    with pytest.raises(ValueError,match='frequency'): collect(device,object(),fixtures,lambda x:None,lambda:bad,samples=50,warmups=0)
    bad=identity(); bad['sources'][next(iter(bad['sources']))]='0'*64
    with pytest.raises(ValueError,match='source identity'): collect(device,object(),fixtures,lambda x:None,lambda:bad,samples=50,warmups=0)
    fixtures['candidate'].initial['out']=np.zeros(21,np.float32)
    with pytest.raises(ValueError,match='capacity'): collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)


def test_guard_reset_contents_must_match():
    _,device,fixtures=setup()
    fixtures['candidate'].initial['out'][-1]=0
    with pytest.raises(ValueError,match='contract mismatch'):
        collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)


def test_transfer_validation_runs_after_clock_stops(monkeypatch):
    import benchmarks.depth_pyramid_isolated_comparison as module
    events,device,fixtures=setup()
    def clock(): events.append(('clock',)); return len(events)*1000
    def validate(value): events.append(('validate',))
    monkeypatch.setattr(module.time,'perf_counter_ns',clock)
    collect(device,object(),fixtures,validate,identity,samples=50,warmups=0)
    # First transfer has a clock after its last readback before validation.
    first_raw=next(i for i,e in enumerate(events) if e[0]=='read_raw')
    last_raw=first_raw+1
    assert events[last_raw][0]=='read_raw'
    assert events[last_raw+1][0]=='clock'
    assert events[last_raw+2][0]=='validate'


def test_input_guard_contents_must_match():
    _,device,fixtures=setup()
    fixtures['candidate'].initial['d'][-1]=0
    with pytest.raises(ValueError,match='contract mismatch'):
        collect(device,object(),fixtures,lambda x:None,identity,samples=50,warmups=0)
