"""Paired color timing recipes retain semantics, identities and pending scope."""
import hashlib
import importlib
import sys
import numpy as np
import pytest


def recipe():
    try:return importlib.import_module('benchmarks.color_timings')
    except ImportError:pytest.fail('color CPU/GPU timing recipe missing')


def test_color_workloads_cover_six_operations_and_preserve_reference_inputs():
    m=recipe();cases=list(m.cases(small_only=True))
    assert len(cases)==24
    assert len({c['operation'] for c in cases})==6
    for c in cases:
        before={k:v.copy() for k,v in c['arrays'].items()}
        a=m.reference_call(c);b=m.reference_call(c)
        assert set(a)==set(c['outputs'])
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])
            assert a[name].dtype==np.uint8
            np.testing.assert_array_equal(a[name][~c['live'][name]],c['arrays'][name][~c['live'][name]])
        for k,v in before.items():np.testing.assert_array_equal(c['arrays'][k],v)
        assert m.workload_identity(c)==m.workload_identity(c)
        changed=dict(c,arrays={k:v.copy() for k,v in c['arrays'].items()})
        changed['arrays']['sourceBytes'][0]^=1
        assert m.workload_identity(changed)['sha256']!=m.workload_identity(c)['sha256']


def test_cpu_collection_is_registry_compatible_and_does_not_import_gpu_runtime():
    m=recipe();was_imported='slangpy' in sys.modules
    report=m.run(gpu=False,iterations=1,warmups=1,small_only=True)
    from sintergpu.timings import build_registry
    from sintergpu.catalog import list_operations
    db=build_registry(list_operations(),report['measurements'])
    assert db['coverage']['cpu_measured']==6 and db['coverage']['gpu_measured']==0
    assert len(report['measurements'])==24
    assert ('slangpy' in sys.modules)==was_imported
    for row in report['measurements']:
        assert row['metrics'].keys()=={'cpu_reference_wall_ms'}
        assert row['iterations']==1 and row['warmups']==1
        assert row['environment']['captured_at_utc']
        assert row['reference_sha256']==hashlib.sha256(m.Path(m.reference.__file__).read_bytes()).hexdigest()
        assert row['runner_dependencies_sha256']


def test_full_recipe_adds_larger_workloads_without_changing_small_identities():
    m=recipe();small=list(m.cases(small_only=True));all_cases=list(m.cases())
    assert len(all_cases)==30
    assert [m.workload_identity(c) for c in all_cases[:24]]==[m.workload_identity(c) for c in small]
    assert all(c['params']['srcWidth']==640 and c['params']['srcHeight']==480 for c in all_cases[24:])
    assert all(c['tags'].matrix=='bt709' and c['tags'].range=='limited' for c in all_cases[24:])


def test_parity_rejects_live_errors_and_any_padding_change():
    m=recipe();case=next(m.cases(small_only=True));expected=m.reference_call(case)
    actual={k:v.copy() for k,v in expected.items()};m.check_parity(case,actual,expected)
    name=case['outputs'][0];live=np.flatnonzero(case['live'][name]);padding=np.flatnonzero(~case['live'][name])
    actual[name][live[0]]=3 if expected[name][live[0]]<2 else expected[name][live[0]]-2
    with pytest.raises(AssertionError):m.check_parity(case,actual,expected)
    actual={k:v.copy() for k,v in expected.items()};actual[name][padding[0]]^=1
    with pytest.raises(AssertionError):m.check_parity(case,actual,expected)


@pytest.mark.parametrize('iterations,warmups',[(0,1),(1,0),(100001,1),(1,10001),(True,1)])
def test_collection_rejects_invalid_sample_counts(iterations,warmups):
    with pytest.raises(ValueError):recipe().run(iterations=iterations,warmups=warmups,small_only=True)


@pytest.mark.parametrize('kind',['nv12_encode','fused_remap'])
def test_gpu_collection_resets_outputs_and_counts_all_transfer_bytes(tmp_path,monkeypatch,kind):
    import types
    import sintergpu.adapters.slangpy as adapter_module
    m=recipe();case=next(c for c in m.cases(small_only=True) if c['kind']==kind)
    expected=m.reference_call(case);events=[];created=[]
    class Buffer:
        def __init__(self,a,name):self.data=a.copy();self.name=name
        def copy_from_numpy(self,a):events.append(('upload',self.name));self.data=a.copy()
        def to_numpy(self):events.append(('read',self.name));return self.data.view(np.uint32).reshape(-1)
    class Query:
        def reset(self):events.append(('query-reset',))
        def get_timestamp_results(self,start,count):return [2.,2.001]
    class Device:
        closed=False
        @staticmethod
        def enumerate_adapters(kind):return [types.SimpleNamespace(vendor_id=0x1002,device_id=0x1586,luid='test')]
        def __init__(self,**kwargs):self.info=types.SimpleNamespace(adapter_name='Test GPU',api_name='Vulkan',timestamp_frequency=1000)
        def create_buffer(self,*,data,**kwargs):
            name=list(case['arrays'])[len(created)];created.append(name);events.append(('allocate',name));return Buffer(data,name)
        def create_query_pool(self,*args):return Query()
        def wait(self):events.append(('wait',))
        def close(self):Device.closed=True
    class Kernel:
        def dispatch(self,*,vars,thread_count,**kwargs):
            assert thread_count==case['dimensions'];events.append(('dispatch','query_pool' in kwargs))
            for name,value in expected.items():vars[name].data=value.copy()
    class Adapter:
        def __init__(self,*args,**kwargs):self.source_identities={case['operation']:'f'*64}
        def kernel(self,operation):return Kernel()
    spy=types.ModuleType('slangpy');spy.__file__=str(tmp_path/'__init__.py');spy.Device=Device
    spy.DeviceType=types.SimpleNamespace(vulkan='vulkan');spy.SlangFloatingPointMode=types.SimpleNamespace(precise='precise')
    spy.QueryType=types.SimpleNamespace(timestamp='timestamp');spy.BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2)
    (tmp_path/'libslang-compiler.so').write_bytes(b'test compiler')
    monkeypatch.setenv('SINTERGPU_GPU_TESTS','1');monkeypatch.setitem(sys.modules,'slangpy',spy)
    monkeypatch.setattr(adapter_module,'SlangPyAdapter',Adapter)
    monkeypatch.setattr(m,'cases',lambda **kwargs:[case])
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:(_ for _ in ()).throw(FileNotFoundError('optional diagnostic')))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda name:'test-version')
    row=m.run(gpu=True,iterations=1,warmups=1,small_only=True)['measurements'][0]
    assert Device.closed
    assert row['metrics']['gpu_kernel_ms']==pytest.approx([1.])
    assert row['uploaded_bytes_per_iteration']==sum(a.nbytes for a in case['arrays'].values())
    assert row['readback_bytes_per_iteration']==sum(case['arrays'][n].nbytes for n in case['outputs'])
    # All allocation precedes sampling; warmup/kernel reset only outputs, then
    # transfer warmup/sample upload every input plus required output seed.
    assert created==list(case['arrays'])
    first_dispatch=next(i for i,e in enumerate(events) if e[0]=='dispatch')
    assert not any(e[0]=='allocate' for e in events[first_dispatch:])
    dispatch_indices=[i for i,e in enumerate(events) if e[0]=='dispatch']
    assert len(dispatch_indices)==4
    for i in dispatch_indices[:2]:
        assert events[i-len(case['outputs'])-1:i]==[('upload',n) for n in case['outputs']]+[('wait',)] or (events[i-1]==('query-reset',) and events[i-len(case['outputs'])-2:i-1]==[('upload',n) for n in case['outputs']]+[('wait',)])
    for i in dispatch_indices[2:]:assert events[i-len(case['arrays']):i]==[('upload',n) for n in case['arrays']]
    assert len([e for e in events if e==('dispatch',True)])==1
