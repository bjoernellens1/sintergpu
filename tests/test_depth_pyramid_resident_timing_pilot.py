# SPDX-License-Identifier: Apache-2.0
"""CPU-only pilot preflight tests; no device is opened."""
import types
import pytest
from benchmarks import depth_pyramid_resident_timing_pilot as pilot


def test_pilot_source_closure_contains_graph_and_timing():
    assert {'benchmarks/depth_pyramid_resident_timing_pilot.py','benchmarks/depth_pyramid_resident_timing.py','benchmarks/depth_pyramid_resident_graph.py','src/sintergpu/vkl.py'} <= set(pilot.EXTRA)


def test_requires_explicit_execution_before_access():
    with pytest.raises(ValueError,match='explicit --execute'):
        pilot.run(types.SimpleNamespace(execute=False))


@pytest.fixture
def runtime(monkeypatch,tmp_path):
    import contextlib
    import hashlib
    import importlib.metadata
    import sys
    import numpy as np
    import benchmarks.depth_pyramid_resident_timing as timing
    import benchmarks.depth_pyramid_consolidation as proto
    import test_depth_pyramid_gpu as reference
    events=[]
    package=tmp_path/'fake-spy'; package.mkdir()
    (package/'__init__.py').write_text('')
    compiler=package/'libslang-compiler-test'; compiler.write_bytes(b'compiler')
    compiler_hash=hashlib.sha256(compiler.read_bytes()).hexdigest()
    ids={pilot.INIT:'a'*64,pilot.OLD:'b'*64,pilot.STREAM:'c'*64,
         'benchmarks/depth_pyramid_resident_timing.py':hashlib.sha256((pilot.ROOT/'benchmarks/depth_pyramid_resident_timing.py').read_bytes()).hexdigest()}
    monkeypatch.setattr(pilot,'closure',lambda commit:dict(ids))
    monkeypatch.setattr(pilot,'verify_loaded_closure',lambda ids:events.append('verify_modules'))
    monkeypatch.setattr(pilot,'require_local_modules',lambda mods:events.append('verify_path'))
    @contextlib.contextmanager
    def lock():
        events.append('lock'); yield; events.append('unlock')
    monkeypatch.setattr(pilot,'lock_device',lock)
    snapshot={'physical_device_uuid':'uuid','driver':{'version':'v','info':'i'},'vendor_id':'0x10de','device_id':'0x1000','adapter_name':'fake-gpu'}
    monkeypatch.setattr(pilot,'snapshot',lambda:dict(snapshot))
    storage={'icd_environment':{'VK_DRIVER_FILES':'fake'},'max_storage_buffer_range':2**28,'raw':'fake','sha256':'d'*64}
    monkeypatch.setattr(pilot,'full_vulkan_storage',lambda uuid:dict(storage))
    monkeypatch.setattr(pilot,'check_limits',lambda *a,**k:{'max_groups_x':65535})
    class Device:
        @staticmethod
        def enumerate_adapters(t):
            events.append('enumerate'); return [types.SimpleNamespace(vendor_id=0x10de,device_id=0x1000,luid='luid')]
        def __init__(self,**kw):
            events.append('device'); self.info=types.SimpleNamespace(adapter_name='fake-gpu',limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535)))
        def load_module(self,path):
            events.append(('load',path)); return types.SimpleNamespace(entry_point=lambda entry:entry)
        def link_program(self,modules,entries): return entries[0]
        def create_compute_kernel(self,program): return object()
        def close(self): events.append('close')
    spy=types.SimpleNamespace(__file__=str(package/'__init__.py'),Device=Device,DeviceType=types.SimpleNamespace(vulkan=1),
        SlangFloatingPointMode=types.SimpleNamespace(precise=1),BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),QueryType=types.SimpleNamespace(timestamp=1))
    monkeypatch.setitem(sys.modules,'slangpy',spy)
    monkeypatch.setattr(importlib.metadata,'version',lambda _: 'fake-runtime')
    class Graph:
        def __init__(self,*a,**kw): events.append(('graph',kw['source_ids']))
    monkeypatch.setattr(pilot,'ResidentGraph',Graph)
    monkeypatch.setattr(proto,'cpu_reference',lambda *a,**kw:{'levels':[(np.array([1],np.float32),)*3]*4})
    def collect(device,query,old,new,validator,identity,**kw):
        events.append(('collect',kw)); metadata=identity()
        return {'identity_unverified':metadata,'runs':[],'scope':'fake'}
    monkeypatch.setattr(timing,'collect',collect)
    monkeypatch.setattr(reference,'compare',lambda *a:None)
    receipt=tmp_path/'receipt.json'; receipt.write_text('{}')
    args=types.SimpleNamespace(execute=True,commit='a'*40,output=str(tmp_path/'report.json'),control_receipt=str(receipt),
        gpu_uuid='uuid',driver_version='v',driver_info='i',compiler_sha256=compiler_hash)
    return args,events,ids,snapshot,storage


def test_fake_end_to_end_runner_writes_unqualified_report(runtime):
    import json
    args,events,ids,snapshot,storage=runtime
    pilot.run(args)
    report=json.loads(open(args.output).read())
    assert report['status']=='measurement_only' and 'unverified' in report['scope']
    assert report['cases'][0]['measurement']['identity_unverified']['receipt_sha256']
    assert [x for x in events if isinstance(x,tuple) and x[0]=='collect']==[('collect',{'samples':50,'warmups':5})]
    assert events.index('lock')<events.index('device')<events.index('close')<events.index('unlock')
    loads=[x[1] for x in events if isinstance(x,tuple) and x[0]=='load']
    assert len(loads)==3 and loads[-1].endswith(pilot.STREAM)


@pytest.mark.parametrize('failure',['identity','closure','lock','write'])
def test_fake_runner_failure_closes_device_and_does_not_publish(runtime,monkeypatch,failure):
    args,events,ids,snapshot,storage=runtime
    if failure=='identity': args.gpu_uuid='wrong'
    if failure=='closure': monkeypatch.setattr(pilot,'closure',lambda _:(_ for _ in ()).throw(ValueError('closure')))
    if failure=='lock':
        def denied(): raise BlockingIOError('lock')
        monkeypatch.setattr(pilot,'lock_device',denied)
    if failure=='write': monkeypatch.setattr(pilot,'write_once',lambda *a:(_ for _ in ()).throw(OSError('write')))
    with pytest.raises((ValueError,BlockingIOError,OSError)): pilot.run(args)
    assert not __import__('pathlib').Path(args.output).exists()
    if failure=='write': assert 'close' in events
    if failure in ('identity','closure','lock'): assert 'device' not in events


def test_closure_change_during_collection_aborts_and_closes(runtime,monkeypatch):
    args,events,ids,snapshot,storage=runtime
    calls=0
    def changing(commit):
        nonlocal calls
        calls+=1
        result=dict(ids)
        if calls>1: result[pilot.OLD]='f'*64
        return result
    monkeypatch.setattr(pilot,'closure',changing)
    with pytest.raises(ValueError,match='source or control receipt changed'):
        pilot.run(args)
    assert 'close' in events
    assert not __import__('pathlib').Path(args.output).exists()


def test_locked_device_limit_failure_does_not_allocate_graph(runtime,monkeypatch):
    args,events,ids,snapshot,storage=runtime
    monkeypatch.setattr(pilot,'check_limits',lambda *a,**k:(_ for _ in ()).throw(ValueError('storage range')))
    with pytest.raises(ValueError,match='storage range'): pilot.run(args)
    assert 'close' in events and not any(isinstance(e,tuple) and e[0]=='graph' for e in events)
    assert not __import__('pathlib').Path(args.output).exists()
