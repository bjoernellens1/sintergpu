# SPDX-License-Identifier: Apache-2.0
"""CPU-only timestamp runner preflight and lock tests."""
import os, fcntl, types
import pytest
from benchmarks import depth_pyramid_timestamp_runner as runner


def test_closure_and_explicit_execution():
    assert {'benchmarks/depth_pyramid_timestamp_runner.py','benchmarks/depth_pyramid_timestamp_probe.py','benchmarks/depth_pyramid_resident_graph.py','src/sintergpu/vkl.py'} <= set(runner.EXTRA)
    with pytest.raises(ValueError,match='explicit --execute'):
        runner.run(types.SimpleNamespace(execute=False))


def test_owned_lease_nonblocking_and_held_until_exit(tmp_path,monkeypatch):
    import benchmarks.depth_pyramid_timestamp_probe as probe
    path=tmp_path/'device.lock'; monkeypatch.setattr(runner,'LOCK',path); monkeypatch.setattr(probe,'LOCK',path)
    with runner.owned_lease() as lease:
        lease.verify()
        other=os.open(path,os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError): fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally: os.close(other)
    with pytest.raises(ValueError): lease.verify()


@pytest.fixture
def fake_runtime(monkeypatch,tmp_path):
    import contextlib, hashlib, sys
    import numpy as np
    import benchmarks.depth_pyramid_consolidation as proto
    import test_depth_pyramid_gpu as reference
    events=[]
    pkg=tmp_path/'spy'; pkg.mkdir(); (pkg/'__init__.py').write_text('')
    binary=pkg/'libslang-compiler-fake'; binary.write_bytes(b'compiler')
    compiler=hashlib.sha256(binary.read_bytes()).hexdigest()
    ids={runner.INIT:'a'*64,runner.STREAM:'b'*64,
         'benchmarks/depth_pyramid_timestamp_probe.py':hashlib.sha256((runner.ROOT/'benchmarks/depth_pyramid_timestamp_probe.py').read_bytes()).hexdigest()}
    monkeypatch.setattr(runner,'closure',lambda commit:dict(ids))
    monkeypatch.setattr(runner,'verify_loaded_closure',lambda ids:events.append('verify_modules'))
    monkeypatch.setattr(runner,'require_local_modules',lambda mods:None)
    @contextlib.contextmanager
    def owned(): events.append('lock'); yield object(); events.append('unlock')
    monkeypatch.setattr(runner,'owned_lease',owned)
    snapshot={'physical_device_uuid':'uuid','driver':{'version':'v','info':'i'},'vendor_id':'0x10de','device_id':'0x1000','adapter_name':'fake'}
    monkeypatch.setattr(runner,'snapshot',lambda:dict(snapshot))
    storage={'icd_environment':{'VK_DRIVER_FILES':'fake'},'max_storage_buffer_range':2**28,'raw':'fake','sha256':'d'*64}
    monkeypatch.setattr(runner,'full_vulkan_storage',lambda uuid:dict(storage))
    monkeypatch.setattr(runner,'check_limits',lambda *a,**k:{'max_groups_x':65535})
    class Device:
        @staticmethod
        def enumerate_adapters(t): return [types.SimpleNamespace(vendor_id=0x10de,device_id=0x1000,luid='luid')]
        def __init__(self,**kw):
            events.append('device'); self.info=types.SimpleNamespace(adapter_name='fake',limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535)))
        def load_module(self,path): events.append(('load',path)); return types.SimpleNamespace(entry_point=lambda entry:entry)
        def link_program(self,modules,entries): return entries[0]
        def create_compute_kernel(self,program): return object()
        def create_command_encoder(self): return types.SimpleNamespace(finish=lambda:object())
        def submit_command_buffer(self,c): events.append('submit')
        def wait_for_idle(self): events.append('wait')
        def close(self): events.append('close')
    spy=types.SimpleNamespace(__file__=str(pkg/'__init__.py'),Device=Device,DeviceType=types.SimpleNamespace(vulkan=1),SlangFloatingPointMode=types.SimpleNamespace(precise=1),BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),QueryType=types.SimpleNamespace(timestamp=1))
    monkeypatch.setitem(sys.modules,'slangpy',spy)
    monkeypatch.setattr(runner.importlib.metadata,'version',lambda _: 'fake')
    class Graph:
        def __init__(self,*args,**kwargs): events.append('graph')
        def record(self,encoder): events.append('record')
        def read_checked(self,validate): events.append('checked'); validate(1,{'depthOut':np.array([1],np.float32)})
    monkeypatch.setattr(runner,'ResidentGraph',Graph)
    monkeypatch.setattr(proto,'cpu_reference',lambda *a,**kw:{'levels':[(np.array([1],np.float32),)*3]*4})
    monkeypatch.setattr(reference,'compare',lambda *a:None)
    def collect(device,query,graph,identity,validate,*,samples,lease):
        events.append(('collect',samples)); metadata=identity(); validate(); validate()
        return {'timestamp_frequency':1,'empty_pairs':[[1,1]],'pipeline_pairs':[[1,2]],'identity_unverified':metadata}
    monkeypatch.setattr(runner,'collect',collect)
    receipt=tmp_path/'receipt'; receipt.write_text('{}')
    args=types.SimpleNamespace(execute=True,commit='a'*40,output=str(tmp_path/'report'),control_receipt=str(receipt),gpu_uuid='uuid',driver_version='v',driver_info='i',compiler_sha256=compiler,samples=50)
    return args,events,ids


def test_fake_orchestration_retains_raw_ticks_and_closes_under_lock(fake_runtime):
    import json
    args,events,ids=fake_runtime
    runner.run(args)
    result=json.loads(open(args.output).read())
    assert result['status']=='diagnostic_only' and result['raw_ticks']['pipeline_pairs']==[[1,2]]
    assert 'no per-sample parity' in result['scope']
    assert events.index('lock')<events.index('device')<events.index('close')<events.index('unlock')
    assert ('collect',50) in events
    assert events.count('record')==1 and events.count('checked')==2
    assert events.index('submit')<events.index('wait')<events.index('checked')


@pytest.mark.parametrize('failure',['identity','closure','lock','limits','write'])
def test_failure_paths_do_not_publish(fake_runtime,monkeypatch,failure):
    args,events,ids=fake_runtime
    def fail(error): return lambda *a,**k:(_ for _ in ()).throw(error)
    if failure=='identity': args.gpu_uuid='wrong'
    if failure=='closure': monkeypatch.setattr(runner,'closure',fail(ValueError('closure')))
    if failure=='lock': monkeypatch.setattr(runner,'owned_lease',fail(BlockingIOError('lock')))
    if failure=='limits': monkeypatch.setattr(runner,'check_limits',fail(ValueError('limits')))
    if failure=='write': monkeypatch.setattr(runner,'write_once',fail(OSError('write')))
    with pytest.raises((ValueError,BlockingIOError,OSError)): runner.run(args)
    assert not __import__('pathlib').Path(args.output).exists()
    if failure in ('limits','write'): assert 'close' in events
    if failure in ('identity','closure','lock'): assert 'device' not in events
