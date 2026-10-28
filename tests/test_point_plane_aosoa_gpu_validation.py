# SPDX-License-Identifier: Apache-2.0
"""CPU-only fake and oracle tests; no GPU execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

path=Path(__file__).with_name('point_plane_aosoa_gpu_validation.py')
spec=importlib.util.spec_from_file_location('point_plane_aosoa_gpu_validation',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def reference_output(arrays):
    result={k:v.copy() for k,v in arrays.items()}
    p=arrays['points'][arrays['points_indices'],:3].astype(np.float64)
    q=arrays['targets'][:1025,:3].astype(np.float64)
    n=arrays['normals'][:1025,:3].astype(np.float64)
    result['out_0'][:1025]=np.sum(n*(p-q),axis=1)
    j=np.c_[np.cross(p,n),n].astype(np.float32)
    result['out_1'][:4100]=j[:,:4].reshape(-1)
    result['out_1'][4128:6178]=j[:,4:].reshape(-1)
    return result


def test_full_oracle_and_all_guards():
    arrays=m.load_fixture(); result=reference_output(arrays)
    m.validate_outputs(arrays,result)
    for name,index in [('out_0',500),('out_0',1038),('out_1',4101),('out_1',6198),('points',0),('points_indices',0)]:
        bad={k:v.copy() for k,v in result.items()}; bad[name].flat[index]+=1
        with pytest.raises(AssertionError): m.validate_outputs(arrays,bad)


def test_paired_full_buffer_parity_and_unique_resources():
    arrays=m.load_fixture(); expected=reference_output(arrays)
    class Buffer:
        count=0
        def __init__(self,a):
            Buffer.count+=1; self.native_handle=NS(value=Buffer.count); self.array=a.copy()
        def to_numpy(self): return self.array.copy()
    class Module:
        def entry_point(self,name): assert name=='main'; return self
    class Kernel:
        def __init__(self,delta=False): self.delta=delta
        def dispatch(self,*,thread_count,vars):
            assert thread_count==(1280,1,1) and vars['params']==m.PARAMS
            for name in ('out_0','out_1'): vars[name].array[:]=expected[name].reshape(-1)
            if self.delta: vars['out_0'].array[500]+=np.float32(1e-7)
    class Device:
        info=NS(limits=NS(max_compute_dispatch_thread_groups=NS(x=5),max_compute_thread_group_size=NS(x=256),max_compute_threads_per_group=256))
        def __init__(self,delta=False): self.ids=[]; self.delta=delta; self.waits=0
        def load_module_from_source(self,name,source): self.ids.append(name); return Module()
        def link_program(self,modules,entries): return self.ids[-1]
        def create_compute_kernel(self,program): return Kernel(self.delta and 'packaged' in program)
        def create_buffer(self,*,element_count,struct_size,usage,data): return Buffer(data)
        def wait_for_idle(self): self.waits+=1
    spy=NS(BufferUsage=NS(shader_resource=1,unordered_access=2))
    sources={'candidate':'old','packaged':'new'}
    device=Device(); result=m.collect(device,spy,arrays,sources,storage_limit=100000)
    assert set(result)==set(m.NAMES) and len(set(device.ids))==2 and device.waits==2
    device=Device(delta=True)
    with pytest.raises(AssertionError,match='paired output differs'):
        m.collect(device,spy,arrays,sources,storage_limit=100000)
    with pytest.raises(ValueError,match='device limits'):
        m.collect(Device(),spy,arrays,sources,storage_limit=64)


def test_device_receipt_rejects_mismatch():
    types=NS(VkInstance=1,VkPhysicalDevice=2,VkDevice=3)
    handles=[NS(type=i,value=i+10) for i in (1,2,3)]
    uuid=bytes(range(16))
    receipt=NS(finalized=True,device_uuid_known=True,instance_external=False,physical_device_external=False,
               device_external=False,device_uuid=uuid,driver_version=9,owner_generation=1,
               instance=handles[0],physical_device=handles[1],device=handles[2])
    device=NS(vulkan_creation_receipt=receipt,native_handles=handles)
    assert m.check_device(NS(NativeHandleType=types),device,uuid,9)['driver']==9
    with pytest.raises(ValueError): m.check_device(NS(NativeHandleType=types),device,uuid,10)
    receipt.device_external=True
    with pytest.raises(ValueError): m.check_device(NS(NativeHandleType=types),device,uuid,9)


def test_source_snapshot_uses_exact_committed_bytes_and_detects_mutation(monkeypatch):
    import hashlib
    payload={m.ADAPTER:b'adapter',m.CORE:b'core',m.CANDIDATE:b'candidate'}
    identities={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    def show(command):
        return payload[command[-1].split(':',1)[1]]
    monkeypatch.setattr(m.subprocess,'check_output',show)
    with m.source_snapshot('a'*40,identities) as (include,sources):
        assert sources=={'packaged':'adapter','candidate':'candidate'}
        core=include/'optimization/point_plane.slang'
        assert core.read_bytes()==b'core'
    assert not core.exists()
    with pytest.raises(ValueError,match='snapshot core changed'):
        with m.source_snapshot('a'*40,identities) as (include,sources):
            core=include/'optimization/point_plane.slang'; core.chmod(0o644); core.write_bytes(b'wrong')
    wrong=dict(identities); wrong[m.CORE]='0'*64
    with pytest.raises(ValueError,match='snapshot source mismatch'):
        with m.source_snapshot('a'*40,wrong): pass


def test_orchestration_rejects_uuid_before_dispatch_and_closes(monkeypatch,tmp_path):
    from contextlib import contextmanager
    import sys
    events=[]
    class Lease:
        def verify(self): events.append('verify')
    monkeypatch.setattr(m,'BorrowedLease',Lease)
    @contextmanager
    def lease():
        events.append('lock')
        try: yield Lease()
        finally: events.append('unlock')
    monkeypatch.setattr(m,'owned_lease',lease)
    monkeypatch.setattr(m,'closure',lambda commit: {'ok':'hash'})
    monkeypatch.setattr(m,'load_fixture',lambda: {})
    monkeypatch.setattr(m,'runtime_check',lambda *args: [])
    monkeypatch.setattr(m,'full_vulkan_storage',lambda uuid: {'max_storage_buffer_range':100000})
    @contextmanager
    def snapshot(*args): yield tmp_path, {'packaged':'p','candidate':'c'}
    monkeypatch.setattr(m,'source_snapshot',snapshot)
    class Device:
        @staticmethod
        def enumerate_adapters(*args): events.append('enumerate'); return [NS(vendor_id=10,device_id=20,luid='x')]
        def __init__(self,**kwargs): events.append('device')
        def close(self): events.append('close')
    spy=NS(Device=Device, DeviceType=NS(vulkan=1), SlangFloatingPointMode=NS(precise=1))
    monkeypatch.setattr(m.importlib,'import_module',lambda name: spy)
    def bad_identity(*args): events.append('identity'); raise ValueError('uuid mismatch')
    monkeypatch.setattr(m,'check_device',bad_identity)
    monkeypatch.setattr(m,'collect',lambda *args,**kwargs: events.append('dispatch'))
    for k in list(sys.modules):
        if k=='slangpy' or k.startswith('slangpy.'): monkeypatch.delitem(sys.modules,k)
    icd=tmp_path/'icd.json';icd.write_text('{}')
    monkeypatch.setenv('VK_DRIVER_FILES',str(icd));monkeypatch.delenv('VK_ICD_FILENAMES',raising=False)
    a=NS(execute=True, output=str(tmp_path/'output.json'),uuid=bytes(range(16)).hex(),driver=9,
         commit='a'*40,vendor=10,device=20)
    with pytest.raises(ValueError,match='uuid mismatch'): m.run(a,_cpu_test_only=True)
    assert 'dispatch' not in events and events.index('device') < events.index('close') < events.index('unlock')
    import json
    assert json.loads(Path(a.output).read_text())['status']=='failure'


def test_production_gate_precedes_runtime_and_device(monkeypatch,tmp_path):
    events=[]
    monkeypatch.setattr(m.importlib,'import_module',lambda *args: events.append('import'))
    monkeypatch.setattr(m,'owned_lease',lambda: events.append('lock'))
    with pytest.raises(RuntimeError,match='GPU execution blocked'):
        m.run(NS(execute=True,output=str(tmp_path/'receipt')))
    assert events==[] and not (tmp_path/'receipt').exists()
