# SPDX-License-Identifier: Apache-2.0
"""CPU-only scheduling and limit tests; no GPU execution."""
import types
import numpy as np
import pytest
from benchmarks.depth_pyramid_resident_parity_runner import check_limits, compare_graphs


def limits(x=65535,threads=256,group=256):
    return types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=x,y=65535,z=65535), max_compute_threads_per_group=threads,max_compute_thread_group_size=types.SimpleNamespace(x=group,y=256,z=64))


def test_actual_dispatch_limits_and_resource_budget():
    plan=[{'thread_count':(257,1,1),'elements':{'depth':257,'depthOut':129}}]
    observed=check_limits(limits(),plan,max_fixture_bytes=2**20,max_storage_buffer_range=4096)
    assert observed['max_groups_x']==65535 and observed['max_storage_buffer_range']==4096
    with pytest.raises(ValueError): check_limits(limits(1),plan,max_fixture_bytes=2**20,max_storage_buffer_range=4096)
    with pytest.raises(ValueError): check_limits(limits(threads=128),plan,max_fixture_bytes=2**20,max_storage_buffer_range=4096)
    with pytest.raises(ValueError): check_limits(limits(),plan,max_fixture_bytes=256,max_storage_buffer_range=4096)


def test_record_submit_wait_validate_before_any_reset():
    events=[]
    class Encoder:
        def finish(self): events.append('finish'); return 'command'
    class Device:
        def create_command_encoder(self): events.append('encoder'); return Encoder()
        def submit_command_buffer(self,command): events.append('submit')
        def wait_for_idle(self): events.append('wait')
    class Buffer:
        def copy_from_numpy(self,x): events.append('reset')
    class Graph:
        def __init__(self):
            self.plans=[{'level':1}]; self.levels=[{'depthOut':Buffer()}]; self.output_initial=[{'depthOut':np.ones(1,np.float32)}]
        def record(self,encoder): events.append('record')
        def read_checked(self,fn): events.append('read'); fn(1,{'depthOut':np.ones(1,np.float32)})
    def validator(level,output): events.append('validate')
    compare_graphs(Device(),Graph(),Graph(),validator,rounds=2)
    first_reset=events.index('reset')
    assert events[:first_reset].count('read')==2 and events[:first_reset].count('validate')==2
    for n,e in enumerate(events):
        if e=='read': assert events[n-1]=='wait'


def test_validation_failure_does_not_reset_either_graph():
    events=[]
    class Encoder:
        def finish(self): return None
    class Device:
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self,cmd): events.append('submit')
        def wait_for_idle(self): events.append('wait')
    class Buffer:
        def copy_from_numpy(self,x): events.append('reset')
    class Graph:
        levels=[{'depthOut':Buffer()}]; output_initial=[{'depthOut':np.ones(1,np.float32)}]
        def record(self,e): events.append('record')
        def read_checked(self,fn): events.append('read'); fn(1,{'depthOut':np.ones(1,np.float32)})
    def reject(level,output): raise AssertionError('oracle mismatch')
    with pytest.raises(AssertionError,match='oracle mismatch'):
        compare_graphs(Device(),Graph(),Graph(),reject)
    assert 'reset' not in events


def test_paired_values_are_bitwise_not_tolerance_checked():
    class Encoder:
        def finish(self): return None
    class Device:
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self,cmd): pass
        def wait_for_idle(self): pass
    class Graph:
        def __init__(self,x): self.x=x
        def record(self,e): pass
        def read_checked(self,fn): fn(1,{'depthOut':np.array([self.x],np.float32)})
    with pytest.raises(AssertionError,match='paired output differs'):
        compare_graphs(Device(),Graph(0.),Graph(-0.),lambda *x:None,rounds=1)


def test_verified_closure_includes_executed_vkl_planner(monkeypatch,tmp_path):
    import hashlib
    import benchmarks.depth_pyramid_resident_parity_runner as runner
    assert 'src/sintergpu/vkl.py' in runner.EXTRA
    name='src/sintergpu/vkl.py'; path=tmp_path/name; path.parent.mkdir(parents=True); path.write_bytes(b'planner')
    monkeypatch.setattr(runner,'ROOT',tmp_path); monkeypatch.setattr(runner,'EXTRA',(name,))
    monkeypatch.setattr(runner,'committed_closure',lambda commit:{})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:b'planner')
    assert runner.closure('a'*40)[name]==hashlib.sha256(b'planner').hexdigest()
    path.write_bytes(b'mutated')
    with pytest.raises(ValueError,match='differs from committed'): runner.closure('a'*40)


def test_full_vulkan_storage_evidence_and_device_limit():
    from benchmarks.depth_pyramid_resident_parity_runner import parse_storage_range
    uuid='96d2dba0-95fd-5c00-e849-2e0610f24e06'
    raw=f'GPU0:\n deviceUUID = {uuid}\n maxStorageBufferRange = 4096\n'
    assert parse_storage_range(raw,uuid)==4096
    with pytest.raises(ValueError): parse_storage_range(raw,'00000000-0000-0000-0000-000000000000')
    with pytest.raises(ValueError): parse_storage_range(raw+' maxStorageBufferRange = 8192\n',uuid)
    with pytest.raises(ValueError): parse_storage_range(raw+'GPU1:\n',uuid)
    plan=[{'thread_count':(257,1,1),'elements':{'depth':257,'depthOut':129}}]
    with pytest.raises(ValueError,match='storage'): check_limits(limits(),plan,max_fixture_bytes=2**20,max_storage_buffer_range=1024)
    result=check_limits(limits(),plan,max_fixture_bytes=2**20,max_storage_buffer_range=4096)
    assert result['max_storage_buffer_range']==4096


def test_required_package_sources_and_loaded_module_provenance(tmp_path):
    import benchmarks.depth_pyramid_resident_parity_runner as runner
    assert {'src/sintergpu/__init__.py','src/sintergpu/data/VERSION'} <= set(runner.EXTRA)
    local=tmp_path/'local.py'; local.write_text('')
    runner.require_local_modules({'good':types.SimpleNamespace(__file__=str(local))},tmp_path)
    with pytest.raises(ValueError,match='outside committed root'):
        runner.require_local_modules({'bad':types.SimpleNamespace(__file__='/tmp/not-local.py')},tmp_path)
    with pytest.raises(ValueError,match='module file'):
        runner.require_local_modules({'bad':types.SimpleNamespace()},tmp_path)


@pytest.mark.parametrize('name',['src/sintergpu/__init__.py','src/sintergpu/data/VERSION'])
def test_package_source_mutation_rejected(name,monkeypatch,tmp_path):
    import benchmarks.depth_pyramid_resident_parity_runner as runner
    path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'initial')
    monkeypatch.setattr(runner,'ROOT',tmp_path); monkeypatch.setattr(runner,'EXTRA',(name,))
    monkeypatch.setattr(runner,'committed_closure',lambda _: {})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:b'initial')
    runner.closure('a'*40)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs from committed'): runner.closure('a'*40)


def test_full_vulkan_parser_rejects_missing_and_conflicting_identifiers():
    from benchmarks.depth_pyramid_resident_parity_runner import parse_storage_range
    uuid='96d2dba0-95fd-5c00-e849-2e0610f24e06'
    prefix=f'GPU0:\n deviceUUID = {uuid}\n'
    with pytest.raises(ValueError): parse_storage_range(prefix,uuid)
    with pytest.raises(ValueError): parse_storage_range(prefix+' maxStorageBufferRange = 0\n',uuid)
    with pytest.raises(ValueError): parse_storage_range(prefix+' deviceUUID = 00000000-0000-0000-0000-000000000000\n maxStorageBufferRange = 4096\n',uuid)
