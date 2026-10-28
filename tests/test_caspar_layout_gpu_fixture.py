# SPDX-License-Identifier: Apache-2.0
"""CPU fake runtime exercises fixture guards and failure boundaries only."""
import types
import numpy as np
import pytest
from caspar_layout_gpu_fixture import LayoutFixture,make_case

class Buffer:
    counter=0
    def __init__(self,data,stride):
        Buffer.counter+=1; self.data=data.copy(); self.size=data.nbytes; self.struct_size=stride
        self.native_handle=types.SimpleNamespace(type=7,value=Buffer.counter)
    def to_numpy(self): return self.data.copy()
    def copy_from_numpy(self,data): self.data[:]=data

class Kernel:
    def __init__(self,entry,events,corrupt=False): self.entry=entry; self.events=events; self.corrupt=corrupt
    def dispatch(self,*,thread_count,vars,command_encoder):
        self.events.append(('dispatch',self.entry)); p=vars['params']; desc=vars['layoutWords'].data
        source=vars['sourceWords'].data; output=vars['outputWords'].data
        for i in range(p['outputCount']):
            if self.entry=='packAosoa':
                lo=0; hi=p['chunkCount']
                while lo<hi:
                    mid=(lo+hi)//2
                    if desc[mid*7]<=i:lo=mid+1
                    else:hi=mid
                base=(lo-1)*7; start=int(desc[base]); width=int(desc[base+1]); live=int(desc[base+2]); lane=(i-start)%width
                value=0 if lane>=live else source[p['sourceBase']+((i-start)//width)*p['rowStride']+int(desc[base+3+lane])*p['componentStride']]
            else:
                row,component=divmod(i,p['components']); base=p['chunkCount']*7+component*3
                start,width,lane=map(int,desc[base:base+3]); value=source[p['sourceBase']+start+row*width+lane]
            output[i]=value
        if self.corrupt: output[-1]^=1

class Device:
    def __init__(self,corrupt=False): self.events=[]; self.paths=[]; self.corrupt=corrupt
    def create_buffer(self,*,data,struct_size,**kwargs): return Buffer(data,struct_size)
    def load_module(self,path): self.paths.append(path); return types.SimpleNamespace(entry_point=lambda x:x)
    def link_program(self,modules,entries): return entries[0]
    def create_compute_kernel(self,entry): return Kernel(entry,self.events,self.corrupt)
    def create_command_encoder(self):
        events=self.events
        class Encoder:
            def global_barrier(self): events.append(('barrier',))
            def finish(self): return object()
        return Encoder()
    def submit_command_buffer(self,x): self.events.append(('submit',))
    def wait_for_idle(self): self.events.append(('wait',))
spy=types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),NativeHandleType=types.SimpleNamespace(VkBuffer=7))
LIMITS={'max_buffer_bytes':1<<20,'max_workgroups_x':65535,'max_threads_per_group':256,'max_group_size_x':256}

@pytest.mark.parametrize('n,d,layout,zero',[(0,3,None,False),(1,3,None,False),(5,7,None,False),(5,5,((3,1,4),(2,0)),False),(3,3,None,True),(1,4096,tuple((i,) for i in reversed(range(4096))),False)])
def test_fake_differential_guards_and_reuse(n,d,layout,zero):
    case=make_case(n,d,layout,zero); device=Device(); fixture=LayoutFixture(device,spy,case,limits=LIMITS)
    assert len(device.paths)==1 and device.paths[0].endswith('kernels/experimental/caspar_layout.slang')
    fixture.run(); fixture.reset(); fixture.run()
    dispatches=[e for e in device.events if e[0]=='dispatch']
    assert len(dispatches)==(4 if n else 0)
    for i,e in enumerate(device.events):
        if e[0]=='dispatch': assert device.events[i-1]==('barrier',)


def test_corrupted_tail_guard_rejected():
    fixture=LayoutFixture(Device(corrupt=True),spy,make_case(2,3,None),limits=LIMITS)
    with pytest.raises(AssertionError,match='packed'): fixture.run()


def test_duplicated_handle_rejected():
    class BadDevice(Device):
        def create_buffer(self,**kw):
            buf=super().create_buffer(**kw); buf.native_handle.value=1; return buf
    with pytest.raises(ValueError,match='unique typed'): LayoutFixture(BadDevice(),spy,make_case(2,3,None),limits=LIMITS)


def test_runner_refuses_before_any_device_import(monkeypatch):
    import caspar_layout_gpu_runner as runner
    monkeypatch.setattr(runner,'closure',lambda commit:{})
    args=types.SimpleNamespace(execute=True,commit='0'*40)
    with pytest.raises(RuntimeError,match='UUID.*not independently verifiable'): runner.run(args)
    with pytest.raises(ValueError,match='explicit --execute'): runner.run(types.SimpleNamespace(execute=False))


def test_closure_rejects_shader_or_committed_source_mutation(monkeypatch,tmp_path):
    import caspar_layout_gpu_runner as runner
    contents={name:('source:'+name).encode() for name in runner.FILES}
    for name,data in contents.items():
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    commit='a'*40
    def git(args,text=False):
        if args[-2:]==['rev-parse','HEAD']: return commit+'\n'
        path=args[-1].split(':',1)[1]; return contents[path]
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.subprocess,'check_output',git)
    monkeypatch.setattr(runner,'sys',types.SimpleNamespace(modules={}))
    assert len(runner.closure(commit))==len(runner.FILES)
    (tmp_path/runner.SHADER).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='differs from commit'): runner.closure(commit)
