# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for caller-owned experimental recording."""
from copy import deepcopy
import pytest
from compact_gpu_support import make_case
from sintergpu.compact_record import PreparedCompact,FORWARD,SCAN

LIMITS={'max_workgroups_x':65535,'max_buffer_bytes':32<<20,'max_threads_per_group':256,'max_group_size_x':256}
class Buffer:
    def __init__(self,size,stride): self.size=size; self.struct_size=stride
    def to_numpy(self): pytest.fail('record performed readback')
    def copy_from_numpy(self,*args): pytest.fail('record performed upload')
class Kernel:
    def __init__(self,name,events): self.name=name; self.events=events
    def dispatch(self,**kwargs): self.events.append(('dispatch',self.name,kwargs))
class Encoder:
    def __init__(self,events): self.events=events
    def global_barrier(self): self.events.append(('barrier',))
    def finish(self): pytest.fail('record finished encoder')
class Adapter:
    allow_experimental=True
    def __init__(self,events): self.events=events
    def prepare(self,name): self.events.append(('prepare',name)); return Kernel(name,self.events)


def setup(n=257,capacity=3):
    plan=make_case(n,capacity,2,0)['plan']
    buffers={name:Buffer(desc['allocated_bytes'],8 if name=='shape_strides' else 4) for name,desc in plan['resources'].items()}
    ranges={name:{'allocation_id':name,'offset':0,'length':buf.size} for name,buf in buffers.items()}
    return plan,buffers,ranges


def prepare(plan,buffers,ranges,events,**overrides):
    ids={(SCAN if s['entry'] in ('scanBlocks','scanCarry') else FORWARD)+s['entry'] for s in plan['stages']}
    kw={'limits':LIMITS,'descriptor_words':plan['shape_strides_u32'],'allocations_verified':True,
        'descriptor_verified':True,'allow_experimental':True,'kernels':{name:Kernel(name,events) for name in ids},
        'prebound_identity_attested':True}
    kw.update(overrides)
    return PreparedCompact(plan,buffers,ranges,**kw)

@pytest.mark.parametrize('n,capacity',[(0,0),(0,3),(1,0),(257,3),(65537,3)])
def test_record_matches_each_plan_stage_and_barrier(n,capacity):
    plan,buffers,ranges=setup(n,capacity); events=[]
    graph=prepare(plan,buffers,ranges,events); events.clear()
    encoder=Encoder(events); graph.record(encoder)
    assert len(events)==2*len(plan['stages'])
    for i,stage in enumerate(plan['stages']):
        assert events[2*i]==('barrier',)
        _,name,kw=events[2*i+1]; assert name==(SCAN if stage['entry'] in ('scanBlocks','scanCarry') else FORWARD)+stage['entry']
        assert kw['thread_count']==stage['thread_count'] and kw['command_encoder'] is encoder
        for slot,role in stage['bindings'].items(): assert kw['vars'][slot] is buffers[role]
        if stage['entry'] in ('scanBlocks','scanCarry'):
            assert kw['vars']['count']==stage['params']['count'] and kw['vars']['product']==0
        else: assert kw['vars']['params']==stage['params']
    if n==65537: assert sum(s['entry']=='scanBlocks' for s in plan['stages'])>=3
    # Caller can record the prepared graph again without hidden setup or I/O.
    events.clear(); graph.record(encoder); assert all(e[0]!='prepare' for e in events)


def test_prebound_kernels_and_experimental_opt_in():
    plan,buffers,ranges=setup(); events=[]
    ids={(SCAN if s['entry'] in ('scanBlocks','scanCarry') else FORWARD)+s['entry'] for s in plan['stages']}
    kernels={name:Kernel(name,events) for name in ids}
    graph=prepare(plan,buffers,ranges,events,kernels=kernels); graph.record(Encoder(events))
    with pytest.raises(ValueError,match='experimental'): prepare(plan,buffers,ranges,[],allow_experimental=False)
    adapter=Adapter([]); adapter.allow_experimental=False
    with pytest.raises(ValueError,match='experimental'): prepare(plan,buffers,ranges,[],adapter=adapter,kernels=None)
    with pytest.raises(ValueError,match='missing prebound'): prepare(plan,buffers,ranges,[],kernels={})
    with pytest.raises(ValueError,match='caller attested'): prepare(plan,buffers,ranges,[],prebound_identity_attested=False)


def test_rejects_malformed_plan_descriptor_and_unverified_claims():
    plan,buffers,ranges=setup()
    bad=deepcopy(plan); bad['stages'][0]['thread_count']=(999,1,1)
    with pytest.raises(ValueError,match='canonical'): prepare(bad,buffers,ranges,[])
    with pytest.raises(ValueError,match='descriptor'): prepare(plan,buffers,ranges,[],descriptor_words=[0])
    with pytest.raises(ValueError,match='attestations'): prepare(plan,buffers,ranges,[],descriptor_verified=False)
    with pytest.raises(ValueError,match='attestations'): prepare(plan,buffers,ranges,[],allocations_verified=False)
    with pytest.raises(ValueError,match='device limits'): prepare(plan,buffers,ranges,[],limits={})


def test_buffer_alias_and_layout_rules():
    plan,buffers,ranges=setup()
    # Read/read sharing is permitted with a consistent caller-verified identity.
    buffers['mask_words']=buffers['value_words']; ranges['mask_words']=dict(ranges['value_words'])
    prepare(plan,buffers,ranges,[])
    ranges['mask_words']['allocation_id']='different'
    with pytest.raises(ValueError,match='inconsistent'): prepare(plan,buffers,ranges,[])
    plan,buffers,ranges=setup(); ranges['valid_out']['allocation_id']=ranges['indices_out']['allocation_id']
    with pytest.raises(ValueError,match='overlap'): prepare(plan,buffers,ranges,[])
    plan,buffers,ranges=setup(); buffers['valid_out']=buffers['indices_out']; ranges['valid_out']=dict(ranges['indices_out'])
    with pytest.raises(ValueError,match='overlap'): prepare(plan,buffers,ranges,[])
    plan,buffers,ranges=setup(); ranges['valid_out']['offset']=4
    with pytest.raises(ValueError,match='whole buffer'): prepare(plan,buffers,ranges,[])
    plan,buffers,ranges=setup(); buffers['shape_strides'].struct_size=4
    with pytest.raises(ValueError,match='stride'): prepare(plan,buffers,ranges,[])
    plan,buffers,ranges=setup(); buffers['valid_out'].size-=4; ranges['valid_out']['length']-=4
    with pytest.raises(ValueError,match='buffer size'): prepare(plan,buffers,ranges,[])


def test_all_prepared_ids_are_packaged_experimental_entries():
    from sintergpu.catalog import get_operation
    plan,buffers,ranges=setup(65537,3); events=[]
    graph=prepare(plan,buffers,ranges,events)
    for name in graph._prepared:
        assert get_operation(name)['qualification']['status']=='experimental'


def test_prepared_bindings_are_not_affected_by_caller_plan_mutation():
    plan,buffers,ranges=setup(); events=[]
    graph=prepare(plan,buffers,ranges,events)
    plan['stages'][0]['params']['capacity']=999
    events.clear(); graph.record(Encoder(events))
    assert events[1][2]['vars']['params']['capacity']==3


def test_real_slangpy_adapter_cache_source_and_runtime_verification(monkeypatch,tmp_path):
    pytest.importorskip('slangpy')
    import shutil
    from pathlib import Path
    import sintergpu.compact_record as rec
    import sintergpu.adapters.slangpy as adapter_mod
    from sintergpu import kernel_path as real_kernel_path
    plan,buffers,ranges=setup(3,2); events=[]
    operation_ids={(SCAN if s['entry'] in ('scanBlocks','scanCarry') else FORWARD)+s['entry'] for s in plan['stages']}
    sources={}
    for op in operation_ids:
        src=real_kernel_path(op); dst=tmp_path/src.parent.name/src.name
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst); sources[op]=dst
    monkeypatch.setattr(rec,'kernel_path',lambda op:sources[op])
    monkeypatch.setattr(adapter_mod,'kernel_path',lambda op:sources[op])
    class Device:
        def load_module(self,path):
            events.append(('load',path)); return type('Module',(),{'entry_point':lambda self,x:x})()
        def link_program(self,modules,entries): return entries[0]
        def create_compute_kernel(self,program): return Kernel(program,events)
    runtime=rec._runtime_hashes(); device=Device(); adapter=adapter_mod.SlangPyAdapter(device,allow_experimental=True)
    graph=prepare(plan,buffers,ranges,events,adapter=adapter,kernels=None,runtime_identity=runtime)
    events.clear()
    # Any filesystem hashing, opening or enumeration during record must fail.
    import builtins, os
    with monkeypatch.context() as mp:
        def forbidden(*args,**kwargs): pytest.fail('record touched filesystem')
        mp.setattr(builtins,'open',forbidden); mp.setattr(Path,'read_bytes',forbidden); mp.setattr(os,'scandir',forbidden)
        graph.record(Encoder(events))
    assert events[0]==('barrier',)
    events.clear(); stage=next(iter(operation_ids)); cached=adapter._kernels[stage]
    adapter._kernels[stage]=Kernel('foreign',events)
    with pytest.raises(ValueError,match='cache identity'): graph.record(Encoder(events))
    assert not events
    adapter._kernels[stage]=cached
    sources[stage].write_bytes(sources[stage].read_bytes()+b'\n// tamper')
    with pytest.raises(ValueError,match='hash mismatch|closure changed'): graph.verify_source_identity()
    assert not events
    fresh=adapter_mod.SlangPyAdapter(device,allow_experimental=True)
    with pytest.raises(ValueError,match='hash mismatch|closure changed'):
        prepare(plan,buffers,ranges,[],adapter=fresh,kernels=None,runtime_identity=runtime)
    assert not events
    # A fresh recorder cannot silently trust a previously cached compiled object.
    with pytest.raises(ValueError,match='cache must be empty'):
        prepare(plan,buffers,ranges,[],adapter=adapter,kernels=None,runtime_identity=runtime)


def test_real_adapter_runtime_mismatch_rejected_before_compilation(monkeypatch):
    pytest.importorskip('slangpy')
    import sintergpu.compact_record as rec
    from sintergpu.adapters.slangpy import SlangPyAdapter
    plan,buffers,ranges=setup(3,2); events=[]
    class Device:
        def load_module(self,path): pytest.fail('must reject before compilation')
    runtime=rec._runtime_hashes(); runtime['compiler_sha256']='0'*64
    with pytest.raises(ValueError,match='runtime hash mismatch'):
        prepare(plan,buffers,ranges,events,adapter=SlangPyAdapter(Device(),allow_experimental=True),kernels=None,runtime_identity=runtime)
