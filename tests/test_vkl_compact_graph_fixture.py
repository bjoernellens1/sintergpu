# SPDX-License-Identifier: Apache-2.0
"""Fake-only compact graph fixture and alias tests; no GPU execution."""
import pytest
from benchmarks.compact_graph_fixture import PreparedCompactGraph
from test_vkl_compact_plan import make

class Buffer:
    def __init__(self,n): self.size=n; self.struct_size=4
class Kernel:
    def __init__(self,name,events): self.name=name; self.events=events
    def dispatch(self,**kwargs): self.events.append(('dispatch',self.name,kwargs))
class Encoder:
    def __init__(self,events): self.events=events
    def global_barrier(self): self.events.append(('barrier',))


def setup():
    p=make(); events=[]
    buffers={name:Buffer(desc['allocated_bytes']) for name,desc in p['resources'].items()}
    buffers['shape_strides'].struct_size=8
    ranges={name:{'allocation_id':name,'offset':0,'length':buffer.size} for name,buffer in buffers.items()}
    kernels={(s['source'],s['entry']):Kernel(s['entry'],events) for s in p['stages']}
    return p,events,buffers,ranges,kernels


def test_record_uses_prebound_resources_and_explicit_barriers():
    p,events,buffers,ranges,kernels=setup()
    graph=PreparedCompactGraph(p,buffers,ranges,kernels)
    enc=Encoder(events); graph.record(enc); graph.record(enc)
    expected=[x for s in p['stages'] for x in [('barrier',),('dispatch',s['entry'])]]*2
    assert [e if len(e)==1 else e[:2] for e in events]==expected
    scans=[e[2] for e in events if len(e)>1 and e[1] in ('scanBlocks','scanCarry')]
    assert scans and all(x['vars']['product']==0 and x['vars']['count']>0 for x in scans)
    assert all(e[2]['command_encoder'] is enc for e in events if len(e)>1)


def test_rejects_writable_overlap_and_undersized_buffers():
    p,events,buffers,ranges,kernels=setup()
    ranges['indices_out']={'allocation_id':'counts_out','offset':0,'length':buffers['indices_out'].size}
    with pytest.raises(ValueError,match='overlap'): PreparedCompactGraph(p,buffers,ranges,kernels)
    p,events,buffers,ranges,kernels=setup(); buffers['action_words'].size=0
    with pytest.raises(ValueError,match='size'): PreparedCompactGraph(p,buffers,ranges,kernels)


def test_read_only_input_overlap_allowed_but_writable_alias_fails():
    p,events,buffers,ranges,kernels=setup()
    ranges['mask_words']['allocation_id']=ranges['value_words']['allocation_id']
    PreparedCompactGraph(p,buffers,ranges,kernels)
    ranges['valid_out']['allocation_id']=ranges['value_words']['allocation_id']
    with pytest.raises(ValueError,match='overlap'): PreparedCompactGraph(p,buffers,ranges,kernels)


def test_candidate_uniforms_and_scan_roles_are_prebound():
    p,events,buffers,ranges,kernels=setup()
    graph=PreparedCompactGraph(p,buffers,ranges,kernels)
    graph.record(Encoder(events))
    calls=[event for event in events if len(event)>1]
    for stage,(_,name,kwargs) in zip(p['stages'],calls):
        assert name==stage['entry']
        for arg,label in stage['bindings'].items(): assert kwargs['vars'][arg] is buffers[label]
        if name in ('scanBlocks','scanCarry'):
            assert kwargs['vars']['count']==stage['params']['count']
            assert kwargs['vars']['product']==0
        else: assert kwargs['vars']['params']==stage['params']


def test_same_buffer_with_false_identity_is_rejected():
    p,events,buffers,ranges,kernels=setup()
    buffers['valid_out']=buffers['flags'] if 'flags' in buffers else buffers['values']
    ranges['valid_out']['length']=buffers['valid_out'].size
    with pytest.raises(ValueError,match='identity'): PreparedCompactGraph(p,buffers,ranges,kernels)


def test_disjoint_claimed_ranges_on_same_full_buffer_are_rejected():
    p,events,buffers,ranges,kernels=setup()
    shared=Buffer(4096)
    buffers['indices_out']=buffers['valid_out']=shared
    ranges['indices_out']={'allocation_id':'shared','offset':0,'length':1024}
    ranges['valid_out']={'allocation_id':'shared','offset':2048,'length':1024}
    with pytest.raises(ValueError,match='offset|shared|full buffer'):
        PreparedCompactGraph(p,buffers,ranges,kernels)


def test_nonzero_offset_and_short_claim_for_full_binding_rejected():
    p,events,buffers,ranges,kernels=setup()
    ranges['value_words']['offset']=4
    with pytest.raises(ValueError,match='offset'): PreparedCompactGraph(p,buffers,ranges,kernels)
    p,events,buffers,ranges,kernels=setup()
    buffers['value_words'].size+=4
    with pytest.raises(ValueError,match='full buffer'): PreparedCompactGraph(p,buffers,ranges,kernels)


def test_shape_descriptor_requires_eight_byte_struct_stride():
    p,events,buffers,ranges,kernels=setup()
    buffers['shape_strides'].struct_size=4
    with pytest.raises(ValueError,match='struct stride'): PreparedCompactGraph(p,buffers,ranges,kernels)


def test_same_writable_buffer_with_same_full_identity_is_rejected():
    p,events,buffers,ranges,kernels=setup()
    shared=Buffer(4096)
    for name in ('indices_out','valid_out'):
        buffers[name]=shared
        ranges[name]={'allocation_id':'shared','offset':0,'length':4096}
    with pytest.raises(ValueError,match='shared buffer'):
        PreparedCompactGraph(p,buffers,ranges,kernels)
