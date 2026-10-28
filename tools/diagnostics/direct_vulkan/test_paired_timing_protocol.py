# SPDX-License-Identifier: Apache-2.0
"""CPU fake protocol tests; no Vulkan or GPU."""
import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).with_name('paired_timing_protocol.py');s=importlib.util.spec_from_file_location('paired_timing',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Fake:
    def __init__(self):self.events=[];self.changed=False
    def contract(self):return {'old':{'shape':1025,'input_sha':'same'},'candidate':{'shape':1025,'input_sha':'same'},'timestamp_period_ns':2,'timestamp_valid_bits':32}
    def reset(self,v):self.events.append(('reset',v))
    def resident(self,v):self.events.append(('resident',v));return {'kernel_ticks':(10,20),'pipeline_ticks':(5,25),'completion_bound_ns':100,'outputs':{'all':b'x'}}
    def transfer(self,v):self.events.append(('transfer',v));return {'host_ns':90,'outputs':{'all':b'y' if self.changed else b'x'}}
    def validate_outputs(self,v,raw):self.events.append(('validate',v))

def test_full_order_counts_and_raw_samples():
    b=Fake();result=m.collect(b,lambda:{'device':'unverified'},lambda phase,outputs:None,samples=50,warmups=0)
    assert len(result['runs'])==3 and len(b.events)==3*50*3*6
    assert b.events[:6]==[('reset','old'),('resident','old'),('validate','old'),('reset','old'),('transfer','old'),('validate','old')]
    for run in result['runs']:
        assert len(run['order'])==150 and run['order'][:3]==['old_before','candidate','old_after']
        assert run['raw_ticks']['candidate'][0]=={'kernel':[10,20],'pipeline':[5,25],'completion_bound_ns':100}
        assert run['kernel_ns']['candidate'][0]==20 and run['resident_pipeline_ns']['candidate'][0]==40
    assert 'no verified backend' in result['scope']

def test_wrap_invalid_and_identity_mismatch():
    assert m.ticks_to_ns((255,1),2,8,10)==4
    for pair,bits in [((1,1),8),((0,256),8),((0,1),0)]:
        with pytest.raises(ValueError):m.ticks_to_ns(pair,1,bits,2)
    with pytest.raises(ValueError,match='wrap ambiguity'):m.ticks_to_ns((1,2),1,8,256)
    with pytest.raises(ValueError,match='completion bound'):m.ticks_to_ns((1,20),1,8,10)
    b=Fake();b.changed=True
    with pytest.raises(AssertionError,match='resident/transfer'):m.collect(b,lambda:{'a':1},lambda *a:None,warmups=0)
    b=Fake();b.contract=lambda:{'old':{'shape':1},'candidate':{'shape':2},'timestamp_period_ns':1,'timestamp_valid_bits':8}
    with pytest.raises(ValueError,match='contract'):m.collect(b,lambda:{'a':1},lambda *a:None)

def test_p95_and_median_detection():
    rows=[]
    for _ in range(3):
        row={metric:{'old_before':[100]*50,'candidate':[100]*50,'old_after':[100]*50} for metric in m.METRICS}
        rows.append(row)
    assert all(x['kernel_ns']['median_candidate_le_baseline'] and x['kernel_ns']['candidate_p95_le_105_percent_reference'] for x in m.assess(rows))
    rows[1]['kernel_ns']['candidate']=[110]*50
    result=m.assess(rows)[1]['kernel_ns'];assert not result['median_candidate_le_baseline'] and not result['candidate_p95_le_105_percent_reference']
