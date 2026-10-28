# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the profile probe; no hardware access."""
import types
import numpy as np
import pytest
from compact_gpu_support import make_case
from compact_backward_profile_gpu_fixture import ProfileFixture,profile_gradient,torch_gradient
from compact_backward_profile_gpu_runner import make_upstream,preflight_oracles
from test_compact_backward_gpu_fixture import FakeDevice, fake_forward, spy
from sintergpu.compact_backward_torch_cpu_profile import torch_cpu_profile_bits

class ProfileKernel:
    def __init__(self,entry,events,corrupt): self.entry=entry; self.events=events; self.corrupt=corrupt
    def dispatch(self,*,thread_count,vars,command_encoder):
        self.events.append(('dispatch',self.entry)); p=vars['params']; inverse=vars['inverse'].data
        if self.entry=='backwardProfileInit': inverse[:p['batches']*p['items']]=0xffffffff
        elif self.entry=='backwardProfileInverse':
            for slot,source in enumerate(vars['slotSource'].data[:p['batches']*p['capacity']]):
                if source!=0xffffffff: inverse[(slot//p['capacity'])*p['items']+source]=slot
        elif self.entry=='backwardProfileGather':
            out=vars['outputWords'].data.view(np.uint8); grad=vars['gradWords'].data.view(np.uint8)
            total=p['logicalBytes']; size=p['bytesPerItem']; comp=p['componentBytes']; out[:(total+3)//4*4]=0
            for start in range(0,total,comp):
                slot=inverse[start//size]
                if slot!=0xffffffff:
                    at=p['gradBase']+int(slot)*size+start%size
                    out[start:start+comp]=np.frombuffer(torch_cpu_profile_bits(grad[at:at+comp],comp),np.uint8)
            if self.corrupt: out[-1]^=1
        else: raise AssertionError('unexpected entry')

class Device(FakeDevice):
    def __init__(self,corrupt=False): super().__init__(corrupt); self.paths=[]
    def load_module(self,path):
        self.paths.append(path); return types.SimpleNamespace(entry_point=lambda x:x)
    def create_compute_kernel(self,program): return ProfileKernel(program,self.events,self.corrupt)

@pytest.mark.parametrize('dtype,width',[('float16',2),('float32',4),('float64',8),('complex64',8),('complex128',16)])
def test_torch_preflight_special_bits(dtype,width):
    pytest.importorskip('torch'); case=make_case(7,5,width,0); upstream=make_upstream(case,dtype)
    np.testing.assert_array_equal(profile_gradient(case,upstream,dtype),torch_gradient(case,upstream,dtype))


def test_complete_profile_preflight():
    pytest.importorskip('torch'); preflight_oracles()

@pytest.mark.parametrize('dtype,width',[('float16',2),('float32',4),('float64',8),('complex64',8),('complex128',16)])
def test_actual_forward_binding_and_profile_stage_sequence(dtype,width):
    pytest.importorskip('torch'); case=make_case(7,5,width,0); fwd,state=fake_forward(case); dev=Device()
    fixture=ProfileFixture(dev,spy,fwd,make_upstream(case,dtype),dtype,max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    assert fixture.slot is fwd.buffers['slot_to_source']
    assert len(dev.paths)==1 and dev.paths[0].endswith('vkl_compact_backward_torch_cpu_profile.slang')
    digest=fixture.run(state,dtype); fixture.reset(); assert fixture.run(state,dtype)==digest
    assert [e[1] for e in dev.events if e[0]=='dispatch']==[s['entry'] for s in fixture.plan['stages']]*2
    for i,e in enumerate(dev.events):
        if e[0]=='dispatch': assert dev.events[i-1]==('barrier',)


def test_bad_slot_map_and_output_guard_fail_closed():
    pytest.importorskip('torch'); case=make_case(7,3,4,0); fwd,state=fake_forward(case); dev=Device(corrupt=True)
    fixture=ProfileFixture(dev,spy,fwd,make_upstream(case,'float32'),'float32',max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    bad={k:v.copy() for k,v in state.items()}; bad['slot_to_source'][0]=0xffffffff
    with pytest.raises(AssertionError): fixture.run(bad,'float32')
    assert not any(e[0]=='dispatch' for e in dev.events)
    with pytest.raises(AssertionError,match='backwardProfileGather outputWords'): fixture.run(state,'float32')


def test_runner_explicit_opt_in_and_source_shadow(monkeypatch):
    import compact_backward_profile_gpu_runner as runner
    monkeypatch.setattr(runner,'closure',lambda _:pytest.fail('closure should not run'))
    with pytest.raises(ValueError,match='explicit --execute'): runner.run(types.SimpleNamespace(execute=False))


def test_loaded_profile_module_shadow_rejected(monkeypatch,tmp_path):
    import compact_backward_profile_gpu_runner as runner
    import compact_backward_profile_gpu_fixture as fixture
    monkeypatch.setattr(runner,'verify_backward',lambda _:None)
    shadow=tmp_path/'fixture.py'; shadow.write_text('shadow')
    monkeypatch.setattr(fixture,'__file__',str(shadow))
    with pytest.raises(ValueError,match='outside committed root'): runner.verify_modules({})


def test_profile_plan_dtype_mismatch_rejected():
    case=make_case(7,3,4,0); fwd,_=fake_forward(case); dev=Device()
    with pytest.raises(ValueError,match='dtype'):
        ProfileFixture(dev,spy,fwd,make_upstream(case,'float32'),'float64',max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    assert not dev.paths


def test_odd_fp16_partial_word_and_poisoned_guard_against_donor():
    pytest.importorskip('torch')
    from compact_backward_profile_gpu_runner import odd_fp16_case,odd_forward_reference
    from compact_gpu_support import POISON
    case=odd_fp16_case(); assert case['shape']==(1,3,1) and case['plan']['outputs']['values']['logical_bytes']==6
    assert odd_forward_reference(case)['values'].nbytes==6
    fwd,state=fake_forward(case); dev=Device(); upstream=make_upstream(case,'float16')
    np.testing.assert_array_equal(profile_gradient(case,upstream,'float16'),torch_gradient(case,upstream,'float16'))
    fixture=ProfileFixture(dev,spy,fwd,upstream,'float16',max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    fixture.run(state,'float16')
    raw=fixture.buffers['outputWords'].data.view(np.uint8)
    assert bytes(raw[6:8])==b'\0\0'
    np.testing.assert_array_equal(fixture.buffers['outputWords'].data[2:],np.full(8,POISON,np.uint32))


def test_odd_forward_stage_model_and_exact_partial_word():
    from compact_backward_profile_gpu_runner import odd_fp16_case, odd_forward_reference
    from compact_gpu_support import initial_state,model_stage,verify_final,POISON
    case=odd_fp16_case(); state=initial_state(case); reference=odd_forward_reference(case)
    for stage in case['plan']['stages']:
        if stage['entry']=='compactGatherWords':
            raw=state['values_out'].view(np.uint8); expected=reference['values'].tobytes()
            raw[:8]=0; raw[:6]=np.frombuffer(expected,np.uint8)
        else: model_stage(case,state,stage)
    verify_final(case,state,reference)
    assert bytes(state['values_out'].view(np.uint8)[6:8])==b'\0\0'
    np.testing.assert_array_equal(state['values_out'][2:],np.full(8,POISON,np.uint32))
