# SPDX-License-Identifier: Apache-2.0
"""CPU-only harness contract tests. Never instantiate an actual GPU device."""
import types
import numpy as np
import pytest
from compact_gpu_support import make_case,initial_state
from compact_backward_gpu_fixture import BackwardFixture,expected_slots,validate_slots,independent_gradient,torch_gradient
from compact_backward_gpu_runner import make_upstream

@pytest.mark.parametrize('n,cap,width,dtype',[(0,3,2,'float16'),(1,0,4,'float32'),(5,4,8,'float64'),(3,2,8,'complex64'),(4,3,16,'complex128')])
def test_logical_oracle_matches_frozen_donor(n,cap,width,dtype):
    pytest.importorskip('torch')
    case=make_case(n,cap,width,0); upstream=make_upstream(case,dtype)
    np.testing.assert_array_equal(independent_gradient(case,upstream),torch_gradient(case,upstream,dtype))


def test_slot_validation_rejects_wrong_range_duplicate_and_order():
    case=make_case(5,4,4,0); expected=expected_slots(case); validate_slots(case,expected)
    for mutate in ('range','duplicate','order'):
        a=expected.copy()
        if mutate=='range': a[0]=5
        elif mutate=='duplicate': a[1]=a[0]
        else: a[0],a[1]=a[1],a[0]
        with pytest.raises(AssertionError): validate_slots(case,a)


class FakeBuffer:
    next_id=1
    def __init__(self,data,stride):
        self.data=data.copy(); self.size=data.nbytes; self.struct_size=stride
        self.native_handle=types.SimpleNamespace(type=7,value=FakeBuffer.next_id); FakeBuffer.next_id+=1
    def to_numpy(self): return self.data.copy()
    def copy_from_numpy(self,data): self.data[:]=data

class FakeKernel:
    def __init__(self,entry,events,corrupt=False): self.entry=entry; self.events=events; self.corrupt=corrupt
    def dispatch(self,*,thread_count,vars,command_encoder):
        self.events.append(('dispatch',self.entry)); p=vars['params']; inverse=vars['inverse'].data
        if self.entry=='backwardInit': inverse[:p['batches']*p['items']]=0xffffffff
        elif self.entry=='backwardInverse':
            for slot,source in enumerate(vars['slotSource'].data[:p['batches']*p['capacity']]):
                if source!=0xffffffff: inverse[(slot//p['capacity'])*p['items']+source]=slot
        else:
            out=vars['outputWords'].data.view(np.uint8); grad=vars['gradWords'].data.view(np.uint8)
            total=p['logicalBytes']; size=p['bytesPerItem']; out[:(total+3)//4*4]=0
            for i in range(total):
                slot=inverse[i//size]
                if slot!=0xffffffff: out[i]=grad[p['gradBase']+int(slot)*size+i%size]
            if self.corrupt: out[-1]^=1

class FakeDevice:
    def __init__(self,corrupt=False): self.events=[]; self.corrupt=corrupt
    def create_buffer(self,*,data,struct_size,**kwargs): return FakeBuffer(data,struct_size)
    def load_module(self,path): return types.SimpleNamespace(entry_point=lambda x:x)
    def link_program(self,modules,entries): return entries[0]
    def create_compute_kernel(self,program): return FakeKernel(program,self.events,self.corrupt)
    def create_command_encoder(self):
        events=self.events
        class Encoder:
            def global_barrier(self): events.append(('barrier',))
            def finish(self): return object()
        return Encoder()
    def submit_command_buffer(self,x): self.events.append(('submit',))
    def wait_for_idle(self): self.events.append(('wait',))

spy=types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),NativeHandleType=types.SimpleNamespace(VkBuffer=7))

def fake_forward(case):
    state=initial_state(case); slots=expected_slots(case); state['slot_to_source'][:len(slots)]=slots
    buffers={name:FakeBuffer(value,4) for name,value in state.items()}
    class Forward:
        handle_identity={name:buf.native_handle.value for name,buf in buffers.items()}
        def compare_all(self,expected):
            for name,value in expected.items(): np.testing.assert_array_equal(buffers[name].data,value)
    f=Forward(); f.case=case; f.buffers=buffers
    return f,state


def test_fake_stage_order_forward_binding_guards_and_reuse():
    pytest.importorskip('torch'); case=make_case(7,3,8,0); forward,state=fake_forward(case); device=FakeDevice()
    f=BackwardFixture(device,spy,forward,make_upstream(case,'float64'),max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    assert f.slot is forward.buffers['slot_to_source']
    result=f.run(state,'float64'); f.reset(); assert f.run(state,'float64')==result
    assert [e for e in device.events if e[0]=='dispatch']==[( 'dispatch',s['entry']) for s in f.plan['stages']]*2
    for i,event in enumerate(device.events):
        if event[0]=='dispatch': assert device.events[i-1]==('barrier',)


def test_fake_corrupt_guard_is_rejected():
    pytest.importorskip('torch'); case=make_case(7,3,8,0); forward,state=fake_forward(case); device=FakeDevice(corrupt=True)
    f=BackwardFixture(device,spy,forward,make_upstream(case,'float64'),max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    with pytest.raises(AssertionError,match='backwardGather outputWords'): f.run(state,'float64')


def test_forward_buffer_mismatch_rejected_before_dispatch():
    case=make_case(7,3,8,0); forward,state=fake_forward(case); device=FakeDevice()
    f=BackwardFixture(device,spy,forward,make_upstream(case,'float64'),max_groups=65535,storage_limit=1<<20,max_threads=256,max_group_size=256)
    forward.buffers['slot_to_source'].data[0]=0xffffffff
    with pytest.raises(AssertionError): f.run(state,'float64')
    assert not any(e[0]=='dispatch' for e in device.events)


def test_float32_signed_zero_donor_discrepancy_blocks_preflight():
    pytest.importorskip('torch')
    from compact_backward_gpu_runner import preflight_oracles
    case=make_case(7,3,4,0); upstream=make_upstream(case,'float32')
    expected=independent_gradient(case,upstream); donor=torch_gradient(case,upstream,'float32')
    assert not np.array_equal(expected,donor)
    assert expected[11]==128 and donor[11]==0
    with pytest.raises(AssertionError,match='bitwise donor mismatch'): preflight_oracles()


def test_torch_index_backward_reproduced_by_zero_scatter_add_for_sample():
    torch=pytest.importorskip('torch')
    case=make_case(7,3,4,0); upstream=make_upstream(case,'float32')
    batches=[]; sources=[]; gradients=[]
    for batch in range(2):
        selected=[i for i in range(7) if case['mask_bytes'][1+batch*7+i]][:3]
        for rank,item in enumerate(selected):
            batches.append(batch); sources.append(item); gradients.append(upstream[batch,rank])
    out=torch.zeros(case['shape'],dtype=torch.float32)
    out.index_put_((torch.tensor(batches),torch.tensor(sources)),torch.from_numpy(np.asarray(gradients)),accumulate=True)
    np.testing.assert_array_equal(out.numpy().view(np.uint8).reshape(-1),torch_gradient(case,upstream,'float32'))
