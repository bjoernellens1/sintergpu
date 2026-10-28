# SPDX-License-Identifier: Apache-2.0
"""Correctness-only opt-in probe: actual forward producer feeds backward.

Fixture deliberately reads back after forward and after each backward pass to
validate inputs and guards. Production recording need not perform these reads.
Fresh VkBuffer handles do not prove physical memory allocation independence.
"""
import importlib.util
from pathlib import Path
import hashlib
import numpy as np
from sintergpu.compact_backward_plan import SENTINEL
from sintergpu.compact_backward_torch_cpu_profile import plan_torch_cpu_profile, torch_cpu_profile_bits, DTYPES
from sintergpu.references.compact_backward import compact_value_backward
from compact_gpu_support import GUARD, POISON

ROOT=Path(__file__).resolve().parents[1]
SHADER='benchmarks/shaders/vkl_compact_backward_torch_cpu_profile.slang'
DONOR=ROOT/'references/donors/vkl/src/vkl/kernels/reference/compact.py'


def expected_slots(case):
    b,n,_=case['shape']; cap=case['capacity']; result=np.full(b*cap,SENTINEL,np.uint32)
    for batch in range(b):
        rank=0
        for item in range(n):
            if case['mask_bytes'][1+batch*n+item] and rank<cap:
                result[batch*cap+rank]=item; rank+=1
    return result


def validate_slots(case,actual):
    b,n,_=case['shape']; cap=case['capacity']; expected=expected_slots(case)
    slots=actual[:b*cap]
    np.testing.assert_array_equal(slots,expected,err_msg='forward slot map differs from oracle')
    for batch in range(b):
        row=slots[batch*cap:(batch+1)*cap]; live=row[row!=SENTINEL]
        if np.any(live>=n) or len(set(map(int,live)))!=len(live): raise AssertionError('invalid forward slot map')
    return expected


def independent_gradient(case,upstream):
    b,n,tail=case['shape']; cap=case['capacity']; width=case['width']
    mask=np.ndarray((b,n),dtype=np.bool_,buffer=case['mask_bytes'],offset=1,strides=(n,1))
    return compact_value_backward((b,n,tail),mask,upstream).view(np.uint8).reshape(-1).copy()


def torch_gradient(case,upstream,dtype):
    import torch
    if hashlib.sha256(DONOR.read_bytes()).hexdigest()!='f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff': raise ValueError('donor hash mismatch')
    spec=importlib.util.spec_from_file_location('compact_backward_frozen_gpu',DONOR)
    donor=importlib.util.module_from_spec(spec); spec.loader.exec_module(donor)
    b,n,_=case['shape']; mask=np.ndarray((b,n),dtype=np.bool_,buffer=case['mask_bytes'],offset=1,strides=(n,1)).copy()
    values=torch.zeros(case['shape'],dtype=getattr(torch,dtype),requires_grad=True)
    donor.select_compact(values,torch.from_numpy(mask),capacity=case['capacity'])['values'].backward(torch.from_numpy(upstream))
    return values.grad.numpy().view(np.uint8).reshape(-1).copy()


def profile_gradient(case,upstream,dtype):
    logical=independent_gradient(case,upstream)
    return np.frombuffer(torch_cpu_profile_bits(logical,DTYPES[dtype][1]),np.uint8).copy()


class ProfileFixture:
    def __init__(self,device,spy,forward,upstream,dtype,*,max_groups,storage_limit,max_threads,max_group_size):
        self.device=device; self.forward=forward; case=forward.case; b,n,tail=case['shape']; cap=case['capacity']; width=case['width']
        if upstream.shape!=(b,cap,tail) or upstream.dtype.itemsize!=width or not upstream.flags.c_contiguous: raise ValueError('upstream shape/dtype/layout')
        if dtype not in DTYPES or DTYPES[dtype][0]!=width or upstream.dtype!=np.dtype(dtype): raise ValueError('profile dtype mismatch')
        self.dtype=dtype
        self.upstream=upstream; raw=upstream.tobytes(); size=((1+len(raw)+3)//4)*4
        limit=min(storage_limit,32<<20)-GUARD*4
        self.plan=plan_torch_cpu_profile(case['shape'],cap,dtype,grad_base=1,grad_storage_bytes=size,max_buffer_bytes=limit,
            max_workgroups_x=max_groups,max_threads_per_group=max_threads,max_group_size_x=max_group_size)
        # The actual produced slot buffer, not an independently populated copy.
        self.slot=forward.buffers['slot_to_source']; self.buffers={}; self.initial={}
        for name in ('gradWords','inverse','outputWords'):
            words=self.plan['buffers'][name]//4+GUARD
            self.initial[name]=np.full(words,POISON,np.uint32)
        self.initial['gradWords'].view(np.uint8)[1:1+len(raw)]=np.frombuffer(raw,np.uint8)
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        handles=set(forward.handle_identity.values())
        for name,data in self.initial.items():
            buf=device.create_buffer(element_count=len(data),struct_size=4,usage=usage,data=data)
            h=buf.native_handle
            if h.type!=spy.NativeHandleType.VkBuffer or not h.value or h.value in handles: raise ValueError('duplicate or invalid VkBuffer')
            if buf.size!=data.nbytes or buf.struct_size!=4 or buf.size>storage_limit: raise ValueError('invalid allocation range')
            handles.add(h.value); self.buffers[name]=buf
            np.testing.assert_array_equal(buf.to_numpy().view(np.uint32).reshape(-1),data,err_msg='setup '+name)
        self.handles={name:int(buf.native_handle.value) for name,buf in self.buffers.items()}
        mod=device.load_module(str(ROOT/SHADER)); self.kernels={}
        for stage in self.plan['stages']:
            entry=stage['entry']; self.kernels[entry]=device.create_compute_kernel(device.link_program([mod],[mod.entry_point(entry)]))

    def reset(self):
        for name,data in self.initial.items(): self.buffers[name].copy_from_numpy(data)
        self.device.wait_for_idle()

    def run(self,forward_state,dtype):
        if dtype!=self.dtype: raise ValueError('profile dtype mismatch')
        case=self.forward.case; b,n,_=case['shape']; cap=case['capacity']; slots_expected=validate_slots(case,forward_state['slot_to_source'])
        # Establish that the very buffer bound below still contains the validated map.
        np.testing.assert_array_equal(self.slot.to_numpy().view(np.uint32).reshape(-1),forward_state['slot_to_source'])
        expected={k:v.copy() for k,v in self.initial.items()}; inverse=expected['inverse']; params=self.plan['params']
        for stage in self.plan['stages']:
            entry=stage['entry']; bindings={'params':dict(params),'inverse':self.buffers['inverse']}
            if entry=='backwardProfileInverse': bindings['slotSource']=self.slot
            if entry=='backwardProfileGather': bindings.update(gradWords=self.buffers['gradWords'],outputWords=self.buffers['outputWords'])
            enc=self.device.create_command_encoder(); enc.global_barrier()
            self.kernels[entry].dispatch(thread_count=stage['thread_count'],vars=bindings,command_encoder=enc)
            self.device.submit_command_buffer(enc.finish()); self.device.wait_for_idle()
            if entry=='backwardProfileInit': inverse[:b*n]=SENTINEL
            elif entry=='backwardProfileInverse':
                for slot,source in enumerate(slots_expected):
                    if source!=SENTINEL: inverse[(slot//cap)*n+int(source)]=slot
            else:
                logical=profile_gradient(case,self.upstream,dtype)
                out=expected['outputWords'].view(np.uint8); words=(len(logical)+3)//4*4
                out[:words]=0; out[:len(logical)]=logical
            for name,want in expected.items(): np.testing.assert_array_equal(self.buffers[name].to_numpy().view(np.uint32).reshape(-1),want,err_msg=entry+' '+name)
            self.forward.compare_all(forward_state)
        logical=profile_gradient(case,self.upstream,dtype)
        actual=self.buffers['outputWords'].to_numpy().view(np.uint8)[:len(logical)]
        np.testing.assert_array_equal(actual,logical)
        np.testing.assert_array_equal(actual,torch_gradient(case,self.upstream,dtype))
        return hashlib.sha256(actual.tobytes()).hexdigest()
