# SPDX-License-Identifier: Apache-2.0
"""Correctness-only Caspar layout probe. No physical-allocation proof or GPU timing.

All buffers are fresh/retained VkBuffers created on the caller's device. The
public API does not reveal physical VkDeviceMemory backing or offsets. Setup and
verification readbacks are fixture-only, not part of shader recording.
"""
import importlib.util
import hashlib
from pathlib import Path
import numpy as np
from sintergpu.caspar_layout_plan import plan_layout
from sintergpu.references import caspar_layout_numeric as independent

ROOT=Path(__file__).resolve().parents[1]
DONOR=ROOT/'references/donors/caspar-slang/src/caspar_slang/layout.py'
SHADER=ROOT/'src/sintergpu/kernels/experimental/caspar_layout.slang'
GUARD=8
POISON=np.uint32(0xa5b6c7d8)


def donor_module():
    if hashlib.sha256(DONOR.read_bytes()).hexdigest()!='d32325aaefaa37df324645683b2e2ff5dfaa39391b53b04bae46c034ff0d93c2': raise ValueError('donor hash mismatch')
    spec=importlib.util.spec_from_file_location('frozen_caspar_layout_gpu',DONOR)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def make_case(n,d,layout,zero_stride=False):
    row=d if not zero_stride else 0
    size=max(1+(max(n-1,0)*row if n else 0)+d,1)
    source=np.arange(size,dtype=np.uint32)*np.uint32(0x9e3779b9)
    specials=np.array([0x80000000,0x7fc12345,0x7f800001,0xff800000,0x7f800000],np.uint32)
    source[1:1+min(d,len(specials))]=specials[:min(d,len(specials))]
    view=np.ndarray((n,d),dtype=np.float32,buffer=source,offset=4,strides=(row*4,4))
    donor=donor_module(); a=independent.pack(view,layout); b=donor.pack(view,layout)
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    np.testing.assert_array_equal(independent.unpack(a,n,d,layout).view(np.uint32),donor.unpack(b,n,d,layout).view(np.uint32))
    return {'n':n,'d':d,'layout':layout,'row_stride':row*4,'source':source,'expected_packed':a.view(np.uint32).copy(),
            'expected_dense':view.copy().view(np.uint32).reshape(-1)}


class LayoutFixture:
    def __init__(self,device,spy,case,*,limits):
        self.device=device; self.case=case
        kwargs={'max_buffer_bytes':min(limits['max_buffer_bytes'],32<<20)-GUARD*4,
                'max_workgroups_x':limits['max_workgroups_x'],'max_threads_per_group':limits['max_threads_per_group'],
                'max_group_size_x':limits['max_group_size_x']}
        if kwargs['max_buffer_bytes']<4: raise ValueError('fixture buffer budget')
        n,d,layout=case['n'],case['d'],case['layout']; source=case['source']
        self.pack=plan_layout('pack',n,d,layout,source_base_bytes=4,source_storage_bytes=source.nbytes,
            row_stride_bytes=case['row_stride'],component_stride_bytes=4,**kwargs)
        packed_bytes=self.pack['buffer_bytes']['outputWords']
        self.unpack=plan_layout('unpack',n,d,layout,source_storage_bytes=packed_bytes,**kwargs)
        if self.pack['descriptor_u32']!=self.unpack['descriptor_u32']: raise ValueError('inconsistent descriptors')
        initial={}
        for name,size in [('source',self.pack['buffer_bytes']['sourceWords']),('descriptor',self.pack['buffer_bytes']['layoutWords']),
                          ('packed',packed_bytes),('unpacked',self.unpack['buffer_bytes']['outputWords'])]:
            initial[name]=np.full(size//4+GUARD,POISON,np.uint32)
        initial['source'][:len(source)]=source
        initial['descriptor'][:len(self.pack['descriptor_u32'])]=self.pack['descriptor_u32']
        self.initial=initial; self.buffers={}; handles=set(); usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        for name,data in initial.items():
            buf=device.create_buffer(element_count=len(data),struct_size=4,usage=usage,data=data)
            h=buf.native_handle
            if h.type!=spy.NativeHandleType.VkBuffer or not h.value or h.value in handles: raise ValueError('unique typed VkBuffer required')
            if buf.size!=data.nbytes or buf.struct_size!=4 or buf.size>limits['max_buffer_bytes']: raise ValueError('buffer range/stride')
            handles.add(h.value); self.buffers[name]=buf
        self.handles={name:int(buf.native_handle.value) for name,buf in self.buffers.items()}
        self.compare(initial)
        module=device.load_module(str(SHADER)); self.kernels={}
        for entry in ('packAosoa','unpackAosoa'):
            self.kernels[entry]=device.create_compute_kernel(device.link_program([module],[module.entry_point(entry)]))

    def compare(self,expected):
        for name,want in expected.items(): np.testing.assert_array_equal(self.buffers[name].to_numpy().view(np.uint32).reshape(-1),want,err_msg=name)

    def reset(self):
        for name,data in self.initial.items(): self.buffers[name].copy_from_numpy(data)
        self.device.wait_for_idle()

    def run(self):
        expected={name:data.copy() for name,data in self.initial.items()}
        for plan,source,target,reference in ((self.pack,'source','packed',self.case['expected_packed']),
                                               (self.unpack,'packed','unpacked',self.case['expected_dense'])):
            if plan['dispatch_required']:
                encoder=self.device.create_command_encoder(); encoder.global_barrier()
                self.kernels[plan['entry']].dispatch(thread_count=plan['thread_count'],
                    vars={'params':dict(plan['params']),'sourceWords':self.buffers[source],'layoutWords':self.buffers['descriptor'],'outputWords':self.buffers[target]},command_encoder=encoder)
                self.device.submit_command_buffer(encoder.finish()); self.device.wait_for_idle()
                expected[target][:len(reference)]=reference
            self.compare(expected)
        return {name:hashlib.sha256(self.buffers[name].to_numpy().tobytes()).hexdigest() for name in ('packed','unpacked')}
