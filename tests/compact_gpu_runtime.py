# SPDX-License-Identifier: Apache-2.0
"""Opt-in test fixture: fresh buffers, guarded state, explicit host readback.

VkBuffer handle uniqueness and fresh create_buffer ownership are checked; this
cannot independently prove distinct physical VkDeviceMemory allocations/ranges.
"""
import numpy as np
from benchmarks.compact_graph_fixture import PreparedCompactGraph
from compact_gpu_support import GUARD, initial_state, model_stage, verify_final, final_reference, frozen_reference


class GPUFixture:
    def __init__(self,device,spy,case):
        self.device=device; self.case=case; self.initial=initial_state(case)
        plan=case['plan']; self.buffers={}; ranges={}; handles=[]
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        for name,data in self.initial.items():
            stride=8 if name=='shape_strides' else 4
            if data.nbytes%stride: raise ValueError('descriptor alignment mismatch')
            buffer=device.create_buffer(element_count=data.nbytes//stride,struct_size=stride,usage=usage,data=data)
            h=buffer.native_handle
            if h.type!=spy.NativeHandleType.VkBuffer or not h.value or h.value in handles: raise ValueError('distinct VkBuffer handles required')
            if buffer.size!=data.nbytes or buffer.struct_size!=stride: raise ValueError('buffer allocation size or stride mismatch')
            handles.append(h.value); self.buffers[name]=buffer
            ranges[name]={'allocation_id':'VkBuffer:'+str(h.value),'offset':0,'length':buffer.size}
        self.handle_identity={name:int(buf.native_handle.value) for name,buf in self.buffers.items()}
        # Explicit setup readback: descriptor and guards are verified before dispatch.
        for name,data in self.initial.items():
            actual=self.buffers[name].to_numpy().view(np.uint32).reshape(-1)
            np.testing.assert_array_equal(actual,data,err_msg='setup '+name)
        modules={}; kernels={}
        for stage in plan['stages']:
            source=stage['source']
            if source not in modules: modules[source]=device.load_module(str(__import__('pathlib').Path(__file__).resolve().parents[1]/source))
            key=(source,stage['entry'])
            if key not in kernels:
                mod=modules[source]; kernels[key]=device.create_compute_kernel(device.link_program([mod],[mod.entry_point(stage['entry'])]))
        self.graph=PreparedCompactGraph(plan,self.buffers,ranges,kernels)

    def compare_all(self,expected):
        observed={}
        for name,want in expected.items():
            actual=self.buffers[name].to_numpy().view(np.uint32).reshape(-1)
            np.testing.assert_array_equal(actual,want,err_msg=name)
            observed[name]=actual.copy()
        return observed

    def reset(self):
        for name,data in self.initial.items(): self.buffers[name].copy_from_numpy(data)
        self.device.wait_for_idle()

    def run_stagewise(self):
        expected={name:data.copy() for name,data in self.initial.items()}
        for stage,(kernel,threads,bindings) in zip(self.case['plan']['stages'],self.graph._dispatches):
            encoder=self.device.create_command_encoder(); encoder.global_barrier()
            kernel.dispatch(thread_count=threads,vars=bindings,command_encoder=encoder)
            self.device.submit_command_buffer(encoder.finish()); self.device.wait_for_idle()
            model_stage(self.case,expected,stage)
            self.compare_all(expected)
        actual=self.compare_all(expected)
        verify_final(self.case,actual,final_reference(self.case))
        donor=frozen_reference(self.case)
        if donor is not None: verify_final(self.case,actual,donor)
        return actual

    def run_graph(self):
        encoder=self.device.create_command_encoder(); self.graph.record(encoder)
        self.device.submit_command_buffer(encoder.finish()); self.device.wait_for_idle()
        expected={name:data.copy() for name,data in self.initial.items()}
        for stage in self.case['plan']['stages']: model_stage(self.case,expected,stage)
        actual=self.compare_all(expected); verify_final(self.case,actual,final_reference(self.case))
        donor=frozen_reference(self.case)
        if donor is not None: verify_final(self.case,actual,donor)
        return actual
