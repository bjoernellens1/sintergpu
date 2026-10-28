# SPDX-License-Identifier: Apache-2.0
"""Compact stage model and opt-in GPU differential tests; no perf claim."""
import os
import numpy as np
import pytest
from compact_gpu_support import make_case, initial_state, model_stage, final_reference, verify_final

@pytest.mark.parametrize('n,capacity,width,action',[(0,3,1,0),(1,0,2,0),(255,5,1,4),(256,2,3,8),(257,4,8,0),(65537,3,2,8)])
def test_independent_stage_model_matches_full_reference(n,capacity,width,action):
    case=make_case(n,capacity,width,action)
    state=initial_state(case)
    for stage in case['plan']['stages']: model_stage(case,state,stage)
    verify_final(case,state,final_reference(case))


def test_input_tampering_is_rejected_by_stage_comparison():
    case=make_case(3,2,1,0); state=initial_state(case)
    old=state['mask_words'].copy(); state['mask_words'][0]^=1
    assert not np.array_equal(state['mask_words'],old)


def test_fake_runtime_uses_fresh_buffers_and_verifies_descriptor():
    import types
    from compact_gpu_runtime import GPUFixture
    events=[]
    class Buffer:
        counter=0
        def __init__(self,data,stride):
            Buffer.counter+=1; self.data=data.copy(); self.size=data.nbytes; self.struct_size=stride
            self.native_handle=types.SimpleNamespace(type=7,value=Buffer.counter)
        def to_numpy(self): return self.data.copy()
    class Device:
        def create_buffer(self,*,data,struct_size,**kwargs): return Buffer(data,struct_size)
        def load_module(self,path): return types.SimpleNamespace(entry_point=lambda x:x)
        def link_program(self,modules,entries): return entries[0]
        def create_compute_kernel(self,program): return object()
    spy=types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),NativeHandleType=types.SimpleNamespace(VkBuffer=7))
    case=make_case(3,2,1,0); fixture=GPUFixture(Device(),spy,case)
    assert len(set(fixture.handle_identity.values()))==len(fixture.buffers)
    assert fixture.buffers['shape_strides'].struct_size==8
    np.testing.assert_array_equal(fixture.buffers['shape_strides'].data[:len(case['plan']['shape_strides_u32'])],case['plan']['shape_strides_u32'])

@pytest.mark.parametrize('n,capacity,width,action',[(0,3,1,0),(1,0,2,0),(255,5,1,4),(257,4,8,0),(65537,3,2,8)])
def test_frozen_torch_donor_matches_stage_model(n,capacity,width,action):
    pytest.importorskip('torch')
    from compact_gpu_support import frozen_reference
    case=make_case(n,capacity,width,action); state=initial_state(case)
    for stage in case['plan']['stages']: model_stage(case,state,stage)
    verify_final(case,state,frozen_reference(case))


def test_runtime_rejects_duplicate_native_handles():
    import types
    from compact_gpu_runtime import GPUFixture
    class Buffer:
        def __init__(self,data,stride):
            self.data=data.copy(); self.size=data.nbytes; self.struct_size=stride
            self.native_handle=types.SimpleNamespace(type=7,value=42)
        def to_numpy(self): return self.data.copy()
    class Device:
        def create_buffer(self,*,data,struct_size,**kwargs): return Buffer(data,struct_size)
    spy=types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),NativeHandleType=types.SimpleNamespace(VkBuffer=7))
    with pytest.raises(ValueError,match='distinct VkBuffer'): GPUFixture(Device(),spy,make_case(3,2,1,0))


def test_runtime_rejects_corrupt_descriptor_upload():
    import types
    from compact_gpu_runtime import GPUFixture
    class Buffer:
        count=0
        def __init__(self,data,stride):
            Buffer.count+=1; self.data=data.copy(); self.size=data.nbytes; self.struct_size=stride
            self.native_handle=types.SimpleNamespace(type=7,value=Buffer.count)
            if stride==8: self.data[0]^=1
        def to_numpy(self): return self.data.copy()
    class Device:
        def create_buffer(self,*,data,struct_size,**kwargs): return Buffer(data,struct_size)
    spy=types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2),NativeHandleType=types.SimpleNamespace(VkBuffer=7))
    with pytest.raises(AssertionError,match='shape_strides'): GPUFixture(Device(),spy,make_case(3,2,1,0))


def test_runner_rejects_small_device_limits():
    import types
    from compact_gpu_runner import case_for_limits
    limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535),max_compute_thread_group_size=types.SimpleNamespace(x=256),max_compute_threads_per_group=256)
    with pytest.raises(ValueError): case_for_limits((257,4,8,0),limits,40)
    limits.max_compute_threads_per_group=128
    with pytest.raises(ValueError,match='workgroup'): case_for_limits((3,2,1,0),limits,1<<20)


def test_runner_requires_execute_before_identity_or_device(monkeypatch,tmp_path):
    import types
    import compact_gpu_runner as runner
    monkeypatch.setattr(runner,'closure',lambda _:pytest.fail('closure should not run'))
    with pytest.raises(ValueError,match='explicit --execute'):
        runner.run(types.SimpleNamespace(execute=False))


def test_comparison_detects_input_and_tail_corruption():
    import types
    from compact_gpu_runtime import GPUFixture
    case=make_case(3,2,1,0); expected=initial_state(case)
    class Buffer:
        def __init__(self,data): self.data=data.copy()
        def to_numpy(self): return self.data.copy()
    fixture=GPUFixture.__new__(GPUFixture)
    fixture.buffers={name:Buffer(data) for name,data in expected.items()}
    fixture.buffers['mask_words'].data[0]^=1
    with pytest.raises(AssertionError,match='mask_words'): fixture.compare_all(expected)
    fixture.buffers['mask_words'].data[:]=expected['mask_words']
    fixture.buffers['values_out'].data[-1]^=1
    with pytest.raises(AssertionError,match='values_out'): fixture.compare_all(expected)


def test_runner_loaded_module_shadow_is_rejected(monkeypatch,tmp_path):
    import compact_gpu_runner as runner
    import compact_gpu_support
    monkeypatch.setattr(runner,'verify_loaded_closure',lambda _:None)
    shadow=tmp_path/'compact_gpu_support.py'; shadow.write_text('shadow')
    monkeypatch.setattr(compact_gpu_support,'__file__',str(shadow))
    with pytest.raises(ValueError,match='outside committed root'): runner.verify_modules({})
