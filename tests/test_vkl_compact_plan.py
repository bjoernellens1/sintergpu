# SPDX-License-Identifier: Apache-2.0
"""CPU-only design/ABI contract tests; no dispatch implementation."""
import pytest
from sintergpu.compact_plan import plan_compact, value_address


def make(shape=(2,257,3),capacity=2,item_bytes=1,source_bytes=1544,strides=(771,3,1),**kwargs):
    args=dict(capacity=capacity,item_bytes=item_bytes,value_strides=strides,value_storage_bytes=source_bytes,
        mask_strides=(257,1),mask_storage_bytes=516,max_workgroups_x=65535,max_buffer_bytes=0xfffffffc,max_threads_per_group=256,max_group_size_x=256)
    args.update(kwargs)
    return plan_compact(shape,**args)


def test_plan_reuses_segmented_scan_and_exposes_ordered_barriers():
    plan=make()
    assert plan['status']=='design_only_no_gpu_implementation'
    entries=[s['entry'] for s in plan['stages']]
    assert entries[0]=='compactInit' and entries[1]=='compactFlags'
    assert 'scanBlocks' in entries and 'scanCarry' in entries
    assert entries[-3:]==['compactScatterMetadata','compactFinalizeCounts','compactGatherWords']
    assert all(s['barrier_before'] for s in plan['stages'])
    assert plan['outputs']['counts']['encoding']=='little_endian_i64_from_u32_words'
    assert plan['autograd']=='backward_required_before_differentiable_use'


def test_value_address_positive_and_zero_strides():
    plan=make()
    assert value_address(plan,1,3,2)==782
    broadcast=make(strides=(0,3,1),source_bytes=772)
    assert value_address(broadcast,1,3,2)==11


def test_zero_shapes_and_output_word_counts():
    p=plan_compact((2,0,5),capacity=3,item_bytes=1,value_strides=(0,5,1),value_storage_bytes=0,
        mask_strides=(0,1),mask_storage_bytes=0,max_workgroups_x=65535,max_buffer_bytes=0xfffffffc,max_threads_per_group=256,max_group_size_x=256)
    assert p['outputs']['values']['logical_bytes']==30 and p['outputs']['values']['buffer_bytes']==32
    assert 'compactFlags' not in [x['entry'] for x in p['stages']]
    assert p['outputs']['counts']['elements']==4
    q=plan_compact((0,5,3),capacity=0,item_bytes=1,value_strides=(15,3,1),value_storage_bytes=0,
        mask_strides=(5,1),mask_storage_bytes=0,max_workgroups_x=1,max_buffer_bytes=0xfffffffc,max_threads_per_group=256,max_group_size_x=256)
    assert q['stages']==[]


def test_storage_stride_overflow_and_capacity_fail_closed():
    with pytest.raises(ValueError): make(value_storage_bytes=1543)
    with pytest.raises(ValueError): make(strides=(771,-3,1))
    with pytest.raises(ValueError): make(capacity=-1)
    with pytest.raises(ValueError): plan_compact((65536,65536),capacity=0,item_bytes=1,value_strides=(1,1),value_storage_bytes=4,mask_strides=(1,1),mask_storage_bytes=4,max_workgroups_x=2**32,max_buffer_bytes=0xfffffffc,max_threads_per_group=256,max_group_size_x=256)


def test_actions_strides_and_byte_buffers_fail_closed():
    p=make(action_item_bytes=8,action_strides=(2056,8),action_storage_bytes=4112)
    assert p['outputs']['actions']['encoding']=='little_endian_i64_from_u32_words'
    assert p['outputs']['actions']['elements']==8
    assert p['inputs']['actions']['strides_bytes']==(2056,8)
    with pytest.raises(ValueError): make(action_item_bytes=8,action_strides=(2056,8),action_storage_bytes=4108)
    with pytest.raises(ValueError): make(action_item_bytes=2,action_strides=(514,2),action_storage_bytes=1028)
    with pytest.raises(ValueError): make(action_item_bytes=0,action_strides=(0,0))
    with pytest.raises(ValueError): make(max_workgroups_x=1)
    with pytest.raises(ValueError): make(max_buffer_bytes=1024)
    with pytest.raises(ValueError): make(max_threads_per_group=128)
    with pytest.raises(ValueError): make(max_group_size_x=128)


def test_unaligned_source_offsets_and_word_owned_cross_slot_packing():
    import numpy as np
    from sintergpu.references.compact_numeric import select_compact
    p=plan_compact((1,3,3),capacity=3,item_bytes=1,value_strides=(9,3,1),value_base_byte=1,value_storage_bytes=12,
        mask_strides=(3,1),mask_storage_bytes=4,max_workgroups_x=100,max_buffer_bytes=1024,max_threads_per_group=256,max_group_size_x=256)
    storage=bytes([99,1,2,3,4,5,6,7,8,9,0,0])
    selected=[0,2,None]; words=[]
    # Model one writer per output word, including bytes from multiple slots.
    for word in range(p['outputs']['values']['buffer_bytes']//4):
        acc=0
        for lane in range(4):
            dst=word*4+lane
            if dst>=p['outputs']['values']['logical_bytes']: continue
            slot,inner=divmod(dst,3)
            if selected[slot] is not None:
                address=value_address(p,0,selected[slot],inner)
                acc|=storage[address]<<(8*lane)
        words.append(acc)
    actual=np.array(words,dtype='<u4').tobytes()[:9]
    values=np.array([[[1,2,3],[4,5,6],[7,8,9]]],np.uint8)
    expected=select_compact(values,np.array([[1,0,1]],bool),capacity=3)['values'].tobytes()
    assert actual==expected
    assert words[-1]>>8==0  # unused bytes in final output word are zero


def test_empty_capacity_and_zero_trailing_elements():
    p=plan_compact((2,3,0,4),capacity=5,item_bytes=8,value_strides=(0,0,32,8),value_storage_bytes=0,
        mask_strides=(3,1),mask_storage_bytes=8,max_workgroups_x=100,max_buffer_bytes=4096,max_threads_per_group=256,max_group_size_x=256)
    assert p['outputs']['values']['logical_bytes']==0
    assert 'compactGatherWords' not in [s['entry'] for s in p['stages']]
    assert p['outputs']['indices']['elements']==20
    z=plan_compact((2,3),capacity=0,item_bytes=4,value_strides=(12,4),value_storage_bytes=24,
        mask_strides=(3,1),mask_storage_bytes=8,max_workgroups_x=100,max_buffer_bytes=4096,max_threads_per_group=256,max_group_size_x=256)
    assert [s['entry'] for s in z['stages']]==['compactInit']


def test_every_dispatch_is_padded_and_slot_map_is_explicit():
    p=make()
    assert p['scratch']['slot_to_source_u32_elements']==4
    for stage in p['stages']:
        assert stage['thread_count'][0]%256==0
        assert stage['thread_count'][0]>=stage['logical_count']
        assert stage['thread_count'][0]-stage['logical_count']<256


def test_scan_stage_uniforms_and_roles_match_actual_recursive_recipe():
    from sintergpu.primitives import scan_layout
    plan=make()
    _,passes=scan_layout(2*257)
    stages=[stage for stage in plan['stages'] if stage['entry'] in ('scanBlocks','scanCarry')]
    assert len(stages)==len(passes) and len(stages)>1
    for stage,expected in zip(stages,passes):
        assert stage['entry']==expected.entry
        assert stage['params']=={'count':expected.count,'product':0}
        assert stage['bindings']==expected.bindings
        assert stage['source']=='src/sintergpu/kernels/primitives/scan_uint.slang'
        assert stage['specialization']=='uint32_segmented_inclusive_sum'
    first=stages[0]
    assert first['bindings']['values']=='values'
    assert first['bindings']['segmentIds']=='segment_ids'
    assert first['bindings']['output']=='output'
