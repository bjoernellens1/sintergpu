# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy compaction compared with unchanged frozen VKL CPU donor."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sintergpu.references.compact_numeric import select_compact

ROOT=Path(__file__).resolve().parents[1]
DONOR=ROOT/'references/donors/vkl/src/vkl/kernels/reference/compact.py'
DONOR_SHA='f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff'

@pytest.fixture(scope='module')
def donor():
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest()==DONOR_SHA
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('frozen_vkl_compact',DONOR)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return torch,module.select_compact


def evaluate(donor,values,mask,codes,capacity):
    torch,fn=donor
    tensors=[torch.from_numpy(x.copy()) if x is not None else None for x in (values,mask,codes)]
    result=fn(tensors[0],tensors[1],action_codes=tensors[2],capacity=capacity)
    actual=select_compact(values,mask,action_codes=codes,capacity=capacity)
    assert set(actual)==set(result)
    for key,value in result.items():
        expected=value.numpy()
        assert actual[key].dtype==expected.dtype and actual[key].shape==expected.shape
        np.testing.assert_array_equal(actual[key],expected,strict=True)


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64])
@pytest.mark.parametrize('tail',[(),(1,),(2,3),(0,2)])
@pytest.mark.parametrize('capacity',[0,1,3,7])
def test_donor_payload_types_shapes_capacity(donor,dtype,tail,capacity):
    shape=(2,5,*tail); n=int(np.prod(shape)); values=(np.arange(n).reshape(shape)%5).astype(dtype)
    mask=np.array([[0,1,1,1,0],[1,0,1,0,1]],bool)
    codes=np.array([[0,-4,5,6,0],[7,0,8,0,9]],np.int64)
    evaluate(donor,values,mask,codes,capacity)


@pytest.mark.parametrize('batch,count,capacity',[(0,0,0),(0,3,4),(2,0,0),(2,0,4),(1,3,0),(1,3,5)])
@pytest.mark.parametrize('code_dtype',[None,np.int32,np.int64])
def test_empty_dimensions_and_optional_actions(donor,batch,count,capacity,code_dtype):
    values=np.arange(batch*count*2,dtype=np.float32).reshape(batch,count,2)
    mask=np.ones((batch,count),bool)
    codes=None if code_dtype is None else np.arange(batch*count,dtype=code_dtype).reshape(batch,count)
    evaluate(donor,values,mask,codes,capacity)


def test_noncontiguous_inputs_and_nan_payload(donor):
    values=np.arange(48,dtype=np.float32).reshape(2,4,6)[:,:,::2]
    values[0,1,0]=np.nan
    mask=np.array([[1,0,1,1],[0,1,0,1]],bool)
    codes=np.arange(16,dtype=np.int32).reshape(2,8)[:,::2]
    evaluate(donor,values,mask,codes,3)


@pytest.mark.parametrize('kind',['values_rank','mask_shape','mask_dtype','negative_capacity','codes_shape','codes_dtype'])
def test_rejects_invalid_inputs_like_donor(donor,kind):
    torch,fn=donor
    values=np.ones((2,3),np.float32); mask=np.ones((2,3),bool); codes=np.ones((2,3),np.int32); capacity=2
    if kind=='values_rank': values=values[0]
    if kind=='mask_shape': mask=mask[0]
    if kind=='mask_dtype': mask=mask.astype(np.uint8)
    if kind=='negative_capacity': capacity=-1
    if kind=='codes_shape': codes=codes[0]
    if kind=='codes_dtype': codes=codes.astype(np.float32)
    with pytest.raises(ValueError): fn(torch.from_numpy(values),torch.from_numpy(mask),action_codes=torch.from_numpy(codes),capacity=capacity)
    with pytest.raises(ValueError): select_compact(values,mask,action_codes=codes,capacity=capacity)


@pytest.mark.parametrize('n',[255,256,257,513])
@pytest.mark.parametrize('capacity',[0,1,128,300])
def test_stable_order_across_block_boundaries(donor,n,capacity):
    d=np.arange(2*n*3,dtype=np.int32).reshape(2,n,3)
    mask=np.zeros((2,n),bool)
    mask[:,::3]=True
    if n>255: mask[:,254:257]=True
    codes=(np.arange(2*n,dtype=np.int64).reshape(2,n)-(1<<40))
    evaluate(donor,d,mask,codes,capacity)


def test_payload_bits_are_preserved(donor):
    words=np.array([0x80000000,0x7fc12345,0x7f800000,0xff800000],np.uint32)
    values=words.view(np.float32).reshape(1,4)
    mask=np.array([[1,1,1,1]],bool)
    torch,fn=donor
    original=fn(torch.from_numpy(values.copy()),torch.from_numpy(mask),capacity=5)['values'].numpy()
    result=select_compact(values,mask,capacity=5)['values']
    np.testing.assert_array_equal(result.view(np.uint32),original.view(np.uint32))
    np.testing.assert_array_equal(result[0,:4].view(np.uint32),words)


@pytest.mark.parametrize('capacity',[True,False,np.int64(2),1.5])
def test_capacity_scalar_behavior_matches_donor(donor,capacity):
    torch,fn=donor
    d=np.ones((1,3),np.float32); m=np.ones((1,3),bool)
    if isinstance(capacity,float):
        with pytest.raises(TypeError): fn(torch.from_numpy(d),torch.from_numpy(m),capacity=capacity)
        with pytest.raises(TypeError): select_compact(d,m,capacity=capacity)
    else: evaluate(donor,d,m,None,capacity)


@pytest.mark.parametrize('zero_stride',[False,True])
@pytest.mark.parametrize('code_dtype',[None,np.int32,np.int64])
def test_actual_strided_numpy_and_torch_views_preserve_bits_and_inputs(donor,zero_stride,code_dtype):
    torch,fn=donor
    words=np.array([0x7fc12345,0x80000000,0x7f800000,0xff800000,0x3f800000,0x40000000],np.uint32)
    if zero_stride:
        # Writable zero-stride views exercise broadcast semantics on both axes.
        row=words.view(np.float32).reshape(1,3,2)
        values=np.lib.stride_tricks.as_strided(row,shape=(2,3,2),strides=(0,*row.strides[1:]),writeable=True)
        mask_base=np.array([[True,True,True]],bool)
        mask=np.lib.stride_tricks.as_strided(mask_base,shape=(2,3),strides=(0,mask_base.strides[1]),writeable=True)
        code_base=np.array([[11,-12,13]],np.int64 if code_dtype is None else code_dtype)
        codes=np.lib.stride_tricks.as_strided(code_base,shape=(2,3),strides=(0,code_base.strides[1]),writeable=True)
    else:
        storage=np.zeros((2,3,4),np.uint32)
        storage[:,:,::2]=np.tile(words.reshape(1,3,2),(2,1,1))
        values=storage.view(np.float32)[:,:,::2]
        mask_storage=np.array([[1,0,0,1,1,0],[1,0,1,0,1,0]],bool)
        mask=mask_storage[:,::2]
        code_storage=np.arange(12,dtype=np.int64 if code_dtype is None else code_dtype).reshape(2,6)
        codes=code_storage[:,::2]
    if code_dtype is None: codes=None
    arrays=[values,mask]+([] if codes is None else [codes])
    snapshots=[(x.tobytes(),x.shape,x.strides) for x in arrays]
    tv,tm=torch.from_numpy(values),torch.from_numpy(mask)
    tc=None if codes is None else torch.from_numpy(codes)
    assert tuple(tv.stride())==tuple(s//values.itemsize for s in values.strides)
    assert tuple(tm.stride())==tuple(s//mask.itemsize for s in mask.strides)
    if tc is not None: assert tuple(tc.stride())==tuple(s//codes.itemsize for s in codes.strides)
    expected=fn(tv,tm,action_codes=tc,capacity=4)
    actual=select_compact(values,mask,action_codes=codes,capacity=4)
    for key,donor_value in expected.items():
        reference=donor_value.numpy()
        assert actual[key].shape==reference.shape and actual[key].dtype==reference.dtype
        np.testing.assert_array_equal(actual[key].view(np.uint8),reference.view(np.uint8))
    for array,(data,shape,strides) in zip(arrays,snapshots):
        assert array.tobytes()==data and array.shape==shape and array.strides==strides
