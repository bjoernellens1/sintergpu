# SPDX-License-Identifier: Apache-2.0
"""Compare independent CPU backward oracle to frozen VKL Torch autograd."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sintergpu.references.compact_backward import compact_value_backward, accumulate_view_gradient

ROOT=Path(__file__).resolve().parents[1]
DONOR=ROOT/'references/donors/vkl/src/vkl/kernels/reference/compact.py'

@pytest.fixture(scope='module')
def donor():
    torch=pytest.importorskip('torch')
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest()=='f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff'
    spec=importlib.util.spec_from_file_location('frozen_vkl_compact_backward',DONOR)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return torch,mod.select_compact

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('capacity',[0,1,2,4,7])
@pytest.mark.parametrize('strides,offset,base_size',[
    ((12,3,1),0,24), ((24,6,2),1,48), ((0,3,1),0,12), ((2,1,1),0,8)])
def test_retained_values_backward_matches_torch_autograd(donor,dtype,capacity,strides,offset,base_size):
    torch,select=donor; shape=(2,4,3)
    base=torch.arange(base_size,dtype=torch.float32 if dtype==np.float32 else torch.float64,requires_grad=True)
    values=torch.as_strided(base,shape,strides,offset)
    mask=np.array([[True,False,True,True],[False,True,True,True]])
    upstream=(np.arange(2*capacity*3,dtype=dtype).reshape(2,capacity,3)%7)-3
    result=select(values,torch.from_numpy(mask),capacity=capacity)
    result['values'].backward(torch.from_numpy(upstream))
    logical=compact_value_backward(shape,mask,upstream)
    expected=accumulate_view_gradient(logical,base_size=base_size,strides=strides,offset=offset)
    assert base.grad.numpy().dtype==dtype
    np.testing.assert_array_equal(base.grad.numpy(),expected,strict=True)

@pytest.mark.parametrize('shape,capacity', [((0,4,3),3),((2,0,3),3),((2,4,0),3),((2,4),2)])
def test_empty_and_scalar_payloads(donor,shape,capacity):
    torch,select=donor
    values=torch.zeros(shape,dtype=torch.float64,requires_grad=True)
    mask=np.ones(shape[:2],bool)
    upstream=np.ones((shape[0],capacity,*shape[2:]),np.float64)
    result=select(values,torch.from_numpy(mask),capacity=capacity)
    result['values'].backward(torch.from_numpy(upstream))
    np.testing.assert_array_equal(values.grad.numpy(),compact_value_backward(shape,mask,upstream),strict=True)


def test_backward_input_validation():
    with pytest.raises(ValueError): compact_value_backward((1,2),np.ones((1,2),np.uint8),np.ones((1,2)))
    with pytest.raises(ValueError): compact_value_backward((1,2),np.ones((1,2),bool),np.ones((1,2),np.int32))
    with pytest.raises(ValueError): accumulate_view_gradient(np.ones((2,),np.float32),base_size=1,strides=(1,))


def test_empty_selection_and_strided_upstream(donor):
    torch,select=donor
    values=torch.ones((2,4,3),dtype=torch.float32,requires_grad=True)
    mask=np.zeros((2,4),bool)
    backing=np.arange(2*3*6,dtype=np.float32).reshape(2,3,6)
    upstream=backing[:,:,::2]
    assert not upstream.flags.c_contiguous
    select(values,torch.from_numpy(mask),capacity=3)['values'].backward(torch.from_numpy(upstream))
    np.testing.assert_array_equal(values.grad.numpy(),compact_value_backward(values.shape,mask,upstream),strict=True)

@pytest.mark.parametrize('dtype,torch_dtype,rtol,atol',[
    (np.float16,'float16',2e-3,2e-3),
    (np.complex64,'complex64',1e-5,1e-6),
    (np.complex128,'complex128',1e-12,1e-12)])
def test_overlap_dtypes_and_unequal_magnitudes(donor,dtype,torch_dtype,rtol,atol):
    torch,select=donor
    shape=(2,4,3); strides=(2,1,1); base_size=8
    base=torch.zeros(base_size,dtype=getattr(torch,torch_dtype),requires_grad=True)
    values=torch.as_strided(base,shape,strides)
    mask=np.array([[True,True,False,True],[True,False,True,True]])
    raw=np.array([0.125,32.,-0.5,2.,-16.,0.25,1.,-4.,0.03125,8.,-1.,0.5,0.25,16.,-2.,1.,-8.,0.0625],dtype=np.float64).reshape(2,3,3)
    if np.dtype(dtype).kind=='c': raw=raw+1j*(raw[::-1]*0.5)
    upstream=raw.astype(dtype)
    select(values,torch.from_numpy(mask),capacity=3)['values'].backward(torch.from_numpy(upstream))
    expected=accumulate_view_gradient(compact_value_backward(shape,mask,upstream),base_size=base_size,strides=strides)
    assert base.grad.numpy().dtype==np.dtype(dtype)
    np.testing.assert_allclose(base.grad.numpy(),expected,rtol=rtol,atol=atol,equal_nan=True)

@pytest.mark.parametrize('dtype,torch_dtype',[(np.float16,'float16'),(np.float32,'float32')])
def test_overlap_nonfinite_gradients(donor,dtype,torch_dtype):
    torch,select=donor
    shape=(1,3,2); strides=(0,1,1); base_size=4
    base=torch.zeros(base_size,dtype=getattr(torch,torch_dtype),requires_grad=True)
    values=torch.as_strided(base,shape,strides)
    mask=np.ones((1,3),bool)
    upstream=np.array([[[np.inf,1],[-np.inf,np.nan],[2,-0.0]]],dtype=dtype)
    select(values,torch.from_numpy(mask),capacity=3)['values'].backward(torch.from_numpy(upstream))
    with np.errstate(invalid='ignore'):
        expected=accumulate_view_gradient(compact_value_backward(shape,mask,upstream),base_size=base_size,strides=strides)
    np.testing.assert_allclose(base.grad.numpy(),expected,rtol=0,atol=0,equal_nan=True)


def test_nonfinite_complex_accumulation_explicitly_unsupported(donor):
    torch,select=donor
    base=torch.zeros(4,dtype=torch.complex64,requires_grad=True)
    values=torch.as_strided(base,(1,3,2),(0,1,1))
    mask=np.ones((1,3),bool)
    upstream=np.array([[[np.inf,1],[-np.inf,np.nan],[2,-0.0]]],dtype=np.complex64)
    select(values,torch.from_numpy(mask),capacity=3)['values'].backward(torch.from_numpy(upstream))
    assert np.isnan(base.grad.numpy().imag[:3]).all()
    logical=compact_value_backward((1,3,2),mask,upstream)
    with pytest.raises(ValueError,match='nonfinite complex'):
        accumulate_view_gradient(logical,base_size=4,strides=(0,1,1))
