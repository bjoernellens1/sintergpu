# SPDX-License-Identifier: Apache-2.0
"""CPU-only fake observations; no Vulkan or GPU initialization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
p=Path(__file__).with_name('rhi_device_limits_observation.py')
spec=importlib.util.spec_from_file_location('rhi_device_limits_observation',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def setup():
    types=NS(VkInstance=1,VkPhysicalDevice=2,VkDevice=3)
    h=[NS(type=i,value=i+10) for i in (1,2,3)]
    r=NS(schema_version=2,finalized=True,device_uuid_known=True,owner_generation=2,
         instance_external=False,physical_device_external=False,device_external=False,
         device_uuid=bytes(range(16)),driver_version=7,instance=h[0],physical_device=h[1],device=h[2],
         device_limits_known=True,max_storage_buffer_range=65536,max_uniform_buffer_range=65536,
         min_storage_buffer_offset_alignment=16,min_uniform_buffer_offset_alignment=16,
         max_compute_work_group_invocations=256,max_compute_work_group_count=(10,2,2),
         max_compute_work_group_size=(256,2,2),dispatch_addresses_known=True,
         get_instance_proc_addr_address=42,get_physical_device_properties_address=43)
    return NS(NativeHandleType=types), NS(vulkan_creation_receipt=r,native_handles=h),r


def test_valid_receipt_is_only_observation():
    spy,dev,r=setup(); v=m.observation(spy,dev,bytes(range(16)),7)
    assert v['max_storage_buffer_range']==65536 and v['dispatch_addresses']==(42,43)
    assert 'provider and physical backing unverified' in v['scope']


@pytest.mark.parametrize('field,value', [('schema_version',1),('owner_generation',0),('device_limits_known',False),
    ('instance_external',True),('device_uuid_known',False),('max_storage_buffer_range',0),
    ('max_uniform_buffer_range',0),('min_storage_buffer_offset_alignment',0),
    ('max_compute_work_group_count',(2,0,1)),('max_compute_work_group_size',(1,1)),
    ('dispatch_addresses_known',False),('get_instance_proc_addr_address',0)])
def test_missing_or_invalid_observation_is_rejected(field,value):
    spy,dev,r=setup();setattr(r,field,value)
    with pytest.raises(ValueError): m.observation(spy,dev,bytes(range(16)),7)


def test_changed_device_handle_rejected():
    spy,dev,r=setup();dev.native_handles=list(dev.native_handles);dev.native_handles[1]=NS(type=2,value=99)
    with pytest.raises(ValueError,match='handles'): m.observation(spy,dev,bytes(range(16)),7)


def test_closed_device_receipt_unavailable():
    spy,dev,r=setup();dev.vulkan_creation_receipt=None
    with pytest.raises(ValueError,match='unavailable'): m.observation(spy,dev,bytes(range(16)),7)


def test_invalid_expected_identity_rejected():
    spy,dev,r=setup()
    with pytest.raises(ValueError,match='UUID'): m.observation(spy,dev,bytes(16),7)
    with pytest.raises(ValueError,match='driver'): m.observation(spy,dev,bytes(range(16)),True)


def test_live_cpu_libc_address_observation_and_bad_address():
    import ctypes
    address=ctypes.cast(ctypes.CDLL(None).printf,ctypes.c_void_p).value
    result=m.inspect_dispatch_address(address)
    assert result['address']==address and len(result['sha256'])==64
    assert 'not authenticated' in result['scope']
    with pytest.raises(ValueError): m.inspect_dispatch_address(0)
    with pytest.raises(ValueError): m.inspect_dispatch_address(1)
