# SPDX-License-Identifier: Apache-2.0
"""CPU fakes only; never create a GPU device."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

path = Path(__file__).with_name('rhi_allocation_vulkan_cases.py')
spec = importlib.util.spec_from_file_location('receipt_cases', path)
cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)


class Lease:
    def __init__(self): self.checks = 0
    def verify(self): self.checks += 1


class Buffer:
    def __init__(self, dev, size, num, shared):
        self.dev, self.shared, self.tainted = dev, shared, False
        self.native_handle = NS(value=num)
        self.value = dict(schema_version=1, owner_generation=num, buffer=num, memory=num+100,
                          binding_offset=0, buffer_size=size, allocation_size=size+8)
    @property
    def vulkan_allocation_observation_v1(self):
        return None if self.dev.closed or self.shared or self.tainted else self.value.copy()
    @property
    def shared_handle(self): self.tainted = True; return None


class Device:
    def __init__(self):
        self.closed = False
        self.buffers = []
        self.native_handles = [NS(type=i, value=i+10) for i in range(3)]
        self.vulkan_creation_receipt = NS(finalized=True, device_uuid_known=True, instance_external=False,
            physical_device_external=False, device_external=False, device_uuid=bytes(range(16)), driver_version=7,
            instance=self.native_handles[0], physical_device=self.native_handles[1], device=self.native_handles[2], owner_generation=1)
    def create_buffer(self, *, size, usage):
        b = Buffer(self, size, len(self.buffers)+1, bool(usage & 4)); self.buffers.append(b); return b
    def close(self): self.closed = True


spy = NS(BufferUsage=NS(shader_resource=1, unordered_access=2, shared=4),
         NativeHandleType=NS(VkInstance=0, VkPhysicalDevice=1, VkDevice=2))


def test_full_mock_contract(monkeypatch):
    monkeypatch.setattr(cases, 'BorrowedLease', Lease)
    lease, device = Lease(), Device()
    result = cases.run_cases(spy, device, lease=lease, expected_uuid=bytes(range(16)), expected_driver=7)
    assert result['status'] == 'diagnostic_only' and device.closed and lease.checks == 3
    assert not device.buffers[0].tainted and device.buffers[2].shared


def test_fail_closed_identity_and_lease(monkeypatch):
    monkeypatch.setattr(cases, 'BorrowedLease', Lease)
    d = Device()
    with pytest.raises(ValueError, match='lease'): cases.run_cases(spy, d, lease=None, expected_uuid=bytes(range(16)), expected_driver=7)
    assert d.closed and not d.buffers
    d = Device()
    with pytest.raises(ValueError, match='identity'): cases.run_cases(spy, d, lease=Lease(), expected_uuid=bytes(range(16)), expected_driver=8)
    assert d.closed and not d.buffers


def test_duplicate_or_changed_observation_fails_and_closes(monkeypatch):
    monkeypatch.setattr(cases, 'BorrowedLease', Lease)
    d = Device()
    original = d.create_buffer
    def create(*, size, usage):
        b = original(size=size, usage=usage)
        if len(d.buffers) == 2: b.value['memory'] = d.buffers[0].value['memory']
        return b
    d.create_buffer = create
    with pytest.raises(AssertionError, match='duplicate'): cases.run_cases(spy, d, lease=Lease(), expected_uuid=bytes(range(16)), expected_driver=7)
    assert d.closed
