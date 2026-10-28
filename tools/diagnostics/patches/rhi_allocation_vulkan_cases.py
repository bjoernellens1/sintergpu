# SPDX-License-Identifier: Apache-2.0
"""Review-required Vulkan diagnostic cases; no standalone GPU runner.

The caller MUST hold the canonical lease before device creation through close,
verify instrumented binaries and ICD, and prohibit foreign Vulkan operations.
No result is allocation attestation or performance evidence.
"""
from benchmarks.depth_pyramid_timestamp_probe import BorrowedLease


def identity(spy, device, uuid, driver):
    receipt = device.vulkan_creation_receipt
    if (receipt is None or not receipt.finalized or not receipt.device_uuid_known
            or receipt.instance_external or receipt.physical_device_external or receipt.device_external
            or bytes(receipt.device_uuid) != uuid or receipt.driver_version != driver
            or receipt.owner_generation <= 0):
        raise ValueError('same-backend Vulkan creation identity unavailable or mismatched')
    handles = device.native_handles
    observed = (receipt.instance, receipt.physical_device, receipt.device)
    types = (spy.NativeHandleType.VkInstance, spy.NativeHandleType.VkPhysicalDevice, spy.NativeHandleType.VkDevice)
    if len(handles) != 3 or any(not a.value or a.type != kind or b.type != kind or a.value != b.value for a, b, kind in zip(handles, observed, types)):
        raise ValueError('creation receipt handles mismatch')
    return tuple((h.type, h.value) for h in handles), bytes(receipt.device_uuid), receipt.owner_generation


def observation(buffer, size):
    value = buffer.vulkan_allocation_observation_v1
    if not isinstance(value, dict): raise AssertionError('ordinary allocation unavailable')
    required = ('schema_version', 'owner_generation', 'buffer', 'memory', 'binding_offset', 'buffer_size', 'allocation_size')
    if set(value) != set(required) or any(type(value[k]) is not int for k in required):
        raise AssertionError('observation schema')
    if (value['schema_version'] != 1 or value['owner_generation'] <= 0 or value['buffer'] <= 0
            or value['memory'] <= 0 or value['binding_offset'] != 0 or value['buffer_size'] != size
            or value['allocation_size'] < size or value['buffer'] != buffer.native_handle.value):
        raise AssertionError('observation contents or buffer identity')
    return value.copy()


def run_cases(spy, device, *, lease, expected_uuid, expected_driver):
    """Record observations under a caller-held lease, then close device.

    Shared support must be available or this fails; no silent skip. Wrapped and
    imported buffers cannot be constructed through this SGL API and remain open.
    Copied observations become historical on close; keep owners alive until then.
    """
    retained = []
    try:
        if not isinstance(lease, BorrowedLease): raise ValueError('borrowed canonical lease required')
        if not isinstance(expected_uuid, bytes) or len(expected_uuid) != 16 or not any(expected_uuid): raise ValueError('expected UUID required')
        if type(expected_driver) is not int or expected_driver < 0: raise ValueError('driver version required')
        lease.verify()
        before = identity(spy, device, expected_uuid, expected_driver)
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        ordinary = [device.create_buffer(size=n, usage=usage) for n in (64, 257)]
        retained.extend(ordinary)
        values = [observation(b, n) for b, n in zip(ordinary, (64, 257))]
        if len({v['owner_generation'] for v in values}) != 2 or len({v['buffer'] for v in values}) != 2 or len({v['memory'] for v in values}) != 2:
            raise AssertionError('duplicate live observations')
        if [observation(b, n) for b, n in zip(ordinary, (64, 257))] != values:
            raise AssertionError('live observation changed')
        shared = device.create_buffer(size=64, usage=usage | spy.BufferUsage.shared)
        retained.append(shared)
        if shared.vulkan_allocation_observation_v1 is not None: raise AssertionError('shared buffer admitted')
        if observation(ordinary[1], 257) != values[1]: raise AssertionError('unrelated live observation changed')
        if identity(spy, device, expected_uuid, expected_driver) != before: raise AssertionError('device identity changed')
        lease.verify()
    finally:
        device.close()
    lease.verify()
    for buffer in retained:
        if buffer.vulkan_allocation_observation_v1 is not None: raise AssertionError('observation survived SGL close')
    return {'status': 'diagnostic_only', 'uuid': expected_uuid.hex(), 'driver': expected_driver,
            'sizes': [64, 257], 'shared': 'rejected', 'post_close': 'rejected',
            'scope': 'no physical backing proof; wrapped/imported cases pending'}
