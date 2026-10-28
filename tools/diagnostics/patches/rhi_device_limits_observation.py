# SPDX-License-Identifier: Apache-2.0
"""Validate version-2 diagnostic observations, not a provenance proof.

The caller retains the device and serializes close/release. Raw dispatch
addresses may be loader trampolines and must not be treated as ICD identity.
"""


def observation(spy, device, expected_uuid, expected_driver):
    if not isinstance(expected_uuid, bytes) or len(expected_uuid) != 16 or not any(expected_uuid):
        raise ValueError('nonzero UUID required')
    if type(expected_driver) is not int or expected_driver < 0:
        raise ValueError('raw driver version required')
    r = device.vulkan_creation_receipt
    if (r is None or r.schema_version != 2 or not r.finalized or not r.device_uuid_known
            or r.owner_generation <= 0 or r.instance_external or r.physical_device_external or r.device_external
            or bytes(r.device_uuid) != expected_uuid or r.driver_version != expected_driver):
        raise ValueError('receipt identity unavailable or mismatch')
    handles = device.native_handles
    types = (spy.NativeHandleType.VkInstance, spy.NativeHandleType.VkPhysicalDevice, spy.NativeHandleType.VkDevice)
    recorded = (r.instance, r.physical_device, r.device)
    if len(handles) != 3 or any(a.type != t or b.type != t or not a.value or a.value != b.value for a,b,t in zip(handles,recorded,types)):
        raise ValueError('live device handles mismatch')
    fields = ('max_storage_buffer_range','max_uniform_buffer_range','min_storage_buffer_offset_alignment',
              'min_uniform_buffer_offset_alignment','max_compute_work_group_invocations')
    if not r.device_limits_known or any(type(getattr(r,k)) is not int or getattr(r,k) <= 0 for k in fields):
        raise ValueError('device limits unavailable')
    count, size = tuple(r.max_compute_work_group_count), tuple(r.max_compute_work_group_size)
    if len(count)!=3 or len(size)!=3 or any(type(x) is not int or x<=0 for x in count+size):
        raise ValueError('compute limits unavailable')
    if (not r.dispatch_addresses_known or type(r.get_instance_proc_addr_address) is not int
            or type(r.get_physical_device_properties_address) is not int
            or r.get_instance_proc_addr_address <= 0 or r.get_physical_device_properties_address <= 0):
        raise ValueError('dispatch addresses unavailable')
    return {**{key:getattr(r,key) for key in fields}, 'max_compute_work_group_count':count,
            'max_compute_work_group_size':size, 'dispatch_addresses': (r.get_instance_proc_addr_address,
            r.get_physical_device_properties_address), 'scope':'backend observations; provider and physical backing unverified'}


def inspect_dispatch_address(address):
    """Read dladdr + executable /proc mapping for an observed address; never call it.

    This is an observation, not authentication: loader trampolines, dynamic
    interposition, mutable files and mapped-page differences remain possible.
    Linux/glibc and a readable /proc/self/maps are required; otherwise fail.
    """
    import ctypes
    import hashlib
    from pathlib import Path
    if type(address) is not int or address <= 0 or address >= 1 << (ctypes.sizeof(ctypes.c_void_p)*8):
        raise ValueError('invalid address')
    class DlInfo(ctypes.Structure):
        _fields_=[('fname',ctypes.c_char_p),('fbase',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
    try:
        dladdr=ctypes.CDLL(None).dladdr
    except AttributeError as exc:
        raise ValueError('dladdr unavailable') from exc
    dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dladdr.restype=ctypes.c_int
    info=DlInfo()
    if dladdr(ctypes.c_void_p(address),ctypes.byref(info)) != 1 or not info.fname or not info.fbase:
        raise ValueError('address has no dladdr mapping')
    path=Path(info.fname.decode('utf-8',errors='strict'))
    if not path.is_absolute(): raise ValueError('nonabsolute provider observation')
    path=path.resolve(strict=True)
    if not path.is_file(): raise ValueError('provider path unavailable')
    st=path.stat(); matches=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or fields[5].endswith(' (deleted)') or 'x' not in fields[1]: continue
        try:
            lo,hi=(int(v,16) for v in fields[0].split('-'))
            inode=int(fields[4])
        except ValueError: continue
        if lo<=address<hi:
            matches.append((fields[5],fields[3],inode))
    if len(matches)!=1: raise ValueError('ambiguous executable mapping')
    mapped, map_device, inode=matches[0]
    if Path(mapped).resolve()!=path or inode!=st.st_ino: raise ValueError('dladdr and mapping differ')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    now=path.stat()
    if (st.st_dev,st.st_ino)!=(now.st_dev,now.st_ino): raise ValueError('provider file changed')
    return {'address':address,'dladdr_base':int(info.fbase),'path':str(path),'inode':st.st_ino,
            'mapping_device':map_device,'stat_device':st.st_dev,'sha256':digest,
            'symbol':info.sname.decode('utf-8',errors='replace') if info.sname else None,
            'scope':'address mapping only; provider identity and mapped bytes not authenticated'}
