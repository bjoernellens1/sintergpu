# SPDX-License-Identifier: Apache-2.0
"""Review-required point-plane diagnostic: never performance or qualification.

Only the runner and listed shader/fixture files are commit-checked; imported
Python helpers, native runtime closure, and compiled shader are not authenticated.

Production execution is blocked; the private test path is CPU-mock-only.
Requires a separately reviewed instrumented SlangPy build exposing same-device
Vulkan creation receipt. GPU execution only after independent review. Caller
owns device, buffers, and canonical lock. Fresh buffers and native handle
uniqueness are observations, not physical backing-disjointness proof.
"""
import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
from benchmarks.depth_pyramid_timestamp_probe import BorrowedLease, LOCK
from benchmarks.depth_pyramid_timestamp_runner import owned_lease
from benchmarks.depth_pyramid_resident_parity_runner import full_vulkan_storage
from benchmarks.depth_pyramid_isolated_pilot import write_once

ROOT = Path(__file__).resolve().parents[1]
PATHS = ('tests/point_plane_aosoa_gpu_validation.py',
         'src/sintergpu/kernels/optimization/point_plane_aosoa_256.slang',
         'src/sintergpu/kernels/optimization/point_plane.slang',
         'src/sintergpu/fixtures/point_plane_aosoa/point_plane_aosoa_1025_candidate.slang',
         'src/sintergpu/fixtures/point_plane_aosoa/point_plane_aosoa_1025.npz')
ADAPTER, CORE, CANDIDATE, FIXTURE = PATHS[1:]
FIXTURE_SHA = '3a9e10f09ee3a312c52c00fe683bae8af09cf7e43b026c51c5ffa974c6edae9f'
PARAMS = dict(problem_size=1025, points_num_alloc=512, targets_num_alloc=1032,
              normals_num_alloc=1032, out_0_num_alloc=1032, out_1_num_alloc=1032)
NAMES = ('points', 'targets', 'normals', 'points_indices', 'out_0', 'out_1')


def closure(commit):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('full commit required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip() != commit: raise ValueError('HEAD mismatch')
    data = {}
    for name in PATHS:
        path = ROOT / name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing/symlinked closure source: '+name)
        raw = path.read_bytes()
        if raw != subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from commit: '+name)
        data[name] = hashlib.sha256(raw).hexdigest()
    if data[FIXTURE] != FIXTURE_SHA: raise ValueError('fixture identity mismatch')
    return data


@contextmanager
def source_snapshot(commit, identities):
    """Compile committed bytes; private read-only copy is not tamper-proof."""
    payload = {}
    for name in (ADAPTER, CORE, CANDIDATE):
        data = subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
        if hashlib.sha256(data).hexdigest() != identities[name]: raise ValueError('snapshot source mismatch')
        payload[name] = data
    with tempfile.TemporaryDirectory(prefix='sinter-point-plane-') as temp:
        include = Path(temp)
        module = include/'optimization/point_plane.slang'
        module.parent.mkdir()
        module.write_bytes(payload[CORE])
        module.chmod(0o444)
        yield include, {'packaged':payload[ADAPTER].decode('utf-8'), 'candidate':payload[CANDIDATE].decode('utf-8')}
        if hashlib.sha256(module.read_bytes()).hexdigest() != identities[CORE]: raise ValueError('snapshot core changed')


def load_fixture():
    data = (ROOT / FIXTURE).read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA: raise ValueError('fixture changed')
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        if set(source.files) != set(NAMES): raise ValueError('fixture keys')
        arrays = {key: source[key].copy() for key in NAMES}
    expected = {'points': ((512,4),np.dtype('float32')), 'targets': ((1032,4),np.dtype('float32')),
                'normals': ((1032,4),np.dtype('float32')), 'points_indices': ((1025,),np.dtype('uint32')),
                'out_0': ((1039,),np.dtype('float32')), 'out_1': ((6199,),np.dtype('float32'))}
    if any(arrays[k].shape != shape or arrays[k].dtype != dtype for k,(shape,dtype) in expected.items()): raise ValueError('fixture shapes/dtypes')
    if np.any(arrays['points_indices'] >= 512) or not np.all(arrays['out_0'] == -901) or not np.all(arrays['out_1'] == -902): raise ValueError('fixture indices/poison')
    return arrays


def validate_outputs(arrays, output):
    if set(output) != set(NAMES): raise AssertionError('missing readbacks')
    for key in NAMES:
        if output[key].shape != arrays[key].shape or output[key].dtype != arrays[key].dtype: raise AssertionError('readback shape/dtype')
    for key in ('points','targets','normals','points_indices'):
        if output[key].tobytes() != arrays[key].tobytes(): raise AssertionError('input changed: '+key)
    p = arrays['points'][arrays['points_indices'], :3].astype(np.float64)
    q = arrays['targets'][:1025,:3].astype(np.float64)
    n = arrays['normals'][:1025,:3].astype(np.float64)
    residual = np.sum(n * (p-q),axis=1)
    jacobian = np.c_[np.cross(p,n),n]
    live = np.c_[output['out_1'][:4100].reshape(1025,4), output['out_1'][4128:6178].reshape(1025,2)]
    np.testing.assert_allclose(output['out_0'][:1025], residual, atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(live, jacobian, atol=2e-6, rtol=3e-6)
    np.testing.assert_array_equal(output['out_0'][1025:], arrays['out_0'][1025:])
    np.testing.assert_array_equal(output['out_1'][4100:4128], arrays['out_1'][4100:4128])
    np.testing.assert_array_equal(output['out_1'][6178:], arrays['out_1'][6178:])


def check_device(spy, device, expected_uuid, expected_driver):
    receipt = device.vulkan_creation_receipt
    if (receipt is None or not receipt.finalized or not receipt.device_uuid_known or receipt.instance_external
            or receipt.physical_device_external or receipt.device_external or bytes(receipt.device_uuid) != expected_uuid
            or receipt.driver_version != expected_driver or receipt.owner_generation <= 0):
        raise ValueError('same-backend Vulkan device receipt unavailable or mismatch')
    expected = (spy.NativeHandleType.VkInstance, spy.NativeHandleType.VkPhysicalDevice, spy.NativeHandleType.VkDevice)
    recorded = (receipt.instance, receipt.physical_device, receipt.device)
    handles = device.native_handles
    if len(handles) != 3 or any(a.type != kind or b.type != kind or not a.value or a.value != b.value for a,b,kind in zip(handles,recorded,expected)):
        raise ValueError('same-backend device handle mismatch')
    return {'uuid': expected_uuid.hex(), 'driver': expected_driver, 'owner_generation': receipt.owner_generation}


def collect(device, spy, arrays, sources, *, storage_limit):
    limits = device.info.limits
    if (int(limits.max_compute_dispatch_thread_groups.x) < 5 or int(limits.max_compute_thread_group_size.x) < 256
            or int(limits.max_compute_threads_per_group) < 256 or storage_limit < max(a.nbytes for a in arrays.values())):
        raise ValueError('insufficient actual device limits')
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    observations = []; all_buffers = []
    for label in ('candidate', 'packaged'):
        source = sources[label]
        module = device.load_module_from_source('pointplane_'+label+'_'+hashlib.sha256(source.encode()).hexdigest()[:16], source)
        kernel = device.create_compute_kernel(device.link_program([module],[module.entry_point('main')]))
        buffers = {name: device.create_buffer(element_count=a.size, struct_size=4, usage=usage, data=np.ascontiguousarray(a).reshape(-1)) for name,a in arrays.items()}
        all_buffers.extend(buffers.values())  # Keep owners alive through both variants.
        handles = [int(b.native_handle.value) for b in all_buffers]
        if any(h <= 0 for h in handles) or len(set(handles)) != len(handles): raise ValueError('duplicate/invalid live buffer handles')
        kernel.dispatch(thread_count=(1280,1,1), vars={**buffers, 'params':PARAMS})
        device.wait_for_idle()
        output = {name: b.to_numpy().view(arrays[name].dtype).reshape(arrays[name].shape).copy() for name,b in buffers.items()}
        validate_outputs(arrays, output)
        observations.append(output)
    for name in NAMES:
        if observations[0][name].tobytes() != observations[1][name].tobytes(): raise AssertionError('paired output differs: '+name)
    return {name:hashlib.sha256(observations[0][name].tobytes()).hexdigest() for name in NAMES}


def mapped_binary(path, expected_sha):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file(): raise ValueError('runtime binary path')
    st = path.stat(); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha: raise ValueError('runtime binary hash mismatch')
    matches = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path.resolve()) and fields[4].isdigit() and int(fields[4]) == st.st_ino:
            matches.append(fields[3])
    if not matches or len(set(matches)) != 1: raise ValueError('runtime binary mapping unavailable or ambiguous')
    current=path.stat()
    if (current.st_dev,current.st_ino)!=(st.st_dev,st.st_ino): raise ValueError('runtime path changed')
    return {'path':str(path), 'sha256':digest, 'inode':st.st_ino, 'mapped_device':matches[0], 'stat_device':st.st_dev}


def runtime_check(args, spy):
    extension = sys.modules.get('slangpy.slangpy_ext')
    if extension is None or Path(extension.__file__).resolve()!=Path(args.extension).resolve(): raise ValueError('unexpected slangpy extension')
    roles = ((args.extension,args.extension_sha),(args.libsgl,args.libsgl_sha),(args.compiler,args.compiler_sha))
    if len({(Path(p).stat().st_dev,Path(p).stat().st_ino) for p,_ in roles})!=3: raise ValueError('runtime binary aliases')
    return [mapped_binary(path,sha) for path,sha in roles]


def run(args, *, _cpu_test_only=False):
    # Production gate: the separate vulkaninfo limit is not tied to this
    # backend's provider, and backing/source/compiled identities are unproven.
    # The private flag exists solely for injected CPU-only fake tests.
    if _cpu_test_only is not True:
        raise RuntimeError('point-plane GPU execution blocked pending same-backend identity and allocation review')
    if not args.execute: raise ValueError('explicit --execute required')
    output = Path(args.output)
    if output.exists() or output.is_symlink(): raise FileExistsError('output exists')
    uuid = bytes.fromhex(args.uuid.replace('-',''))
    if len(uuid)!=16 or not any(uuid) or args.driver<0: raise ValueError('UUID/driver required')
    identities = closure(args.commit)
    arrays = load_fixture()
    error = None; payload = None
    try:
        with owned_lease() as lease:
            if not isinstance(lease,BorrowedLease): raise ValueError('canonical lease required')
            lease.verify()
            device = None
            errors = []
            try:
                if any(k == 'slangpy' or k.startswith('slangpy.') for k in sys.modules): raise ValueError('runtime pre-imported')
                icd = os.environ.get('VK_DRIVER_FILES')
                if not icd or ':' in icd or os.environ.get('VK_ICD_FILENAMES'): raise ValueError('one ICD config required')
                icd_path = Path(icd)
                if not icd_path.is_absolute() or icd_path.is_symlink() or not icd_path.is_file(): raise ValueError('ICD config path')
                icd_sha = hashlib.sha256(icd_path.read_bytes()).hexdigest()
                spy = importlib.import_module('slangpy')
                binaries = runtime_check(args,spy)
                storage = full_vulkan_storage(args.uuid)
                adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
                if len(adapters)!=1 or adapters[0].vendor_id!=args.vendor or adapters[0].device_id!=args.device: raise ValueError('unexpected/ambiguous adapter')
                with source_snapshot(args.commit, identities) as (include_dir, sources):
                    options = {'include_paths':[str(include_dir)], 'floating_point_mode':spy.SlangFloatingPointMode.precise}
                    device = spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options=options)
                    device_receipt=check_device(spy,device,uuid,args.driver)
                    hashes=collect(device,spy,arrays,sources,storage_limit=storage['max_storage_buffer_range'])
                    if check_device(spy,device,uuid,args.driver)!=device_receipt: raise ValueError('device receipt changed')
                payload={'status':'exploratory_correctness_only', 'commit':args.commit, 'shader_fixture_and_runner_source_sha256':identities,
                         'device_receipt':device_receipt, 'runtime_binary_observations':binaries,
                         'icd_config_sha256':icd_sha, 'storage_observation':storage, 'readback_sha256':hashes,
                         'scope':'fixed fixture, paired full outputs and CPU oracle only; private read-only source snapshot is not immutable; no verified compiled binary, provider, allocation backing, performance or device qualification'}
                if closure(args.commit)!=identities or runtime_check(args,spy)!=binaries: raise ValueError('source/runtime changed')
                if hashlib.sha256(icd_path.read_bytes()).hexdigest()!=icd_sha: raise ValueError('ICD config changed')
            except Exception as exc: errors.append(exc)
            finally:
                if device is not None:
                    try: device.close()
                    except Exception as exc: errors.append(exc)
            try: lease.verify()
            except Exception as exc: errors.append(exc)
            if len(errors)==1: raise errors[0]
            if errors: raise ExceptionGroup('device and teardown/lock errors',errors)
    except Exception as exc: error=exc
    if error:
        write_once(output, {'status':'failure','error_type':type(error).__name__,'error':str(error),'scope':'no qualification'})
        raise error
    write_once(output,payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    for k in ('commit','uuid','output','extension','extension-sha','libsgl','libsgl-sha','compiler','compiler-sha'):
        p.add_argument('--'+k,required=True)
    for k in ('driver','vendor','device'): p.add_argument('--'+k,type=lambda x:int(x,0),required=True)
    run(p.parse_args())

if __name__=='__main__': main()
