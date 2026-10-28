# SPDX-License-Identifier: Apache-2.0
"""Review-required opt-in forward/backward correctness probe, no GPU qualification."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import numpy as np
from compact_gpu_runner import (ROOT, closure as base_closure, verify_modules as verify_base,
    owned_lease, snapshot, verify_expected, full_vulkan_storage, write_once, case_for_limits)
from compact_gpu_runtime import GPUFixture
from compact_backward_gpu_fixture import BackwardFixture, independent_gradient, torch_gradient
from compact_gpu_support import make_case
from benchmarks.depth_pyramid_resident_parity_runner import require_local_modules

EXTRA=('tests/compact_backward_gpu_runner.py','tests/compact_backward_gpu_fixture.py',
       'tests/test_compact_backward_gpu_fixture.py','src/sintergpu/compact_backward_plan.py',
       'src/sintergpu/references/compact_backward.py','benchmarks/shaders/vkl_compact_backward_candidate.slang')
CASES=((0,3,2,'float16'),(1,0,4,'float32'),(255,3,4,'float32'),(256,5,8,'float64'),
       (257,4,8,'complex64'),(17,3,16,'complex128'),(7,5,2,'float16'))
DTYPES={'float16':np.float16,'float32':np.float32,'float64':np.float64,'complex64':np.complex64,'complex128':np.complex128}


def closure(commit):
    ids=base_closure(commit)
    for name in EXTRA:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from committed tree: '+name)
        ids[name]=hashlib.sha256(data).hexdigest()
    return ids


def verify_modules(ids):
    verify_base(ids)
    names=('compact_backward_gpu_fixture','sintergpu.compact_backward_plan','sintergpu.references.compact_backward')
    mods={name:sys.modules[name] for name in names if name in sys.modules}; mods['runner']=sys.modules[__name__]
    require_local_modules(mods)
    for mod in mods.values():
        path=Path(mod.__file__).resolve(); name=str(path.relative_to(ROOT.resolve()))
        if ids.get(name)!=hashlib.sha256(path.read_bytes()).hexdigest(): raise ValueError('loaded module differs from closure: '+name)


def make_upstream(case,dtype):
    b,_,tail=case['shape']; cap=case['capacity']; typ=DTYPES[dtype]
    result=np.arange(b*cap*tail,dtype=np.float32).astype(typ).reshape(b,cap,tail)
    if cap and dtype=='float32':
        words=result.view(np.uint32).reshape(-1); words[:min(len(words),4)]=np.array([0x80000000,0x7f800000,0xff800000,0x7fc12345],np.uint32)[:min(len(words),4)]
    return result


def preflight_oracles():
    """Fail before device use when bit-preserving and donor contracts differ."""
    for n,cap,width,dtype in CASES:
        case=make_case(n,cap,width,0); upstream=make_upstream(case,dtype)
        np.testing.assert_array_equal(independent_gradient(case,upstream),torch_gradient(case,upstream,dtype),err_msg=f'bitwise donor mismatch: {dtype}, n={n}, capacity={cap}')


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if Path(args.output).exists(): raise FileExistsError(args.output)
    ids=closure(args.commit); verify_modules(ids)
    import torch
    preflight_oracles()
    import slangpy as spy
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('compiler unavailable')
    compiler=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    with owned_lease() as lease:
        lease.verify(); verify_modules(ids)
        before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler)
        storage=full_vulkan_storage(before['physical_device_uuid']); max_storage=storage['max_storage_buffer_range']
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id!=int(before['vendor_id'],0) or adapters[0].device_id!=int(before['device_id'],0): raise ValueError('adapter mismatch')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name!=before['adapter_name']: raise ValueError('adapter name mismatch')
            limits=device.info.limits; rows=[]
            for n,cap,width,dtype in CASES:
                lease.verify(); case=case_for_limits((n,cap,width,0),limits,max_storage)
                forward=GPUFixture(device,spy,case); upstream=make_upstream(case,dtype)
                backward=BackwardFixture(device,spy,forward,upstream,max_groups=int(limits.max_compute_dispatch_thread_groups.x),storage_limit=max_storage,max_threads=int(limits.max_compute_threads_per_group),max_group_size=int(limits.max_compute_thread_group_size.x))
                hashes=[]
                for iteration in range(2):
                    if iteration: forward.reset(); backward.reset()
                    state=forward.run_graph(); hashes.append(backward.run(state,dtype))
                rows.append({'case':[n,cap,width,dtype],'forward_buffer_handles':forward.handle_identity,'backward_buffer_handles':backward.handles,
                    'upstream_sha256':hashlib.sha256(upstream.tobytes()).hexdigest(),'output_sha256':hashes})
                if closure(args.commit)!=ids: raise ValueError('source changed')
                verify_modules(ids)
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
            storage_after=full_vulkan_storage(before['physical_device_uuid'])
            if storage_after['icd_environment']!=storage['icd_environment'] or storage_after['max_storage_buffer_range']!=max_storage: raise ValueError('ICD or storage changed')
            lease.verify(); write_once(args.output,{'status':'correctness_probe_only','commit':args.commit,'source_sha256':ids,
                'snapshot_before':before,'snapshot_after':after,'storage_before':storage,'storage_after':storage_after,
                'compiler_sha256':compiler,'slangpy_version':importlib.metadata.version('slangpy'),'torch_version':torch.__version__,
                'numpy_version':np.__version__,'python':sys.version,'cases':rows,
                'scope':'Actual forward slot buffer, fixture readbacks between passes; no production host roundtrip implied. VkBuffer uniqueness does not prove physical memory disjointness. No view accumulation, framework interop, GPU qualification or performance claim.'})
        finally: device.close()
        lease.verify()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true')
    for field in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','output'): p.add_argument('--'+field,required=True)
    run(p.parse_args())

if __name__=='__main__': main()
