# SPDX-License-Identifier: Apache-2.0
"""Review-required opt-in compact correctness probe; never performance qualification.

Requires exact committed sources and an exclusive same-process GPU lease. Fresh
VkBuffer handles are checked; physical backing allocation identity is unavailable.
"""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import numpy as np
from benchmarks.depth_pyramid_timestamp_runner import closure as base_closure, owned_lease
from benchmarks.depth_pyramid_resident_parity_runner import (ROOT, verify_loaded_closure,
    require_local_modules, full_vulkan_storage, snapshot, verify_expected, write_once)
from compact_gpu_support import GUARD, make_case
from compact_gpu_runtime import GPUFixture

EXTRA=('tests/compact_gpu_runner.py','tests/compact_gpu_support.py','tests/compact_gpu_runtime.py',
       'tests/test_vkl_compact_gpu_differential.py','benchmarks/compact_graph_fixture.py',
       'benchmarks/shaders/vkl_compact_candidate.slang','src/sintergpu/kernels/primitives/scan_uint.slang',
       'src/sintergpu/compact_plan.py','src/sintergpu/primitives.py',
       'src/sintergpu/references/compact_numeric.py',
       'references/donors/vkl/src/vkl/kernels/reference/compact.py')
CASES=((0,3,1,0),(1,0,2,0),(255,5,1,4),(256,2,3,8),(257,4,8,0),(65537,3,2,8))


def closure(commit):
    identities=base_closure(commit)
    for name in EXTRA:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]):
            raise ValueError('source differs from committed tree: '+name)
        identities[name]=hashlib.sha256(data).hexdigest()
    return identities


def verify_modules(identities):
    verify_loaded_closure(identities)
    names=('compact_gpu_runner','compact_gpu_support','compact_gpu_runtime',
           'benchmarks.compact_graph_fixture','benchmarks.depth_pyramid_timestamp_runner',
           'benchmarks.depth_pyramid_timestamp_probe','sintergpu.compact_plan',
           'sintergpu.primitives','sintergpu.references.compact_numeric','sintergpu')
    modules={name:sys.modules[name] for name in names if name in sys.modules}
    modules['compact_runner_entry']=sys.modules[__name__]
    require_local_modules(modules)
    for module in modules.values():
        path=Path(module.__file__).resolve(); relative=str(path.relative_to(ROOT.resolve()))
        if identities.get(relative)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('loaded module differs from committed closure: '+relative)


def case_for_limits(args,limits,storage):
    x=int(limits.max_compute_dispatch_thread_groups.x)
    tx=int(limits.max_compute_thread_group_size.x); threads=int(limits.max_compute_threads_per_group)
    if x<1 or tx<256 or threads<256: raise ValueError('device dispatch/workgroup limit')
    budget=min(32<<20,storage)-GUARD*4
    if budget<4: raise ValueError('device storage range too small')
    case=make_case(*args,max_workgroups_x=x,max_buffer_bytes=budget,max_threads_per_group=threads,max_group_size_x=tx)
    if any(d['allocated_bytes']+GUARD*4>storage for d in case['plan']['resources'].values()):
        raise ValueError('bound range exceeds maxStorageBufferRange')
    return case


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if Path(args.output).exists(): raise FileExistsError(args.output)
    identities=closure(args.commit); verify_modules(identities)
    import torch
    import slangpy as spy
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('compiler unavailable')
    compiler=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    with owned_lease() as lease:
        lease.verify(); verify_modules(identities)
        before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler)
        storage=full_vulkan_storage(before['physical_device_uuid'])
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id!=int(before['vendor_id'],0) or adapters[0].device_id!=int(before['device_id'],0): raise ValueError('adapter mismatch')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name!=before['adapter_name']: raise ValueError('adapter name mismatch')
            rows=[]
            for params in CASES:
                lease.verify(); case=case_for_limits(params,device.info.limits,storage['max_storage_buffer_range'])
                fixture=GPUFixture(device,spy,case)
                fixture.run_stagewise(); fixture.reset(); fixture.run_graph(); fixture.reset(); fixture.run_graph()
                rows.append({'case':params,'buffer_handles':fixture.handle_identity,
                    'input_sha256':{k:hashlib.sha256(case[k].tobytes()).hexdigest() for k in ('values_bytes','mask_bytes','action_bytes')},
                    'torch_donor_checked':params[2] in (1,2,4,8)})
                if closure(args.commit)!=identities: raise ValueError('source changed')
                verify_modules(identities)
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
            storage_after=full_vulkan_storage(before['physical_device_uuid'])
            if storage_after['icd_environment']!=storage['icd_environment'] or storage_after['max_storage_buffer_range']!=storage['max_storage_buffer_range']: raise ValueError('ICD or storage limit changed')
            lease.verify()
            write_once(args.output,{'status':'correctness_probe_only','commit':args.commit,'source_sha256':identities,
                'snapshot_before':before,'snapshot_after':after,'storage_before':storage,'storage_after':storage_after,
                'compiler_sha256':compiler,'slangpy_version':importlib.metadata.version('slangpy'),'numpy_version':np.__version__,
                'torch_version':torch.__version__,'python':sys.version,'cases':rows,
                'scope':'Fresh retained VkBuffer handles; physical backing allocation disjointness is unverified. Explicit barriers and readback after each stage. No gradient or performance qualification.'})
        finally: device.close()
        lease.verify()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute',action='store_true')
    for field in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','output'):
        parser.add_argument('--'+field,required=True)
    run(parser.parse_args())

if __name__=='__main__': main()
