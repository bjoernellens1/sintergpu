# SPDX-License-Identifier: Apache-2.0
"""Opt-in timestamp diagnostics; reviewed before use, never qualification."""
import argparse
import contextlib
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from benchmarks.depth_pyramid_resident_parity_runner import (ROOT, INIT, STREAM, closure as base_closure,
    verify_loaded_closure, require_local_modules, snapshot, verify_expected, full_vulkan_storage, check_limits, write_once)
from benchmarks.depth_pyramid_resident_graph import ResidentGraph
from benchmarks.depth_pyramid_timestamp_probe import BorrowedLease, collect, analyze
from sintergpu.vkl import depth_pyramid_plan

LOCK=Path('/tmp/sintergpu-gpu.lock')
EXTRA=('benchmarks/depth_pyramid_timestamp_runner.py','benchmarks/depth_pyramid_timestamp_probe.py',
       'benchmarks/depth_pyramid_resident_graph.py','src/sintergpu/vkl.py')


@contextlib.contextmanager
def owned_lease():
    fd=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=BorrowedLease(fd); lease.verify()
        yield lease
    finally: os.close(fd)


def closure(commit):
    identities=base_closure(commit)
    for name in EXTRA:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from committed tree: '+name)
        identities[name]=hashlib.sha256(data).hexdigest()
    return identities


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if not 50<=args.samples<=1024: raise ValueError('50 to 1024 samples required')
    identities=closure(args.commit); output=Path(args.output); receipt=Path(args.control_receipt)
    if output.exists(): raise FileExistsError('output exists')
    if receipt.is_symlink() or not receipt.is_file(): raise ValueError('regular control receipt required')
    receipt_hash=hashlib.sha256(receipt.read_bytes()).hexdigest()
    sys.path.insert(0,str(ROOT/'tests'))
    from test_depth_pyramid_gpu import compare
    from benchmarks.depth_pyramid_consolidation import cpu_reference,VKL_DONOR_EPSILON
    import benchmarks.device_snapshot
    import sintergpu.references.depth_pyramid_profiles
    verify_loaded_closure(identities)
    module=sys.modules['benchmarks.depth_pyramid_timestamp_probe']
    require_local_modules({'timestamp_probe':module})
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=identities['benchmarks/depth_pyramid_timestamp_probe.py']:
        raise ValueError('timestamp probe differs from committed source')
    import slangpy as spy
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('compiler unavailable')
    compiler=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    with owned_lease() as lease:
        before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler)
        storage=full_vulkan_storage(before['physical_device_uuid'])
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id!=int(before['vendor_id'],0) or adapters[0].device_id!=int(before['device_id'],0): raise ValueError('adapter mismatch')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name!=before['adapter_name']: raise ValueError('device name mismatch')
            def load(path,entry):
                mod=device.load_module(str(ROOT/path))
                return device.create_compute_kernel(device.link_program([mod],[mod.entry_point(entry)]))
            kernels={'initializeDepth':load(INIT,'initializeDepth'),'depthPyramidLevel':load(STREAM,'depthPyramidLevel')}
            shape=(3,121,161); rng=np.random.default_rng(613)
            d=rng.uniform(.1,4,shape).astype(np.float32); d.flat[0]=np.nan
            v=(rng.random(shape)>.2).astype(np.uint32)*7; c=rng.random(shape,dtype=np.float32)
            inputs={'depth':d,'valid':v,'confidence':c}
            plan=depth_pyramid_plan(*shape,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups.x),confidence=True)
            limits=check_limits(device.info.limits,plan,max_fixture_bytes=32<<20,max_storage_buffer_range=storage['max_storage_buffer_range'])
            graph=ResidentGraph(device,kernels,inputs,usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access,
                max_workgroups_x=limits['max_groups_x'],source_ids={k:identities[k] for k in (INIT,STREAM)},confidence=True)
            refs=cpu_reference(inputs,profile='vkl',threshold=VKL_DONOR_EPSILON)['levels']
            expected={p['level']:dict(zip(('depthOut','validOut','confidenceOut'),x)) for p,x in zip(plan,refs)}
            def identity():
                if closure(args.commit)!=identities or hashlib.sha256(receipt.read_bytes()).hexdigest()!=receipt_hash: raise ValueError('source or receipt changed')
                verify_loaded_closure(identities)
                if hashlib.sha256(libs[0].read_bytes()).hexdigest()!=compiler: raise ValueError('compiler changed')
                return {'device_uuid':before['physical_device_uuid'],'driver':before['driver']['version']+' / '+before['driver']['info'],
                        'compiler_sha256':compiler,'source_sha256':identities,'lock_receipt_sha256':receipt_hash}
            validations=0
            def validate():
                nonlocal validations
                if validations==0:
                    encoder=device.create_command_encoder(); graph.record(encoder)
                    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
                graph.read_checked(lambda level,values:compare(values,expected[level])); validations+=1
            report=collect(device,spy.QueryType.timestamp,graph,identity,validate,samples=args.samples,lease=lease)
            after_storage=full_vulkan_storage(before['physical_device_uuid'])
            if after_storage['icd_environment']!=storage['icd_environment'] or after_storage['max_storage_buffer_range']!=storage['max_storage_buffer_range']:
                raise ValueError('ICD or storage device limit changed')
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
            numpy_binary=Path(np._core._multiarray_umath.__file__).resolve(); python_binary=Path(sys.executable).resolve()
            payload={'status':'diagnostic_only','commit':args.commit,'source_sha256':identities,'compiler_sha256':compiler,
                'runtime':{'python':{'version':sys.version,'sha256':hashlib.sha256(python_binary.read_bytes()).hexdigest()},
                           'numpy':{'version':np.__version__,'sha256':hashlib.sha256(numpy_binary.read_bytes()).hexdigest()},
                           'slangpy':{'version':importlib.metadata.version('slangpy'),'compiler_sha256':compiler}},
                'snapshot_before':before,'snapshot_after':after,'storage_before':storage,'storage_after':after_storage,
                'input_sha256':{k:hashlib.sha256(x.tobytes()).hexdigest() for k,x in inputs.items()},'shape':shape,'limits':limits,
                'completed_utc':datetime.now(timezone.utc).isoformat(),'raw_ticks':report,'analysis':analyze(report),
                'scope':'Before/after batch oracle and guard validation only, no per-sample parity. Caller control receipt and interference unverified. Observed tick grid does not establish timer resolution, cause, or performance qualification.'}
            write_once(output,payload)
        finally: device.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true')
    for k in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','control-receipt','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--samples',type=int,default=50); run(p.parse_args())

if __name__=='__main__': main()
