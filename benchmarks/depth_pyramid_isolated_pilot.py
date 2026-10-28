# SPDX-License-Identifier: Apache-2.0
"""Review-required single-dispatch pilot; explicit --execute and GPU lock required.

Only an exact committed checkout is supported. GPU results are measurement-only;
caller-provided control evidence is not verified by this runner.
"""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from benchmarks.depth_pyramid_gpu_fixture import ROOT

LOCK=Path('/tmp/sintergpu-gpu.lock')
CLOSURE=(
 'benchmarks/depth_pyramid_isolated_pilot.py', 'benchmarks/depth_pyramid_isolated_comparison.py',
 'benchmarks/depth_pyramid_gpu_fixture.py', 'benchmarks/depth_pyramid_consolidation.py',
 'benchmarks/device_snapshot.py','tests/depth_pyramid_candidate_fixture.py',
 'tests/test_depth_pyramid_gpu.py','src/sintergpu/references/depth_pyramid_profiles.py',
 'src/sintergpu/kernels/common/depth_pyramid_core.slang',
 'src/sintergpu/kernels/vkl/depth_pyramid.slang','src/sintergpu/kernels/slang_rgbd_slam/tracker.slang',
 'src/sintergpu/kernels/generated/primitives.generated.slang',
 'benchmarks/shaders/depth_pyramid_vkl_candidate.slang','benchmarks/shaders/depth_pyramid_slam_candidate.slang',
 'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py',
 'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/shaders/tracker.slang',
 'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/tracker.py')


def workloads():
    return [{'profile':'vkl','batch':3,'height':121,'width':161},
            {'profile':'slam','batch':1,'height':120,'width':160}]


@contextlib.contextmanager
def lock_device(path=LOCK):
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def verify_expected(snapshot, uuid, version, info, expected_compiler, actual_compiler):
    if (snapshot['physical_device_uuid'].lower()!=uuid.lower() or
        snapshot['driver']['version']!=version or snapshot['driver']['info']!=info or
        expected_compiler != actual_compiler):
        raise ValueError('device, driver or compiler identity mismatch')


def write_once(path, payload):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(payload,f,indent=2,allow_nan=False); f.write('\n')


def committed_closure(commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('full lowercase commit SHA required')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head != commit: raise ValueError('checkout HEAD differs from requested commit')
    result={}
    for name in CLOSURE:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked closure file: '+name)
        actual=path.read_bytes()
        expected=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
        if actual != expected: raise ValueError('source differs from committed tree: '+name)
        result[name]=hashlib.sha256(actual).hexdigest()
    return result


def snapshot():
    from benchmarks.device_snapshot import parse_snapshot
    raw=subprocess.check_output(['vulkaninfo','--summary'],text=True,stderr=subprocess.PIPE)
    return parse_snapshot(raw,hostname=socket.gethostname(),captured_at_utc=datetime.now(timezone.utc).isoformat())


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if args.samples < 50 or args.warmups < 0: raise ValueError('at least 50 samples required')
    closure=committed_closure(args.commit)
    receipt_path=Path(args.control_receipt)
    if receipt_path.is_symlink() or not receipt_path.is_file(): raise ValueError('control receipt must be a regular file')
    receipt_hash=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    output=Path(args.output_directory)
    if not output.is_dir(): raise ValueError('existing output directory required')
    if any((output/('isolated-'+w['profile']+'.json')).exists() for w in workloads()):
        raise FileExistsError('pilot output already exists')
    sys.path.insert(0,str(ROOT/'tests'))
    from depth_pyramid_candidate_fixture import candidate_preflight, candidate_source, candidate_fixture
    from test_depth_pyramid_gpu import expected, compare
    from benchmarks.depth_pyramid_consolidation import make_inputs, VKL_DONOR_EPSILON
    from benchmarks.depth_pyramid_gpu_fixture import Fixture, specification
    from benchmarks.depth_pyramid_isolated_comparison import collect, evaluate
    import slangpy as spy
    import numpy as np
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('cannot identify SlangPy compiler')
    compiler_hash=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    candidate_preflight()
    compositions={p:hashlib.sha256(candidate_source(p).encode()).hexdigest() for p in ('vkl','slam')}
    with lock_device():
        before=snapshot()
        verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler_hash)
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id != int(before['vendor_id'],0) or adapters[0].device_id != int(before['device_id'],0):
            raise ValueError('selected adapter and Vulkan snapshot differ')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name != before['adapter_name']: raise ValueError('device name mismatch')
            frequency=int(device.info.timestamp_frequency)
            start=datetime.now(timezone.utc).isoformat()
            def preflight():
                if committed_closure(args.commit)!=closure or hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=receipt_hash:
                    raise ValueError('source or control receipt changed')
                if hashlib.sha256(libs[0].read_bytes()).hexdigest()!=compiler_hash: raise ValueError('compiler changed')
                candidate_preflight()
                if any(hashlib.sha256(candidate_source(p).encode()).hexdigest()!=compositions[p] for p in compositions):
                    raise ValueError('composed shader changed')
                return {'sources':closure,'compiler':libs[0].name,'compiler_sha256':compiler_hash,
                        'runtime':'slangpy '+importlib.metadata.version('slangpy'),'device':before['adapter_name'],
                        'driver':before['driver']['version']+' / '+before['driver']['info'],
                        'device_uuid':before['physical_device_uuid'],'timestamp_frequency':frequency,
                        'control_receipt':{'sha256':receipt_hash,'verification':'caller-supplied, not independently verified'}}
            for workload in workloads():
                inp=make_inputs(workload)
                if workload['profile']=='vkl':
                    spec=specification('vkl',inp['depth'],valid=inp['valid'],confidence=inp['confidence'],threshold=VKL_DONOR_EPSILON)
                else:
                    spec=specification('slam',inp['depth'],threshold=.1)
                oracle=expected(spec)
                fixtures={'old':Fixture(device,spec),'candidate':candidate_fixture(device,spec)}
                report=collect(device,spy.QueryType.timestamp,fixtures,lambda result:compare(result,oracle),preflight,samples=args.samples,warmups=args.warmups)
                after=snapshot()
                verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
                report.update(commit=args.commit,composed_shader_sha256=compositions, snapshot_before=before,snapshot_after=after,
                    lock=str(LOCK),started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),
                    assessment=evaluate(report),status='measurement_only',
                    limitations='Caller control receipt and interference are unverified. Three loops in one process are not three independently controlled runs. No qualification or adoption claim.')
                write_once(output/('isolated-'+workload['profile']+'.json'),report)
        finally:
            device.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    for name in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','control-receipt','output-directory'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--samples',type=int,default=50); parser.add_argument('--warmups',type=int,default=5)
    run(parser.parse_args())

if __name__=='__main__': main()
