# SPDX-License-Identifier: Apache-2.0
"""Native top/bottom adjacent-query diagnostic runner. Review before GPU use; no speed claim."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from benchmarks.depth_pyramid_timestamp_runner import owned_lease
from benchmarks.depth_pyramid_resident_parity_runner import ROOT, closure as base_closure, snapshot, verify_expected, full_vulkan_storage, write_once, verify_loaded_closure, require_local_modules

SOURCE='benchmarks/native/vulkan_timestamp_matched.c'
EXTRA=(SOURCE,'benchmarks/depth_pyramid_native_matched_runner.py','benchmarks/depth_pyramid_timestamp_runner.py','benchmarks/depth_pyramid_timestamp_probe.py')
FLAGS=['-std=c11','-O2','-Wall','-Wextra','-Werror']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closure(commit):
    identities=base_closure(commit)
    for name in EXTRA:
        p=ROOT/name
        if p.is_symlink() or not p.is_file(): raise ValueError('source unavailable: '+name)
        data=p.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from committed tree: '+name)
        identities[name]=hashlib.sha256(data).hexdigest()
    return identities


def verify_modules(identities,modules=None):
    """Bind all executed local helpers to their exact committed bytes."""
    verify_loaded_closure(identities)
    if modules is None:
        import benchmarks.device_snapshot
        names=('benchmarks.depth_pyramid_native_matched_runner',
               'benchmarks.depth_pyramid_timestamp_runner',
               'benchmarks.depth_pyramid_timestamp_probe',
               'benchmarks.depth_pyramid_resident_parity_runner',
               'benchmarks.depth_pyramid_stream_pilot',
               'benchmarks.device_snapshot')
        modules={name:(sys.modules[__name__] if name=='benchmarks.depth_pyramid_native_matched_runner' else sys.modules[name]) for name in names}
    require_local_modules(modules,ROOT)
    for name,module in modules.items():
        path=Path(module.__file__).resolve(); relative=str(path.relative_to(ROOT.resolve()))
        if relative not in identities or digest(path)!=identities[relative]:
            raise ValueError('loaded module absent from committed closure: '+name)


def validate_result(data,uuid,vendor,device,driver,family,samples):
    if not isinstance(data,dict) or (data.get('device_uuid'),data.get('vendor_id'),data.get('device_id'),data.get('driver_version'),data.get('queue_family'))!=(uuid,vendor,device,driver,family): raise ValueError('native device or queue identity mismatch')
    period=data.get('timestamp_period_ns'); bits=data.get('timestamp_valid_bits')
    if not isinstance(period,(int,float)) or isinstance(period,bool) or not 0<period<float('inf') or type(bits) is not int or not 1<=bits<=64: raise ValueError('native timestamp capability invalid')
    for key in ('top_pairs','bottom_pairs'):
        pairs=data.get(key)
        if not isinstance(pairs,list) or len(pairs)!=samples or any(not isinstance(p,list) or len(p)!=2 or any(type(x) is not int or not 0<=x<2**64 for x in p) for p in pairs): raise ValueError('invalid raw query pairs')


def dependency_hashes(compiler):
    result=subprocess.run([str(compiler),*FLAGS,'-M',str(ROOT/SOURCE)],text=True,capture_output=True,check=True)
    line=result.stdout.replace('\\\n',' ')
    if ':' not in line: raise ValueError('compiler dependency output invalid')
    paths=shlex.split(line.split(':',1)[1])
    if not paths: raise ValueError('compiler dependencies unavailable')
    hashes={}
    for name in paths:
        p=Path(name).resolve(strict=True)
        if not p.is_file(): raise ValueError('dependency unavailable')
        hashes[str(p)]=digest(p)
    if str((ROOT/SOURCE).resolve()) not in hashes: raise ValueError('probe source missing from dependencies')
    return hashes


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if not 50<=args.samples<=1024: raise ValueError('50 to 1024 samples required')
    identities=closure(args.commit); verify_modules(identities)
    output=Path(args.output); receipt=Path(args.control_receipt)
    if output.exists(): raise FileExistsError('output exists')
    if receipt.is_symlink() or not receipt.is_file(): raise ValueError('regular control receipt required')
    receipt_hash=digest(receipt)
    cc_path=shutil.which(args.cc)
    if not cc_path: raise ValueError('compiler unavailable')
    compiler=Path(cc_path).resolve(strict=True); cc_hash=digest(compiler)
    if cc_hash!=args.cc_sha256: raise ValueError('C compiler hash mismatch')
    icd_vars={k:os.environ[k] for k in ('VK_DRIVER_FILES','VK_ICD_FILENAMES') if os.environ.get(k)}
    if len(icd_vars)!=1: raise ValueError('exactly one explicit ICD variable required')
    icd=Path(next(iter(icd_vars.values())))
    if not icd.is_absolute() or not icd.is_file(): raise ValueError('single ICD file required')
    icd_hash=digest(icd)
    build_dir=Path(args.build_directory)
    if not build_dir.is_dir(): raise ValueError('existing build directory required')
    with tempfile.TemporaryDirectory(prefix='native-timestamp-',dir=build_dir) as scratch:
        binary=Path(scratch)/'probe'
        dependencies=dependency_hashes(compiler)
        version=subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]
        command=[str(compiler),*FLAGS,str(ROOT/SOURCE),'-lvulkan','-o',str(binary)]
        build=subprocess.run(command,text=True,capture_output=True,check=True)
        binary_hash=digest(binary)
        linkage=subprocess.check_output(['ldd',str(binary)],text=True)
        if 'not found' in linkage or 'libvulkan.so' not in linkage: raise ValueError('Vulkan loader linkage unavailable')
        with owned_lease() as lease:
            lease.verify(); verify_modules(identities)
            before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,'same','same')
            storage=full_vulkan_storage(before['physical_device_uuid'])
            if storage['icd_environment']!=icd_vars: raise ValueError('ICD environment mismatch')
            execution=subprocess.run([str(binary),args.gpu_uuid,str(args.driver_version_raw),str(args.queue_family),str(args.samples)],text=True,capture_output=True,check=True)
            result=json.loads(execution.stdout)
            validate_result(result,args.gpu_uuid,int(before['vendor_id'],0),int(before['device_id'],0),args.driver_version_raw,args.queue_family,args.samples)
            lease.verify()
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,'same','same')
            storage_after=full_vulkan_storage(before['physical_device_uuid'])
            if storage_after['icd_environment']!=icd_vars or storage_after['max_storage_buffer_range']!=storage['max_storage_buffer_range']: raise ValueError('ICD or limits changed')
            verify_modules(identities)
            if closure(args.commit)!=identities or dependency_hashes(compiler)!=dependencies or digest(compiler)!=cc_hash or digest(binary)!=binary_hash or digest(icd)!=icd_hash or digest(receipt)!=receipt_hash:
                raise ValueError('source, build or control identity changed')
            report={'status':'diagnostic_only','commit':args.commit,'source_sha256':identities,'build':{'compiler':str(compiler),'compiler_sha256':cc_hash,'version':version,'command':command,'stdout':build.stdout,'stderr':build.stderr,'dependency_sha256':dependencies,'binary_sha256':binary_hash,'ldd':linkage},
                    'icd':{'environment':icd_vars,'json_sha256':icd_hash},'control_receipt_sha256_unverified':receipt_hash,'snapshot_before':before,'snapshot_after':after,'storage_before':storage,'storage_after':storage_after,
                    'completed_utc':datetime.now(timezone.utc).isoformat(),'native':result,'stderr':execution.stderr,
                    'query_stage':'Alternating adjacent TOP_OF_PIPE and BOTTOM_OF_PIPE pairs in one native command buffer; no shader dispatched',
                    'tick_interpretation':'raw ticks only; no tick differences or wrap assumptions',
                    'scope':'Native TOP_OF_PIPE and BOTTOM_OF_PIPE adjacent empty query pairs on one explicitly selected queue; no identical SGL queue or command composition established; no shader. No shader execution, per-sample parity, timer-resolution inference, root cause, speed or performance qualification. ICD library binary identity and caller control receipt require independent verification.'}
            write_once(output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true')
    for k in ('commit','gpu-uuid','driver-version','driver-info','cc-sha256','control-receipt','build-directory','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--queue-family',required=True,type=int); p.add_argument('--driver-version-raw',required=True,type=int); p.add_argument('--cc',default='cc'); p.add_argument('--samples',default=50,type=int)
    run(p.parse_args())

if __name__=='__main__': main()
