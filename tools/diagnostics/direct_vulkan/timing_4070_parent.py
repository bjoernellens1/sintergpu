# SPDX-License-Identifier: Apache-2.0
"""RTX 4070 trusted-host timing candidate. Independent review required before execution."""
import argparse
import fcntl
import hashlib
import json
import stat
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
BASE='tools/diagnostics/direct_vulkan/'
LOCAL=('timing_4070_parent.py','timing_4070_trusted.cpp','timing_parent.py','timing_report.py','timing_executable.cpp','point_plane.cpp','timing_resources.h',
       'timing_resource_plan.h','timing_pipelines.h','timing_readback.h','timing_oracle.h','timing_tick_bound.h',
       'timing_drain.h','timing_journal.h','vulkan_point_plane_timing.h','vulkan_point_plane_record.h',
       'vulkan_transfer_pass.h','vulkan_timestamp_pair.h','engineering_attempt/runner.py',
       'engineering_attempt/spirv_contract.py')
# Target build is provisional until exact committed source is independently reviewed.
NATIVE_SHA='f8fbce518a37ee0a22e4b74700820b6aa526bfbab273e9984620dffce3ffdca5'
CPP=Path('/usr/bin/c++')
CPP_SHA='1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769'
FLAGS=['-std=c++17','-O0','-g0','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers']
UUID='96d2dba095fd5c00e8492e0610f24e06'
DRIVER=2497233024
LOCK=Path('/tmp/sintergpu-gpu.lock')

def sha(data): return hashlib.sha256(data).hexdigest()

def committed_sources(commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('full commit required')
    result={}
    for rel in LOCAL:
        path=ROOT/BASE/rel
        if path.is_symlink() or path.resolve()!=path: raise ValueError('source path or symlink')
        source=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+BASE+rel])
        if path.read_bytes()!=source: raise ValueError('source mismatch: '+rel)
        result[rel]=source
    return result

def load_checked(source,name,path):
    scope={'__name__':name,'__file__':str(path)}
    exec(compile(source,str(path),'exec'),scope)
    return scope

def build_native(sources, compiler=CPP):
    if sha(compiler.read_bytes())!=CPP_SHA: raise ValueError('C++ compiler mismatch')
    with tempfile.TemporaryDirectory(prefix='sinter-timing-build-') as tmp:
        directory=Path(tmp)
        # All local includes are copied from verified Git bytes to this directory.
        for rel,data in sources.items():
            if '/' not in rel:
                (directory/rel).write_bytes(data)
        builds=[]
        for number in range(2):
            binary=directory/('binary'+str(number))
            subprocess.run([str(compiler),*FLAGS,str(directory/'timing_4070_trusted.cpp'),'-o',str(binary),'-lvulkan','-lcrypto'],check=True,capture_output=True,env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
            builds.append(binary.read_bytes())
        if builds[0]!=builds[1] or sha(compiler.read_bytes())!=CPP_SHA: raise ValueError('native build changed')
        if NATIVE_SHA is None or sha(builds[0])!=NATIVE_SHA: raise ValueError('reviewed native build pin unavailable or mismatched')
        return builds[0]

def controlled_runs(*,lease_fd, executable, shader_paths, fixture_path, journal, helpers, parse_report, arrays, controls, process=subprocess.run, samples=50):
    """One lock, three child processes; caller provides independently verified closure.

    The test injects a fake process. Peer/clock/power controls are external and
    recorded separately; no performance qualification follows from this helper.
    """
    if type(samples)!=int or samples!=50: raise ValueError('exactly 50 samples required')
    runs=[]
    for index in range(3):
        helpers['lease_event'](lease_fd,journal,f'run-{index}-before')
        controls(journal,f'run-{index}-before')
        output=journal.root/f'run-{index}.samples'
        cmd=[str(executable),UUID,str(DRIVER),str(shader_paths[0]),str(shader_paths[1]),str(fixture_path),str(output),str(lease_fd),str(samples)]
        try:
            child=process(cmd,check=False,capture_output=True,pass_fds=(lease_fd,))
        except Exception as exc:
            journal.event('process_exception',run=index,error=str(exc));raise
        journal.event('process_exit',run=index,returncode=child.returncode)
        journal.put(f'run-{index}.stdout',child.stdout);journal.put(f'run-{index}.stderr',child.stderr)
        helpers['lease_event'](lease_fd,journal,f'run-{index}-after')
        controls(journal,f'run-{index}-after')
        if child.returncode: raise RuntimeError(f'run {index} failed ({child.returncode})')
        def read_regular(path,limit):
            import stat
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                st=os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size>limit: raise ValueError('artifact size/type')
                with os.fdopen(fd,'rb',closefd=False) as f: data=f.read()
                if len(data)!=st.st_size: raise ValueError('artifact changed')
                os.fsync(fd);os.fsync(journal.directory)
                return data
            finally: os.close(fd)
        sample_bytes=read_regular(output,4*1024*1024)
        raw=read_regular(Path(str(output)+'.raw'),samples*3*2*(74268+32))
        def validate(data):
            decoded=helpers['decode'](data)
            helpers['validate'](arrays,decoded,decoded)
        report=parse_report(sample_bytes,raw,validate,samples=samples)
        journal.event('run_validated',run=index,sample_sha256=sha(sample_bytes),raw_sha256=sha(raw),samples=samples)
        runs.append(report)
    return runs

def check_peer(journal,stage):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,check=False,text=True)
    journal.event('compute_processes',stage=stage,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode: raise RuntimeError('nvidia-smi process query failed')
    target='GPU-96d2dba0-95fd-5c00-e849-2e0610f24e06'
    for line in result.stdout.splitlines():
        if line.strip().startswith(target+','): raise RuntimeError('target GPU compute process present')


def receipt_once(path,payload):
    path=Path(path);directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    temporary='.sinter-timing-receipt-'+os.urandom(16).hex();published=False
    try:
        fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600,dir_fd=directory)
        with os.fdopen(fd,'w') as f:json.dump(payload,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temporary,path.name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False);published=True
        os.unlink(temporary,dir_fd=directory);os.fsync(directory)
    except Exception as original:
        errors=[original]
        if published:
            try:os.unlink(path.name,dir_fd=directory);os.fsync(directory)
            except Exception as rollback:errors.append(rollback)
        try:os.unlink(temporary,dir_fd=directory)
        except FileNotFoundError:pass
        except Exception as cleanup:errors.append(cleanup)
        if len(errors)>1:raise ExceptionGroup('receipt publication failed',errors)
        raise
    finally:os.close(directory)


def run(args):
    raise RuntimeError('4070 timing candidate blocked pending independent review')
    if NATIVE_SHA is None: raise RuntimeError('reviewed native build pin unavailable')
    if not args.execute: raise ValueError('explicit execute required')
    if Path(args.output).exists() or Path(args.output).is_symlink(): raise FileExistsError(args.output)
    try:
        sources=committed_sources(args.commit)
        helper=load_checked(sources['engineering_attempt/runner.py'],'sinter_timing_helper',ROOT/BASE/'engineering_attempt/runner.py')
        report=load_checked(sources['timing_report.py'],'sinter_timing_report',ROOT/BASE/'timing_report.py')
        spirv=load_checked(sources['engineering_attempt/spirv_contract.py'],'sinter_timing_spirv',ROOT/BASE/'engineering_attempt/spirv_contract.py')
        journal=helper['ArtifactJournal'](args.output)
    except Exception as early_error:
        try:receipt_once(args.output,{'status':'failed','stage':'source_or_journal_setup','error_type':type(early_error).__name__,'error':str(early_error)})
        except Exception as receipt_error:raise ExceptionGroup('setup and failure receipt',[early_error,receipt_error])
        raise
    lock_fd=None;error=None;payload=None
    try:
        icd_hash=helper['checked_icd'](args.icd,args.icd_sha)
        shader_sources=helper['checked_sources'](args.commit);arrays,packed=helper['fixture'](shader_sources['fixture'])
        compiler=Path(args.slangc).resolve(strict=True)
        if sha(compiler.read_bytes())!=helper['COMPILER_SHA']:raise ValueError('Slang compiler mismatch')
        binary=build_native(sources);journal.put('native',binary);os.chmod(journal.root/'native',0o700)
        journal.put('input.bin',packed)
        with tempfile.TemporaryDirectory(prefix='sinter-timing-shaders-') as tmp:
            directory=Path(tmp);(directory/'optimization').mkdir();(directory/'optimization/point_plane.slang').write_bytes(shader_sources['core']);shaders=[];abi=[]
            for role,entry,group in (('donor','computeMain',1024),('candidate','main',256)):
                source=directory/(role+'.slang');source.write_bytes(shader_sources[role]);output=directory/(role+'.spv')
                result=helper['captured_process']([str(compiler),str(source),'-I',str(directory),'-entry',entry,'-target','spirv','-o',str(output)],journal,role+'-compiler')
                if result.returncode: raise RuntimeError('shader compile failed: '+role)
                data=output.read_bytes()
                if sha(data)!=helper['SPV_SHA'][role]:raise ValueError('shader identity: '+role)
                abi.append(spirv['validate'](data,group));journal.put(role+'.spv',data);shaders.append(journal.root/(role+'.spv'))
            lock_fd=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            runs=controlled_runs(lease_fd=lock_fd,executable=journal.root/'native',shader_paths=shaders,fixture_path=journal.root/'input.bin',journal=journal,helpers=helper,parse_report=report['parse'],arrays=arrays,controls=check_peer,samples=args.samples)
            helper['lease_event'](lock_fd,journal,'final');check_peer(journal,'final')
            if committed_sources(args.commit)!=sources or helper['checked_icd'](args.icd,args.icd_sha)!=icd_hash or sha(compiler.read_bytes())!=helper['COMPILER_SHA'] or (journal.root/'native').read_bytes()!=binary:raise ValueError('closure changed')
            payload={'status':'unqualified_trusted_host_timing_only','commit':args.commit,'source_sha256':{k:sha(v) for k,v in sources.items()},'binary_sha256':sha(binary),'cxx_sha256':CPP_SHA,'slang_sha256':helper['COMPILER_SHA'],'icd_configuration_sha256':icd_hash,'shader_sha256':helper['SPV_SHA'],'shader_abi':abi,'target_uuid':UUID,'driver':DRIVER,'runs':len(runs),'samples_per_run':args.samples,'scope':'three child processes; no independent environmental control, statistical acceptance, hardware or performance qualification; loader and runtime trusted'}
    except Exception as exc:
        error=exc
    finally:
        if lock_fd is not None:
            try:os.close(lock_fd);journal.event('lock_closed')
            except Exception as exc:error=ExceptionGroup('operation and lock closure', [error,exc]) if error else exc
        try:journal.close()
        except Exception as exc:error=ExceptionGroup('operation and journal closure',[error,exc]) if error else exc
    if error:
        try:receipt_once(args.output,{'status':'failed','error_type':type(error).__name__,'error':str(error),'artifact_directory':str(journal.root)})
        except Exception as receipt_error:raise ExceptionGroup('operation and failure receipt', [error,receipt_error])
        raise error
    receipt_once(args.output,payload)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--slangc',required=True);p.add_argument('--icd',required=True);p.add_argument('--icd-sha',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=50);args=p.parse_args()
    run(args)
if __name__=='__main__': main()
