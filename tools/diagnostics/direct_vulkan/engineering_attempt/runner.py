# SPDX-License-Identifier: Apache-2.0
"""Trusted-workstation engineering parity only; independent review before GPU.

This does not qualify SlangPy, provider/ICD identity, physical backing or speed.
The canonical GPU lease is held across both child processes and output validation.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import stat
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCK = Path('/tmp/sintergpu-gpu.lock')
NAMES = ('points','targets','normals','out_0','out_1','points_indices')
PATHS = {'donor':'src/sintergpu/kernels/caspar_slang/composed/point_plane_residual.slang',
         'candidate':'src/sintergpu/kernels/optimization/point_plane_aosoa_256.slang',
         'core':'src/sintergpu/kernels/optimization/point_plane.slang',
         'fixture':'src/sintergpu/fixtures/point_plane_aosoa/point_plane_aosoa_1025.npz'}
SOURCE_SHA = {'donor':'b425f7562662c7d4fedfa9c0ca5c8c83ac45b764bf75ca1bf8d2cb50147eeeb6',
              'candidate':'86cc3d94eff9193c4000d95aceed9cedf6fce5dcbd8971f8e8e6eab5b8751d91',
              'core':'9763495aa01821c4a9fa52db689310cf787cee01b63b8b50bf0daaae43131bcc',
              'fixture':'3a9e10f09ee3a312c52c00fe683bae8af09cf7e43b026c51c5ffa974c6edae9f'}
SPV_SHA = {'donor':'b34c971681eba7c1c7d9a885a13223fc3d481ab72fa39f2b9bf0ddbda2f00f76',
           'candidate':'8e56c1349d3a3fdf0ce9c88ee008f977818abbfae368adb4216d74720691afc6'}
COMPILER_SHA='19c0217f848537db2a87120eb1b3f882ab2e3eb54fdec4cf5300035943704145'
INPUT_SHA='e8d3cf128e6f3c547433f2bc457efc26ce0e24147592dc76bcbe4aee726178f7'
# This pin identifies the proposed engineering build only; it is not runtime attestation.
NATIVE_SOURCE_SHA='ab1da2441f8554828e2bc792d84aad71ce0ba0ce838428fb16e4653cf34471d2'
NATIVE_BINARY_SHA='b04a1e4d73dd2b4db1820a6d6816ecb9fc03c32f8a47f96b0bcbaa0dbbe431ad'
SHAPES={'points':((512,4),'<f4'), 'targets':((1032,4),'<f4'), 'normals':((1032,4),'<f4'),
        'out_0':((1039,),'<f4'), 'out_1':((6199,),'<f4'), 'points_indices':((1025,),'<u4')}

def sha(data): return hashlib.sha256(data).hexdigest()
def checked_sources(commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('full commit required')
    result={}
    for role,path in PATHS.items():
        data=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])
        if sha(data)!=SOURCE_SHA[role]: raise ValueError('committed source mismatch: '+role)
        result[role]=data
    return result

def checked_tool_sources(commit):
    directory=Path(__file__).resolve().parent
    result={}
    for name in ('runner.py','verified_build.py','spirv_contract.py'):
        local=directory/name
        if local.is_symlink(): raise ValueError('tool symlink')
        data=local.read_bytes();relative=local.relative_to(ROOT).as_posix()
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])
        if data!=committed: raise ValueError('tool differs from commit: '+name)
        result[name]=sha(data)
    return result

def load_checked_helpers(commit, hashes):
    # Compile exactly the Git-verified bytes; never consume timestamp-based pyc.
    directory=Path(__file__).resolve().parent
    functions={}
    for name,symbol in (('verified_build.py','build'),('spirv_contract.py','validate')):
        local=directory/name;relative=local.relative_to(ROOT).as_posix()
        data=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])
        if sha(data)!=hashes.get(name) or sha(local.read_bytes())!=hashes.get(name): raise ValueError('helper source changed: '+name)
        namespace={'__file__':str(local),'__name__':'sinter_verified_'+name.replace('.','_')}
        exec(compile(data,str(local),'exec'),namespace)
        functions[symbol]=namespace[symbol]
    return functions['build'],functions['validate']

def checked_icd(path, expected):
    p=Path(path)
    if not p.is_absolute() or p.is_symlink() or not p.is_file(): raise ValueError('ICD configuration path')
    if os.environ.get('VK_DRIVER_FILES')!=str(p) or any(os.environ.get(k) for k in ('VK_ICD_FILENAMES','VK_ADD_DRIVER_FILES','VK_LAYER_PATH','VK_ADD_LAYER_PATH','VK_INSTANCE_LAYERS','LD_PRELOAD')): raise ValueError('ICD environment')
    data=p.read_bytes()
    if sha(data)!=expected: raise ValueError('ICD configuration hash')
    return sha(data)

def fixture(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        if set(z.files)!=set(NAMES): raise ValueError('fixture names')
        arrays={name:z[name].copy() for name in NAMES}
    for name,(shape,dtype) in SHAPES.items():
        a=arrays[name]
        if a.shape!=shape or a.dtype!=np.dtype(dtype): raise ValueError('fixture shape/dtype')
    if np.any(arrays['points_indices']>=512): raise ValueError('fixture index')
    packed=b''.join(arrays[k].tobytes() for k in NAMES)
    if sha(packed)!=INPUT_SHA: raise ValueError('fixture content')
    return arrays,packed

def decode(data):
    output={};offset=0
    for name,(shape,dtype) in SHAPES.items():
        size=int(np.prod(shape))*4; part=data[offset:offset+size]
        if len(part)!=size: raise AssertionError('short output')
        output[name]=np.frombuffer(part,dtype=dtype).reshape(shape).copy();offset+=size
    if offset!=len(data): raise AssertionError('extra output')
    return output

def validate(arrays, a, b):
    for name in NAMES:
        if a[name].tobytes()!=b[name].tobytes(): raise AssertionError('paired mismatch: '+name)
    for name in ('points','targets','normals','points_indices'):
        if a[name].tobytes()!=arrays[name].tobytes(): raise AssertionError('input changed: '+name)
    p=arrays['points'][arrays['points_indices'],:3].astype(np.float64)
    q=arrays['targets'][:1025,:3].astype(np.float64);n=arrays['normals'][:1025,:3].astype(np.float64)
    residual=np.sum(n*(p-q),axis=1);jac=np.c_[np.cross(p,n),n]
    actual=np.c_[a['out_1'][:4100].reshape(1025,4),a['out_1'][4128:6178].reshape(1025,2)]
    np.testing.assert_allclose(a['out_0'][:1025],residual,atol=2e-6,rtol=3e-6)
    np.testing.assert_allclose(actual,jac,atol=2e-6,rtol=3e-6)
    for name,sl in (('out_0',slice(1025,None)),('out_1',slice(4100,4128)),('out_1',slice(6178,None))):
        if a[name][sl].tobytes()!=arrays[name][sl].tobytes(): raise AssertionError('tail changed: '+name)

def write_once(path,payload):
    path=Path(path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    temporary='.sinter-receipt-'+os.urandom(16).hex();published=False
    try:
        fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600,dir_fd=directory)
        with os.fdopen(fd,'w') as out:
            json.dump(payload,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
        os.link(temporary,path.name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
        published=True
        os.unlink(temporary,dir_fd=directory)
        os.fsync(directory)
    except Exception as original:
        errors=[original]
        if published:
            try: os.unlink(path.name,dir_fd=directory);os.fsync(directory)
            except Exception as rollback: errors.append(rollback)
        try: os.unlink(temporary,dir_fd=directory)
        except FileNotFoundError: pass
        except Exception as cleanup: errors.append(cleanup)
        if len(errors)>1: raise ExceptionGroup('receipt publication failed',errors)
        raise
    finally: os.close(directory)

class ArtifactJournal:
    """Unique private directory; fixed filenames and durable, append-only events."""
    def __init__(self, receipt):
        self.root=Path(str(receipt)+'.artifacts')
        os.mkdir(self.root,0o700)  # Refuse existing directories and symlinks.
        parent=os.open(self.root.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: os.fsync(parent)
        finally: os.close(parent)
        self.directory=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: self.events=os.open('events.jsonl',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW,0o600,dir_fd=self.directory)
        except Exception:
            os.close(self.directory)
            raise
    def put(self,name,data):
        if '/' in name or name in ('','.','..'): raise ValueError('artifact name')
        fd=os.open(name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600,dir_fd=self.directory)
        with os.fdopen(fd,'wb') as out: out.write(data);out.flush();os.fsync(out.fileno())
        os.fsync(self.directory)
        self.event('artifact',name=name,size=len(data),sha256=sha(data))
    def readback(self,name,role):
        try: fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=self.directory)
        except FileNotFoundError: return None
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size>1024*1024: raise ValueError('readback file')
            with os.fdopen(fd,'rb',closefd=False) as source: data=source.read()
            if len(data)!=info.st_size: raise ValueError('readback changed')
            os.fsync(fd);os.fsync(self.directory)
        finally: os.close(fd)
        self.event('readback',role=role,name=name,size=len(data),sha256=sha(data))
        return data
    def event(self,kind,**fields):
        data=(json.dumps({'event':kind,'monotonic_ns':time.monotonic_ns(),**fields},sort_keys=True)+'\n').encode()
        if os.write(self.events,data)!=len(data): raise OSError('short event write')
        os.fsync(self.events)
    def close(self):
        try: os.close(self.events)
        finally: os.close(self.directory)


def lease_event(fd,journal,stage):
    verify_lease(fd)
    st=os.fstat(fd);info=Path('/proc/self/fdinfo',str(fd)).read_text()
    journal.event('lock_verified',stage=stage,pid=os.getpid(),fd=fd,device=st.st_dev,inode=st.st_ino,fdinfo=info)


def captured_process(cmd, journal, label, **kwargs):
    try:
        result=subprocess.run(cmd,check=False,capture_output=True,**kwargs)
    except Exception as exc:
        journal.event('process_exception',label=label,error_type=type(exc).__name__,error=str(exc))
        raise
    def as_bytes(value):
        if value is None: return b''
        return value.encode() if isinstance(value,str) else value
    journal.event('process_exit',label=label,returncode=result.returncode)
    journal.put(label+'.stdout',as_bytes(result.stdout));journal.put(label+'.stderr',as_bytes(result.stderr))
    return result

def verify_lease(fd):
    actual=os.fstat(fd);expected=LOCK.lstat()
    if LOCK.is_symlink() or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino): raise ValueError('lock inode')
    text=Path('/proc/self/fdinfo',str(fd)).read_text()
    entries=re.findall(r'^lock:\s*\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+[0-9a-fA-F]+:[0-9a-fA-F]+:(\d+)\s+0\s+EOF\s*$',text,re.MULTILINE)
    if (str(os.getpid()),str(actual.st_ino)) not in entries: raise ValueError('exclusive lock unavailable')

def _run_impl(args,journal):
    if not args.execute: raise ValueError('explicit execute required after independent review')
    if Path(args.output).exists() or Path(args.output).is_symlink(): raise FileExistsError(args.output)
    if args.uuid!='96d2dba095fd5c00e8492e0610f24e06' or args.driver!=2497233024: raise ValueError('approved RTX 4070 Ti identity required')
    if not 0<=args.driver<=0xffffffff: raise ValueError('driver')
    tool_ids=checked_tool_sources(args.commit);build_native,validate_spirv=load_checked_helpers(args.commit,tool_ids)
    icd_hash=checked_icd(args.icd,args.icd_sha)
    sources=checked_sources(args.commit);arrays,packed=fixture(sources['fixture'])
    compiler=Path(args.slangc).resolve(strict=True);binary=Path(args.binary).resolve(strict=True)
    if sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('compiler identity')
    if sha((Path(__file__).resolve().parent/'point_plane.cpp').read_bytes())!=NATIVE_SOURCE_SHA: raise ValueError('native source identity')
    if args.binary_sha!=NATIVE_BINARY_SHA or sha(binary.read_bytes())!=NATIVE_BINARY_SHA: raise ValueError('binary identity')
    committed_native=subprocess.check_output(['git','-C',str(ROOT),'show',args.commit+':tools/diagnostics/direct_vulkan/engineering_attempt/point_plane.cpp'])
    if build_native(committed_native)!=binary.read_bytes(): raise ValueError('native source/build mismatch')
    fd=os.open(LOCK,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    primary=None;close_error=None;event_error=None;payload=None;acquired=False
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
        lease_event(fd,journal,'acquired')
        journal.put('input.bin',packed)
        with tempfile.TemporaryDirectory(prefix='sinter-direct-point-plane-') as temp:
            d=Path(temp);(d/'optimization').mkdir();(d/'optimization/point_plane.slang').write_bytes(sources['core'])
            inp=journal.root/'input.bin';results={};metadata={}
            for role,groups,entry in (('donor',2,'computeMain'),('candidate',5,'main')):
                source=d/(role+'.slang');source.write_bytes(sources[role]);spv=d/(role+'.spv')
                cmd=[str(compiler),str(source),'-I',str(d),'-entry',entry,'-target','spirv','-o',str(spv)]
                compiled=captured_process(cmd,journal,role+'-compiler')
                if compiled.returncode!=0: raise subprocess.CalledProcessError(compiled.returncode,cmd)
                if sha(spv.read_bytes())!=SPV_SHA[role]: raise ValueError('compiled shader mismatch: '+role)
                metadata[role+'_abi']=validate_spirv(spv.read_bytes(),1024 if role=='donor' else 256)
                output=journal.root/(role+'.bin')
                call=[str(binary),args.uuid,str(args.driver),str(spv),SPV_SHA[role],'main',str(groups),str(inp),str(output),str(fd)]
                lease_event(fd,journal,role+'-before')
                result=captured_process(call,journal,role+'-native',text=True,pass_fds=(fd,))
                lease_event(fd,journal,role+'-after')
                raw=journal.readback(output.name,role)
                if result.returncode!=0: raise subprocess.CalledProcessError(result.returncode,call)
                if raw is None: raise ValueError('native output unavailable')
                results[role]=decode(raw);metadata[role]=result.stdout.strip()
            validate(arrays,results['donor'],results['candidate'])
            if sha(binary.read_bytes())!=args.binary_sha or sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('binary changed')
            if checked_tool_sources(args.commit)!=tool_ids or checked_icd(args.icd,args.icd_sha)!=icd_hash: raise ValueError('tool or ICD configuration changed')
            lease_event(fd,journal,'validated')
            payload={'artifact_directory':str(journal.root),'device_uuid':args.uuid,'raw_driver':args.driver,'packed_fixture_sha256':INPUT_SHA,'tool_source_sha256':tool_ids,'icd_configuration_sha256':icd_hash,'status':'trusted_host_engineering_parity_only','commit':args.commit,'source_sha256':SOURCE_SHA,
                'spirv_sha256':SPV_SHA,'native_binary_sha256':NATIVE_BINARY_SHA,'native_source_sha256':NATIVE_SOURCE_SHA,'compiler_file_sha256':COMPILER_SHA,
                'device_output':metadata,'result_sha256':{n:sha(results['candidate'][n].tobytes()) for n in NAMES},
                'scope':'trusted workstation; not provider, physical-backing, SlangPy, performance or device qualification'}
            lease_event(fd,journal,'before-close')
    except Exception as exc:
        primary=exc
    finally:
        try: os.close(fd)
        except Exception as exc: close_error=exc
    if not close_error:
        try: journal.event('lock_closed',pid=os.getpid(),acquired=acquired)
        except Exception as exc: event_error=exc
    errors=[error for error in (primary,close_error,event_error) if error is not None]
    if len(errors)>1: raise ExceptionGroup('operation, lock close or evidence failed',errors)
    if errors: raise errors[0]
    return payload


def run(args):
    # Deliberate engineering attempt; independent review and external GPU lock required.
    if Path(args.output).exists() or Path(args.output).is_symlink(): raise FileExistsError(args.output)
    journal=None;created=False
    try:
        journal=ArtifactJournal(args.output);created=True
        payload=_run_impl(args,journal)
        journal.event('validation_passed')
        journal.close();journal=None
    except Exception as exc:
        errors=[exc]
        if journal is not None:
            try: journal.event('failure',error_type=type(exc).__name__,error=str(exc))
            except Exception as log_error: errors.append(log_error)
            try: journal.close()
            except Exception as close_error: errors.append(close_error)
        failure={'status':'failure','error_type':type(exc).__name__,'error':str(exc),'scope':'no GPU qualification'}
        if created: failure['artifact_directory']=str(args.output)+'.artifacts'
        try: write_once(args.output,failure)
        except Exception as receipt_error: errors.append(receipt_error)
        if len(errors)>1: raise ExceptionGroup('operation and evidence handling failed',errors)
        raise
    try: write_once(args.output,payload)
    except Exception as exc:
        try: write_once(args.output,{'status':'failure','error_type':type(exc).__name__,'error':str(exc),
                                    'artifact_directory':str(args.output)+'.artifacts','scope':'no GPU qualification'})
        except Exception as receipt_error: raise ExceptionGroup('success and failure receipt publication failed',[exc,receipt_error])
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for key in ('commit','uuid','binary','binary-sha','slangc','output','icd','icd-sha'):p.add_argument('--'+key,required=True)
    p.add_argument('--driver',type=lambda x:int(x,0),required=True);run(p.parse_args())
if __name__=='__main__':main()
