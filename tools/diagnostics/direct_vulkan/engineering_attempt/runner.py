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
from pathlib import Path
import subprocess
import tempfile
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
NATIVE_BINARY_SHA='44b5e6b2d27e159201bce180984b4a1450c1b1c8dcd745ab8d04f3c4ce7fe301'
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
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as out:
        json.dump(payload,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())

def verify_lease(fd):
    actual=os.fstat(fd);expected=LOCK.lstat()
    if LOCK.is_symlink() or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino): raise ValueError('lock inode')
    text=Path('/proc/self/fdinfo',str(fd)).read_text()
    entries=re.findall(r'^lock:\s*\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+[0-9a-fA-F]+:[0-9a-fA-F]+:(\d+)\s+0\s+EOF\s*$',text,re.MULTILINE)
    if (str(os.getpid()),str(actual.st_ino)) not in entries: raise ValueError('exclusive lock unavailable')

def _run_impl(args):
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
    primary=None;close_error=None;payload=None
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_lease(fd)
        with tempfile.TemporaryDirectory(prefix='sinter-direct-point-plane-') as temp:
            d=Path(temp);(d/'optimization').mkdir();(d/'optimization/point_plane.slang').write_bytes(sources['core'])
            inp=d/'input.bin';inp.write_bytes(packed);results={};metadata={}
            for role,groups,entry in (('donor',2,'computeMain'),('candidate',5,'main')):
                source=d/(role+'.slang');source.write_bytes(sources[role]);spv=d/(role+'.spv')
                cmd=[str(compiler),str(source),'-I',str(d),'-entry',entry,'-target','spirv','-o',str(spv)]
                subprocess.run(cmd,check=True,capture_output=True)
                if sha(spv.read_bytes())!=SPV_SHA[role]: raise ValueError('compiled shader mismatch: '+role)
                metadata[role+'_abi']=validate_spirv(spv.read_bytes(),1024 if role=='donor' else 256)
                output=d/(role+'.bin')
                call=[str(binary),args.uuid,str(args.driver),str(spv),SPV_SHA[role],'main',str(groups),str(inp),str(output),str(fd)]
                verify_lease(fd)
                result=subprocess.run(call,check=True,capture_output=True,text=True,pass_fds=(fd,))
                verify_lease(fd)
                results[role]=decode(output.read_bytes());metadata[role]=result.stdout.strip()
            validate(arrays,results['donor'],results['candidate'])
            if sha(binary.read_bytes())!=args.binary_sha or sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('binary changed')
            if checked_tool_sources(args.commit)!=tool_ids or checked_icd(args.icd,args.icd_sha)!=icd_hash: raise ValueError('tool or ICD configuration changed')
            verify_lease(fd)
            payload={'device_uuid':args.uuid,'raw_driver':args.driver,'packed_fixture_sha256':INPUT_SHA,'tool_source_sha256':tool_ids,'icd_configuration_sha256':icd_hash,'status':'trusted_host_engineering_parity_only','commit':args.commit,'source_sha256':SOURCE_SHA,
                'spirv_sha256':SPV_SHA,'native_binary_sha256':NATIVE_BINARY_SHA,'native_source_sha256':NATIVE_SOURCE_SHA,'compiler_file_sha256':COMPILER_SHA,
                'device_output':metadata,'result_sha256':{n:sha(results['candidate'][n].tobytes()) for n in NAMES},
                'scope':'trusted workstation; not provider, physical-backing, SlangPy, performance or device qualification'}
            verify_lease(fd)
    except Exception as exc:
        primary=exc
    finally:
        try: os.close(fd)
        except Exception as exc: close_error=exc
    if primary and close_error: raise ExceptionGroup('operation and lock close failed',[primary,close_error])
    if primary: raise primary
    if close_error: raise close_error
    return payload


def run(args):
    # Deliberate engineering attempt; independent review and external GPU lock required.
    if Path(args.output).exists() or Path(args.output).is_symlink(): raise FileExistsError(args.output)
    try:
        payload=_run_impl(args)
    except Exception as exc:
        try: write_once(args.output,{'status':'failure','error_type':type(exc).__name__,'error':str(exc),'scope':'no GPU qualification'})
        except Exception as receipt_error: raise ExceptionGroup('operation and failure receipt failed',[exc,receipt_error])
        raise
    write_once(args.output,payload)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for key in ('commit','uuid','binary','binary-sha','slangc','output','icd','icd-sha'):p.add_argument('--'+key,required=True)
    p.add_argument('--driver',type=lambda x:int(x,0),required=True);run(p.parse_args())
if __name__=='__main__':main()
