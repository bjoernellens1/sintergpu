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
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
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
# This pin is a blocked local diagnostic build; a future reviewed executable
# requires a separately reviewed source/build provenance and updated pins.
NATIVE_SOURCE_SHA='777ae9d98ff5eba0f43e3ce230dc8ed655ec71946b7d7217b2128fc5f53064d5'
NATIVE_BINARY_SHA='d2c6a0b7427fc2cb3b3c5b53209f71798ca22db89980bcb4dff1c6f85bcae16f'
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

def run(args, *, _cpu_test_only=False):
    # Hard gate: an injected CPU test must never be used for GPU execution.
    if _cpu_test_only is not True:
        raise RuntimeError('direct Vulkan production path blocked pending review and authenticated executable')
    if not args.execute: raise ValueError('explicit execute required after independent review')
    if Path(args.output).exists() or Path(args.output).is_symlink(): raise FileExistsError(args.output)
    if len(args.uuid)!=32 or any(x not in '0123456789abcdef' for x in args.uuid) or int(args.uuid,16)==0: raise ValueError('UUID')
    if not 0<=args.driver<=0xffffffff: raise ValueError('driver')
    sources=checked_sources(args.commit);arrays,packed=fixture(sources['fixture'])
    compiler=Path(args.slangc).resolve(strict=True);binary=Path(args.binary).resolve(strict=True)
    if sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('compiler identity')
    if sha((Path(__file__).resolve().parent/'point_plane.cpp').read_bytes())!=NATIVE_SOURCE_SHA: raise ValueError('native source identity')
    if args.binary_sha!=NATIVE_BINARY_SHA or sha(binary.read_bytes())!=NATIVE_BINARY_SHA: raise ValueError('binary identity')
    fd=os.open(LOCK,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='sinter-direct-point-plane-') as temp:
            d=Path(temp);(d/'optimization').mkdir();(d/'optimization/point_plane.slang').write_bytes(sources['core'])
            inp=d/'input.bin';inp.write_bytes(packed);results={};metadata={}
            for role,groups,entry in (('donor',2,'computeMain'),('candidate',5,'main')):
                source=d/(role+'.slang');source.write_bytes(sources[role]);spv=d/(role+'.spv')
                cmd=[str(compiler),str(source),'-I',str(d),'-entry',entry,'-target','spirv','-o',str(spv)]
                subprocess.run(cmd,check=True,capture_output=True)
                if sha(spv.read_bytes())!=SPV_SHA[role]: raise ValueError('compiled shader mismatch: '+role)
                output=d/(role+'.bin')
                call=[str(binary),args.uuid,str(args.driver),str(spv),SPV_SHA[role],'main',str(groups),str(inp),str(output),str(fd)]
                result=subprocess.run(call,check=True,capture_output=True,text=True,pass_fds=(fd,))
                results[role]=decode(output.read_bytes());metadata[role]=result.stdout.strip()
            validate(arrays,results['donor'],results['candidate'])
            if sha(binary.read_bytes())!=args.binary_sha or sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('binary changed')
            write_once(args.output,{'status':'engineering_parity_only','commit':args.commit,'source_sha256':SOURCE_SHA,
                'spirv_sha256':SPV_SHA,'native_binary_sha256':NATIVE_BINARY_SHA,'native_source_sha256':NATIVE_SOURCE_SHA,'compiler_file_sha256':COMPILER_SHA,
                'device_output':metadata,'result_sha256':{n:sha(results['candidate'][n].tobytes()) for n in NAMES},
                'scope':'trusted workstation; not provider, physical-backing, SlangPy, performance or device qualification'})
    finally:
        os.close(fd)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for key in ('commit','uuid','binary','binary-sha','slangc','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--driver',type=lambda x:int(x,0),required=True);run(p.parse_args())
if __name__=='__main__':main()
