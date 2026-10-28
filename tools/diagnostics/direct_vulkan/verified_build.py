# SPDX-License-Identifier: Apache-2.0
"""Rebuild the blocked native scaffold twice; not toolchain or runtime attestation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
SOURCE='tools/diagnostics/direct_vulkan/point_plane.cpp'
SOURCE_SHA='24c32e16b2d560514925d70ae6d61ced4a051488ff544a32a54a6f8a7366be32'
BINARY_SHA='c623778a0db171127b20e21bbd3ac63bbcfcae8d79691b60a4857c1a7a45f06b'
COMPILER=Path('/usr/bin/c++')
COMPILER_SHA='579d25c2d1e45abeed0c63a517b5c52f6d09ea811a787bf7a7c677a7fd0726f6'
FLAGS=['-std=c++17','-O0','-g0','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers']

def sha(data): return hashlib.sha256(data).hexdigest()
def build(source, compiler=COMPILER):
    if sha(source)!=SOURCE_SHA: raise ValueError('native source mismatch')
    if sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('native compiler mismatch')
    with tempfile.TemporaryDirectory(prefix='sinter-native-build-') as tmp:
        d=Path(tmp);src=d/'point_plane.cpp';src.write_bytes(source);built=[]
        for i in range(2):
            target=d/('probe'+str(i))
            subprocess.run([str(compiler),*FLAGS,str(src),'-o',str(target),'-lvulkan','-lcrypto'],check=True,capture_output=True,env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
            built.append(target.read_bytes())
        if built[0]!=built[1] or sha(built[0])!=BINARY_SHA: raise ValueError('native binary mismatch')
        if sha(src.read_bytes())!=SOURCE_SHA or sha(compiler.read_bytes())!=COMPILER_SHA: raise ValueError('build inputs changed')
        return built[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True);args=p.parse_args()
    if len(args.commit)!=40 or any(c not in '0123456789abcdef' for c in args.commit): raise ValueError('full commit required')
    if any(Path(x).exists() or Path(x).is_symlink() for x in (args.output,args.receipt)): raise FileExistsError('output exists')
    source=subprocess.check_output(['git','-C',str(ROOT),'show',args.commit+':'+SOURCE])
    binary=build(source)
    fd=os.open(args.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o700)
    with os.fdopen(fd,'wb') as f: f.write(binary);f.flush();os.fsync(f.fileno())
    payload={'status':'blocked_scaffold_build_only','commit':args.commit,'source_sha256':SOURCE_SHA,'binary_sha256':BINARY_SHA,
             'compiler_path':str(COMPILER),'compiler_file_sha256':COMPILER_SHA,'flags':FLAGS,
             'scope':'two same-host builds; compiler dependencies, linker, libraries and mapped runtime not authenticated'}
    fd=os.open(args.receipt,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f: json.dump(payload,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
