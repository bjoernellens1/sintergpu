# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import struct
import subprocess
import pytest

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('contract',HERE/'spirv_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COMPILER=Path('/home/bjoern/.local/opt/slang-2026.19/bin/slangc')

def mutate(data,op,predicate,change):
    words=list(struct.unpack('<%dI'%(len(data)//4),data));i=5
    while i<len(words):
        count=words[i]>>16;kind=words[i]&0xffff
        if kind==op and predicate(words[i+1:i+count]):
            change(words,i);return struct.pack('<%dI'%len(words),*words)
        i+=count
    raise AssertionError('mutation target missing')

@pytest.mark.parametrize('name,entry,group',[('caspar_slang/composed/point_plane_residual.slang','computeMain',1024),('optimization/point_plane_aosoa_256.slang','main',256)])
def test_real_compiler_and_abi_mutations(tmp_path,name,entry,group):
    if not COMPILER.is_file(): pytest.skip('local compiler unavailable')
    out=tmp_path/'shader.spv';kernels=ROOT/'src/sintergpu/kernels'
    subprocess.run([str(COMPILER),str(kernels/name),'-I',str(kernels),'-entry',entry,'-target','spirv','-o',str(out)],check=True)
    data=out.read_bytes();assert m.validate(data,group)['uniform_allocated_bytes']==32
    with pytest.raises(ValueError):m.validate(data,group//2)
    corrupt=mutate(data,71,lambda a:len(a)==3 and a[1]==33 and a[2]==6,lambda w,i:w.__setitem__(i+3,7))
    with pytest.raises(ValueError):m.validate(corrupt,group)
    corrupt=mutate(data,72,lambda a:len(a)==4 and a[1:]==[5,35,20],lambda w,i:w.__setitem__(i+4,24))
    with pytest.raises(ValueError):m.validate(corrupt,group)
    words=struct.unpack('<%dI'%(len(data)//4),data);i=5;runtime=[]
    while i<len(words):
        count=words[i]>>16
        if words[i]&0xffff==29: runtime.append(words[i+1])
        i+=count
    assert runtime
    corrupt=mutate(data,71,lambda a:len(a)==3 and a[0] in runtime and a[1:]==[6,4],lambda w,i:w.__setitem__(i+3,8))
    with pytest.raises(ValueError):m.validate(corrupt,group)
    with pytest.raises(ValueError):m.validate(data[:-4],group)
