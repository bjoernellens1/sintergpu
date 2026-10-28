# SPDX-License-Identifier: Apache-2.0
"""CPU mocks only; no GPU or native driver execution."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import fcntl
import os
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sinter_direct_point_plane',HERE/'runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
FIXTURE=r.ROOT/r.PATHS['fixture']

def simulated_output(arrays):
    out={k:v.copy() for k,v in arrays.items()}
    p=arrays['points'][arrays['points_indices'],:3].astype(np.float64)
    q=arrays['targets'][:1025,:3].astype(np.float64);n=arrays['normals'][:1025,:3].astype(np.float64)
    out['out_0'][:1025]=np.sum(n*(p-q),axis=1)
    jac=np.c_[np.cross(p,n),n].astype(np.float32)
    out['out_1'][:4100]=jac[:,:4].reshape(-1)
    out['out_1'][4128:6178]=jac[:,4:].reshape(-1)
    return out

def test_fixture_oracle_and_guards():
    arrays,packed=r.fixture(FIXTURE.read_bytes());out=simulated_output(arrays)
    r.validate(arrays,out,out)
    decoded=r.decode(b''.join(out[n].tobytes() for n in r.NAMES));r.validate(arrays,out,decoded)
    bad={k:v.copy() for k,v in out.items()};bad['out_1'][-1]=0
    with pytest.raises(AssertionError):r.validate(arrays,bad,bad)
    bad={k:v.copy() for k,v in out.items()};bad['points'][0,0]=9
    with pytest.raises(AssertionError):r.validate(arrays,bad,bad)
    bad={k:v.copy() for k,v in out.items()};bad['out_0'][5]+=1
    with pytest.raises(AssertionError):r.validate(arrays,bad,out)
    with pytest.raises(AssertionError):r.decode(packed[:-1])

def test_orchestration_lock_parity_and_failure(tmp_path,monkeypatch):
    arrays,packed=r.fixture(FIXTURE.read_bytes());out=simulated_output(arrays)
    binary=tmp_path/'native';binary.write_bytes(b'native')
    compiler=tmp_path/'slangc';compiler.write_bytes(b'compiler')
    monkeypatch.setattr(r,'COMPILER_SHA',r.sha(b'compiler'))
    monkeypatch.setattr(r,'NATIVE_BINARY_SHA',r.sha(b'native'))
    monkeypatch.setattr(r,'LOCK',tmp_path/'lock')
    monkeypatch.setattr(r,'checked_sources',lambda commit:{'fixture':FIXTURE.read_bytes(),'core':b'core','donor':b'donor','candidate':b'candidate'})
    monkeypatch.setattr(r,'SPV_SHA',{'donor':r.sha(b'donor-spv'),'candidate':r.sha(b'candidate-spv')})
    calls=[]
    def fake_run(command, **kw):
        calls.append(command)
        # A competing open-file description cannot acquire the lock during compilation or dispatch.
        competing=os.open(r.LOCK,os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(competing)
        if command[0]==str(compiler):
            role=Path(command[1]).stem;Path(command[-1]).write_bytes((role+'-spv').encode())
            return SimpleNamespace(stdout='')
        assert command[0]==str(binary) and kw['pass_fds']
        assert Path(command[7]).read_bytes()==packed
        Path(command[8]).write_bytes(b''.join(out[n].tobytes() for n in r.NAMES))
        return SimpleNamespace(stdout='fake device')
    monkeypatch.setattr(r.subprocess,'run',fake_run)
    args=SimpleNamespace(execute=True,output=str(tmp_path/'result.json'),uuid='01'*16,driver=4,commit='a'*40,
                         slangc=str(compiler),binary=str(binary),binary_sha=r.sha(b'native'))
    r.run(args,_cpu_test_only=True);assert len(calls)==4 and Path(args.output).is_file()
    with pytest.raises(FileExistsError):r.run(args,_cpu_test_only=True)
    args.output=str(tmp_path/'failure.json')
    def fail(*a,**kw):raise subprocess.CalledProcessError(1,a[0])
    monkeypatch.setattr(r.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):r.run(args,_cpu_test_only=True)
    assert not Path(args.output).exists()


def test_production_gate_precedes_everything(monkeypatch):
    monkeypatch.setattr(r,'checked_sources',lambda *args: pytest.fail('must not inspect or create device'))
    with pytest.raises(RuntimeError, match='production path blocked'):
        r.run(object())
