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
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'fake committed source')
    monkeypatch.setattr(r,'build_native',lambda source:b'native')
    monkeypatch.setattr(r,'checked_sources',lambda commit:{'fixture':FIXTURE.read_bytes(),'core':b'core','donor':b'donor','candidate':b'candidate'})
    monkeypatch.setattr(r,'SPV_SHA',{'donor':r.sha(b'donor-spv'),'candidate':r.sha(b'candidate-spv')})
    monkeypatch.setattr(r,'validate_spirv',lambda *args:{'synthetic':True})
    calls=[]; native_fd=[]
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
        native_fd[:]=[kw['pass_fds'][0]]
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
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'
    # A lock-descriptor teardown failure cannot leave a success receipt.
    args.output=str(tmp_path/'close-failure.json');monkeypatch.setattr(r.subprocess,'run',fake_run)
    original_close=os.close
    def close_then_fail(fd):
        original_close(fd)
        if native_fd and fd==native_fd[0]: raise OSError('simulated lock close failure')
    monkeypatch.setattr(r.os,'close',close_then_fail)
    with pytest.raises(OSError,match='lock close failure'):r.run(args,_cpu_test_only=True)
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'


def test_production_gate_precedes_everything(monkeypatch):
    monkeypatch.setattr(r,'checked_sources',lambda *args: pytest.fail('must not inspect or create device'))
    with pytest.raises(RuntimeError, match='production path blocked'):
        r.run(object())

def test_lease_verification_rejects_unlocked_and_other_inode(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'LOCK',tmp_path/'lock')
    fd=os.open(r.LOCK,os.O_CREAT|os.O_RDWR,0o600)
    try:
        with pytest.raises(ValueError,match='exclusive lock'):r.verify_lease(fd)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);r.verify_lease(fd)
        other=tmp_path/'other';wrong=os.open(other,os.O_CREAT|os.O_RDWR,0o600)
        try:
            with pytest.raises(ValueError,match='inode'):r.verify_lease(wrong)
        finally:os.close(wrong)
    finally:os.close(fd)

def test_failure_receipt_write_error_preserves_both_exceptions(tmp_path,monkeypatch):
    def fail(*a,**k):raise ValueError('original error')
    def receipt(*a,**k):raise OSError('receipt error')
    monkeypatch.setattr(r,'_run_impl',fail);monkeypatch.setattr(r,'write_once',receipt)
    with pytest.raises(ExceptionGroup) as info:r.run(SimpleNamespace(output=str(tmp_path/'out')),_cpu_test_only=True)
    assert [str(e) for e in info.value.exceptions]==['original error','receipt error']
