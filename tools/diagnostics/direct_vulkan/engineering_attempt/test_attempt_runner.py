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
    monkeypatch.setattr(r,'checked_tool_sources',lambda commit:{'test':'mock'})
    monkeypatch.setattr(r,'checked_icd',lambda path,expected:'mock')
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'fake committed source')
    monkeypatch.setattr(r,'load_checked_helpers',lambda commit,hashes:(lambda source:b'native',lambda *args:{'synthetic':True}))
    monkeypatch.setattr(r,'checked_sources',lambda commit:{'fixture':FIXTURE.read_bytes(),'core':b'core','donor':b'donor','candidate':b'candidate'})
    monkeypatch.setattr(r,'SPV_SHA',{'donor':r.sha(b'donor-spv'),'candidate':r.sha(b'candidate-spv')})
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
            return SimpleNamespace(stdout='',stderr='',returncode=0)
        assert command[0]==str(binary) and kw['pass_fds']
        native_fd[:]=[kw['pass_fds'][0]]
        assert Path(command[7]).read_bytes()==packed
        Path(command[8]).write_bytes(b''.join(out[n].tobytes() for n in r.NAMES))
        return SimpleNamespace(stdout='fake device',stderr='',returncode=0)
    monkeypatch.setattr(r.subprocess,'run',fake_run)
    args=SimpleNamespace(execute=True,output=str(tmp_path/'result.json'),uuid='96d2dba095fd5c00e8492e0610f24e06',driver=2497233024,commit='a'*40,
                         slangc=str(compiler),binary=str(binary),binary_sha=r.sha(b'native'),icd='/fake/icd.json',icd_sha='mock')
    r.run(args);assert len(calls)==4 and Path(args.output).is_file()
    evidence=Path(args.output+'.artifacts')
    assert (evidence/'input.bin').read_bytes()==packed
    assert (evidence/'donor.bin').read_bytes()==(evidence/'candidate.bin').read_bytes()
    events=[__import__('json').loads(line) for line in (evidence/'events.jsonl').read_text().splitlines()]
    assert [e['stage'] for e in events if e['event']=='lock_verified']==['acquired','donor-before','donor-after','candidate-before','candidate-after','validated','before-close']
    assert [e['returncode'] for e in events if e['event']=='process_exit']==[0,0,0,0]
    assert events[-1]['event']=='validation_passed' and events[-2]['event']=='lock_closed'
    with pytest.raises(FileExistsError):r.run(args)
    # A failed native process retains any partial bytes, stderr and exit code.
    args.output=str(tmp_path/'partial.json')
    def partial(command,**kw):
        if command[0]==str(compiler):return fake_run(command,**kw)
        Path(command[8]).write_bytes(b'partial')
        return SimpleNamespace(stdout='partial out',stderr='device failed',returncode=17)
    monkeypatch.setattr(r.subprocess,'run',partial)
    with pytest.raises(subprocess.CalledProcessError):r.run(args)
    partial_dir=Path(args.output+'.artifacts')
    assert (partial_dir/'donor.bin').read_bytes()==b'partial'
    assert (partial_dir/'donor-native.stderr').read_bytes()==b'device failed'
    partial_events=[__import__('json').loads(line) for line in (partial_dir/'events.jsonl').read_text().splitlines()]
    assert any(e['event']=='process_exit' and e['label']=='donor-native' and e['returncode']==17 for e in partial_events)
    assert not (partial_dir/'candidate.bin').exists()
    # Journal write failure preserves the raw buffer already written by native.
    args.output=str(tmp_path/'journal-failure.json');monkeypatch.setattr(r.subprocess,'run',fake_run)
    original_put=r.ArtifactJournal.put
    def failing_put(self,name,data):
        if name=='donor-native.stderr': raise OSError('journal write failed')
        return original_put(self,name,data)
    monkeypatch.setattr(r.ArtifactJournal,'put',failing_put)
    with pytest.raises(OSError,match='journal write failed'):r.run(args)
    assert (Path(args.output+'.artifacts')/'donor.bin').is_file()
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'
    monkeypatch.setattr(r.ArtifactJournal,'put',original_put)
    args.output=str(tmp_path/'failure.json')
    def fail(*a,**kw):raise subprocess.CalledProcessError(1,a[0])
    monkeypatch.setattr(r.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):r.run(args)
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'
    # A lock-descriptor teardown failure cannot leave a success receipt.
    args.output=str(tmp_path/'close-failure.json');monkeypatch.setattr(r.subprocess,'run',fake_run)
    original_close=os.close
    def close_then_fail(fd):
        original_close(fd)
        if native_fd and fd==native_fd[0]: raise OSError('simulated lock close failure')
    monkeypatch.setattr(r.os,'close',close_then_fail)
    with pytest.raises(OSError,match='lock close failure'):r.run(args)
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'
    monkeypatch.setattr(r.os,'close',original_close)
    args.output=str(tmp_path/'success-write-failure.json')
    original_write=r.write_once
    def fail_success(path,payload):
        if payload['status']=='trusted_host_engineering_parity_only': raise OSError('success receipt failed')
        return original_write(path,payload)
    monkeypatch.setattr(r,'write_once',fail_success)
    with pytest.raises(OSError,match='success receipt failed'):r.run(args)
    assert __import__('json').loads(Path(args.output).read_text())['status']=='failure'
    assert (Path(args.output+'.artifacts')/'candidate.bin').is_file()


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
    with pytest.raises(ExceptionGroup) as info:r.run(SimpleNamespace(output=str(tmp_path/'out')))
    assert [str(e) for e in info.value.exceptions]==['original error','receipt error']

def test_icd_environment_and_file_hash(tmp_path,monkeypatch):
    icd=tmp_path/'icd.json';icd.write_text('{"ICD":{}}')
    monkeypatch.setenv('VK_DRIVER_FILES',str(icd))
    for k in ('VK_ICD_FILENAMES','VK_ADD_DRIVER_FILES','VK_LAYER_PATH','VK_ADD_LAYER_PATH','VK_INSTANCE_LAYERS','LD_PRELOAD'):monkeypatch.delenv(k,raising=False)
    digest=r.sha(icd.read_bytes());assert r.checked_icd(str(icd),digest)==digest
    monkeypatch.setenv('LD_PRELOAD','untrusted.so')
    with pytest.raises(ValueError,match='environment'):r.checked_icd(str(icd),digest)
    monkeypatch.delenv('LD_PRELOAD');icd.write_text('changed')
    with pytest.raises(ValueError,match='hash'):r.checked_icd(str(icd),digest)

def test_tool_sources_must_match_commit(monkeypatch):
    paths={p.relative_to(r.ROOT).as_posix():p.read_bytes() for p in (HERE/'runner.py',HERE/'verified_build.py',HERE/'spirv_contract.py')}
    def read_git(cmd):return paths[cmd[-1].split(':',1)[1]]
    monkeypatch.setattr(r.subprocess,'check_output',read_git)
    assert len(r.checked_tool_sources('a'*40))==3
    paths[(HERE/'runner.py').relative_to(r.ROOT).as_posix()]=b'changed'
    with pytest.raises(ValueError,match='differs from commit'):r.checked_tool_sources('a'*40)

def test_checked_helpers_do_not_execute_timestamp_matched_stale_pyc(tmp_path,monkeypatch):
    import py_compile
    directory=tmp_path/'helpers';directory.mkdir();marker=tmp_path/'marker'
    runner=directory/'runner.py';runner.write_text('# runner\n')
    build=directory/'verified_build.py';contract=directory/'spirv_contract.py'
    good="def build(x): return b'ok'\n"
    bad=f"open({str(marker)!r}, 'w').write('x')\n"+good
    good=good+'#'*(len(bad)-len(good))
    build.write_text(bad);saved=build.stat();py_compile.compile(str(build),doraise=True)
    build.write_text(good);os.utime(build,ns=(saved.st_atime_ns,saved.st_mtime_ns))
    contract.write_text('def validate(x, g): return g\n')
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'__file__',str(runner))
    files={p.relative_to(tmp_path).as_posix():p.read_bytes() for p in (runner,build,contract)}
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:files[cmd[-1].split(':',1)[1]])
    identities=r.checked_tool_sources('a'*40);build_fn,validate_fn=r.load_checked_helpers('a'*40,identities)
    assert build_fn(None)==b'ok' and validate_fn(None,3)==3 and not marker.exists()
    files[build.relative_to(tmp_path).as_posix()]=b'modified'
    with pytest.raises(ValueError,match='helper source changed'):r.load_checked_helpers('a'*40,identities)


def test_receipt_publication_fsync_failure_does_not_publish(tmp_path,monkeypatch):
    out=tmp_path/'receipt.json'
    def fail(fd): raise OSError('fsync failed')
    monkeypatch.setattr(r.os,'fsync',fail)
    with pytest.raises(OSError,match='fsync failed'):r.write_once(out,{'status':'success'})
    assert not out.exists() and not list(tmp_path.glob('.sinter-receipt-*'))

def test_journal_refuses_reuse_and_symlink(tmp_path):
    receipt=tmp_path/'receipt';j=r.ArtifactJournal(receipt)
    try:
        j.put('input.bin',b'input')
        with pytest.raises(FileExistsError):j.put('input.bin',b'changed')
        (j.root/'donor.bin').symlink_to(j.root/'input.bin')
        with pytest.raises(OSError):j.readback('donor.bin','donor')
        with pytest.raises(FileExistsError):r.ArtifactJournal(receipt)
    finally:j.close()

def test_journal_syncs_directory_on_event_creation(tmp_path,monkeypatch):
    synced=[];real_fsync=os.fsync
    def tracking(fd):
        st=os.fstat(fd)
        if __import__('stat').S_ISDIR(st.st_mode):synced.append((st.st_dev,st.st_ino))
        return real_fsync(fd)
    monkeypatch.setattr(r.os,'fsync',tracking)
    journal=r.ArtifactJournal(tmp_path/'receipt')
    try:
        for path in (tmp_path,journal.root):
            st=path.stat();assert (st.st_dev,st.st_ino) in synced
    finally:journal.close()
