"""CPU-only contract tests; never opens a GPU."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from benchmarks.depth_pyramid_isolated_pilot import lock_device, verify_expected, write_once, workloads


def test_canonical_lock_is_nonblocking(tmp_path):
    path=tmp_path/'lock'
    with lock_device(path):
        with pytest.raises(BlockingIOError):
            with lock_device(path): pass
    with lock_device(path): pass
    path.unlink(); path.symlink_to(tmp_path/'target')
    with pytest.raises(OSError):
        with lock_device(path): pass


def test_expected_identity_must_match_actual_snapshot_and_compiler():
    snapshot={'physical_device_uuid':'a','driver':{'version':'v','info':'i'}}
    verify_expected(snapshot,'a','v','i','f'*64,'f'*64)
    for args in [('b','v','i','f'*64),('a','x','i','f'*64),('a','v','x','f'*64),('a','v','i','e'*64)]:
        with pytest.raises(ValueError): verify_expected(snapshot,*args,'f'*64)


def test_write_once_never_overwrites(tmp_path):
    path=tmp_path/'result.json'
    write_once(path,{'scope':'unqualified','samples':[1.]})
    with pytest.raises(FileExistsError): write_once(path,{'changed':True})
    assert json.loads(path.read_text())['samples']==[1.]


def test_predeclared_pilot_workloads():
    w=workloads()
    assert [(x['profile'],x['height'],x['width']) for x in w] == [('vkl',121,161),('slam',120,160)]


def test_runner_requires_explicit_execution_before_device_import():
    import types
    from benchmarks.depth_pyramid_isolated_pilot import run
    with pytest.raises(ValueError,match='explicit --execute'):
        run(types.SimpleNamespace(execute=False))


def test_committed_closure_rejects_changed_source(monkeypatch,tmp_path):
    import benchmarks.depth_pyramid_isolated_pilot as pilot
    commit='a'*40; source=tmp_path/'one.slang'; source.write_bytes(b'original')
    monkeypatch.setattr(pilot,'ROOT',tmp_path); monkeypatch.setattr(pilot,'CLOSURE',('one.slang',))
    def git(args,**kw):
        return commit+'\n' if kw.get('text') else b'original'
    monkeypatch.setattr(pilot.subprocess,'check_output',git)
    assert pilot.committed_closure(commit)['one.slang']==hashlib.sha256(b'original').hexdigest()
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs from committed'): pilot.committed_closure(commit)
    with pytest.raises(ValueError,match='commit SHA'): pilot.committed_closure('main')
