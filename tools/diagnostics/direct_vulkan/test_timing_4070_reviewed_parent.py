# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
import timing_4070_reviewed_parent as enabled
import timing_4070_parent as blocked

def test_explicit_execute_precedes_all_io(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('unexpected source/device access')
    monkeypatch.setattr(enabled,'committed_sources',forbidden)
    with pytest.raises(ValueError,match='explicit execute required'):
        enabled.run(SimpleNamespace(execute=False))
    with pytest.raises(RuntimeError,match='blocked pending independent review'):
        blocked.run(None)

def test_exact_target_pin_and_full_source_includes():
    assert enabled.NATIVE_SHA=='f8fbce518a37ee0a22e4b74700820b6aa526bfbab273e9984620dffce3ffdca5'
    assert 'timing_4070_reviewed_parent.py' in enabled.LOCAL
    assert 'timing_4070_trusted.cpp' in enabled.LOCAL

def test_source_setup_failure_retained_without_device(tmp_path,monkeypatch):
    def fail(_):raise ValueError('bad source closure')
    monkeypatch.setattr(enabled,'committed_sources',fail)
    args=SimpleNamespace(execute=True,output=tmp_path/'receipt.json',commit='a'*40)
    with pytest.raises(ValueError,match='bad source closure'):enabled.run(args)
    assert 'source_or_journal_setup' in args.output.read_text()

def test_existing_receipt_fails_before_source_access(tmp_path,monkeypatch):
    path=tmp_path/'receipt';path.write_text('preserve')
    monkeypatch.setattr(enabled,'committed_sources',lambda *args:(_ for _ in ()).throw(AssertionError('must not run')))
    with pytest.raises(FileExistsError):enabled.run(SimpleNamespace(execute=True,output=path))
    assert path.read_text()=='preserve'


def test_filesystem_capacity_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(enabled.os,'statvfs',lambda p:SimpleNamespace(f_bavail=1,f_frsize=4096))
    with pytest.raises(OSError,match='insufficient artifact filesystem capacity'):enabled.require_capacity(tmp_path/'result')
    monkeypatch.setattr(enabled.os,'statvfs',lambda p:SimpleNamespace(f_bavail=40000,f_frsize=4096))
    assert enabled.require_capacity(tmp_path/'result')==40000*4096


def test_capacity_event_failure_closes_journal_and_records_failure(tmp_path,monkeypatch):
    import os
    descriptors=[]
    class FailedJournal:
        def __init__(self,path):
            self.fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);descriptors.append(self.fd)
        def event(self,*args,**kwargs):raise OSError('event failure')
        def close(self):os.close(self.fd)
    monkeypatch.setattr(enabled,'committed_sources',lambda c:{'engineering_attempt/runner.py':b'x','timing_report.py':b'x','engineering_attempt/spirv_contract.py':b'x'})
    monkeypatch.setattr(enabled,'load_checked',lambda source,name,path:{'ArtifactJournal':FailedJournal} if name=='sinter_timing_helper' else {})
    monkeypatch.setattr(enabled,'require_capacity',lambda path:128*1024*1024)
    out=tmp_path/'failed.json'
    with pytest.raises(OSError,match='event failure'):enabled.run(SimpleNamespace(execute=True,output=out,commit='a'*40))
    assert 'event failure' in out.read_text()
    with pytest.raises(OSError):os.fstat(descriptors[0])

def test_capacity_event_close_and_receipt_errors_are_preserved(tmp_path,monkeypatch):
    class FailedJournal:
        def __init__(self,path):pass
        def event(self,*args,**kwargs):raise ValueError('event failure')
        def close(self):raise OSError('close failure')
    monkeypatch.setattr(enabled,'committed_sources',lambda c:{'engineering_attempt/runner.py':b'x','timing_report.py':b'x','engineering_attempt/spirv_contract.py':b'x'})
    monkeypatch.setattr(enabled,'load_checked',lambda source,name,path:{'ArtifactJournal':FailedJournal} if name=='sinter_timing_helper' else {})
    monkeypatch.setattr(enabled,'require_capacity',lambda path:128*1024*1024)
    monkeypatch.setattr(enabled,'receipt_once',lambda *a:(_ for _ in ()).throw(RuntimeError('receipt failure')))
    with pytest.raises(ExceptionGroup) as group:enabled.run(SimpleNamespace(execute=True,output=tmp_path/'failed.json',commit='a'*40))
    assert [str(e) for e in group.value.exceptions]==['event failure','close failure','receipt failure']
