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
