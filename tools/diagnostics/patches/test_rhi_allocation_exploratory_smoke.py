# SPDX-License-Identifier: Apache-2.0
"""CPU-only failure and orchestration checks; never initialize Vulkan."""
import importlib.util
import json
import mmap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

path = Path(__file__).with_name('rhi_allocation_exploratory_smoke.py')
spec = importlib.util.spec_from_file_location('receipt_exploratory', path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def args(path):
    return NS(output=str(path), uuid=bytes(range(16)).hex(), driver=7, vendor=10, device=20, execute=True)


def test_setup_rejects_ambiguous_icd_before_import(monkeypatch):
    monkeypatch.delenv('VK_DRIVER_FILES', raising=False)
    with pytest.raises(ValueError, match='VK_DRIVER_FILES'): smoke.setup_runtime(NS())


def test_mapped_file_requires_real_matching_inode(tmp_path):
    p=tmp_path/'mapped'; p.write_bytes(b'a'*4096)
    with pytest.raises(ValueError, match='mapping'): smoke.mapped_file(p)
    with p.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ):
        assert smoke.mapped_file(p)['sha256'] == smoke.file_hash(p)
    with pytest.raises(ValueError, match='mapping'): smoke.mapped_file(p)


def fake_environment(monkeypatch, events, *, mismatch=False, case_fails=False):
    class Lease:
        def verify(self): events.append('verify')
    monkeypatch.setattr(smoke, 'BorrowedLease', Lease)
    @contextmanager
    def context():
        events.append('lock')
        try: yield Lease()
        finally: events.append('unlock')
    class Device:
        @staticmethod
        def enumerate_adapters(kind):
            events.append('enumerate')
            return [NS(vendor_id=10, device_id=20, luid='x')]
        def __init__(self, **kwargs): events.append('create')
        def close(self): events.append('close')
    spy=NS(Device=Device, DeviceType=NS(vulkan=1))
    def identity(*args):
        events.append('identity')
        if mismatch: raise ValueError('uuid mismatch')
    def cases(spy, device, **kwargs):
        events.append('cases')
        try:
            if case_fails: raise RuntimeError('case failure')
            return {'status':'diagnostic_only'}
        finally: device.close()
    def setup(args): events.append('setup'); return spy, NS(identity=identity, run_cases=cases), {'scope':'fake'}
    return context, setup


def test_success_lifecycle_and_exploratory_label(monkeypatch, tmp_path):
    events=[]; context, setup=fake_environment(monkeypatch, events)
    a=args(tmp_path/'receipt'); smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert events.index('lock') < events.index('setup') < events.index('create') < events.index('identity') < events.index('cases') < events.index('close') < events.index('unlock')
    assert events[-2:] == ['verify', 'unlock']
    assert json.loads(Path(a.output).read_text())['status']=='exploratory_only'


def test_wrong_uuid_closes_before_allocation(monkeypatch, tmp_path):
    events=[]; context, setup=fake_environment(monkeypatch, events, mismatch=True)
    a=args(tmp_path/'receipt')
    with pytest.raises(ValueError, match='uuid'): smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert 'cases' not in events and events.index('close') < events.index('unlock')
    assert json.loads(Path(a.output).read_text())['status']=='failure'


def test_case_failure_still_closes_and_records(monkeypatch, tmp_path):
    events=[]; context, setup=fake_environment(monkeypatch, events, case_fails=True)
    a=args(tmp_path/'receipt')
    with pytest.raises(RuntimeError, match='case failure'): smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert events.count('close') == 1 and events.index('close') < events.index('unlock')
    assert json.loads(Path(a.output).read_text())['status']=='failure'


def test_existing_output_and_bad_uuid_do_not_lock(monkeypatch, tmp_path):
    events=[]; context, setup=fake_environment(monkeypatch, events)
    p=tmp_path/'receipt'; p.write_text('keep')
    with pytest.raises(FileExistsError): smoke.execute(args(p), setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert p.read_text() == 'keep' and not events
    a=args(tmp_path/'other'); a.uuid=bytes(16).hex()
    with pytest.raises(ValueError): smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert not events


def test_final_lease_failure_never_records_success(monkeypatch, tmp_path):
    events=[]
    class Lease:
        calls=0
        def verify(self):
            self.calls += 1
            if self.calls == 2: raise RuntimeError('lease lost')
    monkeypatch.setattr(smoke, 'BorrowedLease', Lease)
    @contextmanager
    def context():
        events.append('lock')
        try: yield Lease()
        finally: events.append('unlock')
    class Device:
        @staticmethod
        def enumerate_adapters(kind): return [NS(vendor_id=10, device_id=20, luid='x')]
        def __init__(self, **kwargs): pass
        def close(self): events.append('close')
    def cases(spy, device, **kwargs): device.close(); return {}
    spy=NS(Device=Device, DeviceType=NS(vulkan=1))
    def setup(args): return spy, NS(identity=lambda *x: None, run_cases=cases), {}
    a=args(tmp_path/'receipt')
    with pytest.raises(RuntimeError, match='lease lost'): smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert events == ['lock', 'close', 'unlock']
    assert json.loads(Path(a.output).read_text())['status']=='failure'


def test_wrong_adapter_never_creates_device(monkeypatch, tmp_path):
    events=[]; context, setup=fake_environment(monkeypatch, events)
    a=args(tmp_path/'receipt'); a.vendor=999
    with pytest.raises(ValueError, match='adapter'): smoke.execute(a, setup=setup, lease_context=context, verify_runtime=lambda *x: None)
    assert 'create' not in events and json.loads(Path(a.output).read_text())['status']=='failure'


def test_verified_case_bytes_ignore_stale_pyc_and_reject_mutation(tmp_path):
    import hashlib
    import os
    import py_compile
    source = tmp_path/'cases.py'
    marker = tmp_path/'executed'
    malicious = f"open({str(marker)!r}, 'w').write('bad')\nRESULT = 'bad'\n"
    good = "RESULT = 'good'\n".ljust(len(malicious), ' ')
    source.write_text(malicious)
    stamp = source.stat().st_mtime_ns
    py_compile.compile(str(source), doraise=True)
    source.write_text(good)
    os.utime(source, ns=(stamp, stamp))
    expected = hashlib.sha256(good.encode()).hexdigest()
    module = smoke.load_verified_cases(source, expected)
    assert module.RESULT == 'good' and not marker.exists()
    source.write_text(malicious)
    with pytest.raises(ValueError, match='differs'):
        smoke.load_verified_cases(source, expected)
    assert not marker.exists()


def test_production_path_blocked_before_import_or_lock(monkeypatch, tmp_path):
    events=[]
    def bad_import(*args, **kwargs):
        events.append('import')
        raise AssertionError('runtime import must not happen')
    monkeypatch.setattr(smoke.importlib, 'import_module', bad_import)
    @contextmanager
    def bad_context():
        events.append('lock')
        raise AssertionError('lock must not be acquired')
        yield
    with pytest.raises(RuntimeError, match='execution blocked'):
        smoke.execute(args(tmp_path/'receipt'), lease_context=bad_context)
    assert events == [] and not (tmp_path/'receipt').exists()
    with pytest.raises(RuntimeError, match='execution blocked'):
        smoke.execute(args(tmp_path/'receipt'), setup=smoke.setup_runtime, lease_context=bad_context)
    assert events == []
