# SPDX-License-Identifier: Apache-2.0
"""CPU-only mocked orchestration; no Vulkan or GPU initialization."""
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

path = Path(__file__).with_name('rhi_allocation_vulkan_runner.py')
spec = importlib.util.spec_from_file_location('receipt_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
UUID = bytes(range(16))


def harness(events):
    @contextmanager
    def context():
        events.append('lock')
        class Lease:
            def verify(self): events.append('verify')
        try: yield Lease()
        finally: events.append('unlock')
    return context


def test_production_provider_gate_fails_before_creation(tmp_path):
    events=[]; path=tmp_path/'receipt.json'
    with pytest.raises(runner.IdentityUnavailable, match='provider'):
        runner.run(uuid=UUID, driver=7, output=path, lease_context=harness(events),
                   create_device=lambda: events.append('device'), run_cases=lambda *a, **kw: None)
    assert 'device' not in events and events == ['lock', 'verify', 'verify', 'unlock']
    assert json.loads(path.read_text())['status'] == 'failure'


def test_device_close_under_lock_on_success_and_case_failure(tmp_path):
    for failed in (False, True):
        events=[]
        class Device:
            def close(self): events.append('close')
        def create(): events.append('device'); return Device()
        def cases(device, **kwargs):
            events.append('cases')
            try:
                if failed: raise RuntimeError('case failed')
                return {'status':'diagnostic_only'}
            finally: device.close()
        args=dict(uuid=UUID, driver=7, output=tmp_path/str(failed), preflight=lambda: events.append('preflight'),
                  create_device=create, run_cases=cases, lease_context=harness(events))
        if failed:
            with pytest.raises(RuntimeError, match='case failed'): runner.run(**args)
        else:
            runner.run(**args)
        assert events.index('lock') < events.index('device') < events.index('close') < events.index('unlock')
        assert json.loads(Path(args['output']).read_text())['status'] == ('failure' if failed else 'diagnostic_only')


def test_write_once_and_failure_preserves_original(tmp_path):
    path=tmp_path/'receipt'; path.write_text('original')
    with pytest.raises(FileExistsError): runner.run(uuid=UUID, driver=7, output=path, lease_context=harness([]))
    assert path.read_text() == 'original'


def test_invalid_inputs_never_acquire_lock(tmp_path):
    events=[]
    with pytest.raises(ValueError): runner.run(uuid=bytes(16), driver=7, output=tmp_path/'x', lease_context=harness(events))
    assert events == []


def test_close_failure_writes_failure_not_success(tmp_path):
    events=[]
    class Device:
        def close(self): events.append('close'); raise RuntimeError('close failed')
    def cases(device, **kwargs): device.close()
    path=tmp_path/'receipt'
    with pytest.raises(RuntimeError, match='close failed'):
        runner.run(uuid=UUID, driver=7, output=path, preflight=lambda: None,
                   create_device=Device, run_cases=cases, lease_context=harness(events))
    assert events.index('close') < events.index('unlock')
    assert json.loads(path.read_text())['status']=='failure'


def test_final_lease_failure_writes_failure_not_success(tmp_path):
    events=[]
    @contextmanager
    def context():
        class Lease:
            count=0
            def verify(self):
                self.count += 1
                if self.count == 2: raise RuntimeError('lost lease')
        try: yield Lease()
        finally: events.append('unlock')
    class Device:
        def close(self): events.append('close')
    def cases(device, **kwargs): device.close(); return {}
    path=tmp_path/'receipt'
    with pytest.raises(RuntimeError, match='lost lease'):
        runner.run(uuid=UUID, driver=7, output=path, preflight=lambda: None,
                   create_device=Device, run_cases=cases, lease_context=context)
    assert events == ['close','unlock']
    assert json.loads(path.read_text())['status']=='failure'
