# SPDX-License-Identifier: Apache-2.0
"""Real Linux user/mount namespace CPU experiment; no GPU or timing promotion."""
import pytest
from timing_v2_snapshot_gate import run,SnapshotGateError


def test_fresh_isolated_read_only_snapshot_and_unprivileged_child():
    result=run()
    assert result['before']==result['after']
    assert result['child']['allowed_module_executed'] is True
    assert result['child']['allowed_module_sha256']==result['before']['tests/timing_v2_snapshot_allow.py']
    for key in ('isolated','write_blocked','remount_blocked','nested_overmount_blocked','capabilities_dropped'): assert result['child'][key] is True

@pytest.mark.parametrize('scenario,reason',[('late','late_import'),('preimport','preimport_or_shadow'),('pyc','bytecode'),('unexpected','unexpected_import'),('shadow','preimport_or_shadow'),('cap_prm','privilege_drop'),('cap_inh','privilege_drop'),('cap_amb','privilege_drop'),('cap_missing','privilege_drop')])
def test_late_stale_and_bytecode_fail_closed(scenario,reason):
    assert run()['child']['allowed_module_executed'] is True
    with pytest.raises(SnapshotGateError) as exc: run(scenario)
    assert (exc.value.phase,exc.value.reason)==('child',reason)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError): run('unknown')


def test_unrelated_file_does_not_change_snapshot_closure(monkeypatch,tmp_path):
    import shutil
    import timing_v2_snapshot_gate as gate
    paths=set(gate.POLICIES[gate.FAMILY]['files'].values())|set(gate.HELPERS)
    for name in paths:
        dest=tmp_path/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(gate.ROOT/name,dest)
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    first=gate.run()
    (tmp_path/'unrelated').write_text('not in policy')
    second=gate.run()
    assert first['before']==second['before']


def test_symlinked_source_ancestor_rejected(monkeypatch,tmp_path):
    import shutil
    import timing_v2_snapshot_gate as gate
    paths=set(gate.POLICIES[gate.FAMILY]['files'].values())|set(gate.HELPERS)
    for name in paths:
        dest=tmp_path/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(gate.ROOT/name,dest)
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    assert gate.run()['child']['allowed_module_executed'] is True
    directory=tmp_path/'src/sintergpu/references/vkl'; target=tmp_path/'moved-vkl'; directory.rename(target); directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(SnapshotGateError) as exc: gate.run()
    assert (exc.value.phase,exc.value.reason)==('preflight','symlink_source')


def test_setup_failure_cannot_masquerade_as_late_import(monkeypatch):
    import types
    import timing_v2_snapshot_gate as gate
    fake=types.SimpleNamespace(returncode=1,stderr='{"phase":"setup","reason":"setup_failure"}',stdout='')
    monkeypatch.setattr(gate.subprocess,'run',lambda *args,**kwargs:fake)
    with pytest.raises(SnapshotGateError) as exc: gate.run('late')
    assert (exc.value.phase,exc.value.reason)==('setup','setup_failure')
    assert exc.value.reason!='late_import'
