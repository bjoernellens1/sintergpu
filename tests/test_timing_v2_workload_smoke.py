# SPDX-License-Identifier: Apache-2.0
"""Actual NumPy CPU recipe smoke in experimental namespace; no GPU/timing."""
import pytest
from timing_v2_workload_gate import run,SmokeError


def test_real_depth_convert_recipe_in_snapshot():
    result=run(); child=result['child']
    assert result['status']=='provisional_cpu_smoke_not_measurement'
    assert result['source_before']==result['source_after']
    assert result['numpy_before']==result['numpy_after']
    assert child['recipe']=='vkl.depth_convert.depthConvert' and child['case_count']>=1
    assert {'depthOut','inverseOut','validOut','confidenceOut'}<=set(child['outputs'])
    assert child['scope'].startswith('One scalar')
    assert child['write_blocked'] and child['remount_blocked'] and child['torch_loaded'] is False
    assert child['loaded']['benchmarks.vkl_timings']['path']=='benchmarks/vkl_timings.py'
    assert child['loaded']['sintergpu.references.vkl.numpy']['path']=='src/sintergpu/references/vkl/numpy.py'
    assert child['numpy_native_modules_unverified'] and child['native_mappings_unverified']
    assert len(result['runtime_executable_unverified']['sha256'])==64
    assert result['bootstrap_unverified']['sha256_observed']==result['source_before']['tests/timing_v2_workload_namespace.py']

@pytest.mark.parametrize('scenario,reason',[('preimport','preimport_or_shadow'),('shadow','preimport_or_shadow'),('pyc','bytecode'),('late','late_import'),('unexpected','unexpected_import'),('numpy_source_tamper','numpy_identity'),('numpy_native_tamper','numpy_identity'),('source_drift','module_source_preload'),('source_side_effect','module_source_preload')])
def test_failures_are_specific_after_positive_preflight(scenario,reason):
    assert run()['child']['recipe']=='vkl.depth_convert.depthConvert'
    with pytest.raises(SmokeError) as exc: run(scenario)
    assert (exc.value.phase,exc.value.reason)==('child',reason)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):run('unknown')


def test_symlinked_source_ancestor_rejected_after_preflight(monkeypatch,tmp_path):
    import shutil
    import timing_v2_workload_gate as gate
    for name in set(gate.POLICIES[gate.FAMILY]['files'].values())|set(gate.HELPERS):
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(gate.ROOT/name,dest)
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    assert gate.run()['child']['recipe']=='vkl.depth_convert.depthConvert'
    parent=tmp_path/'src/sintergpu/references/vkl';dest=tmp_path/'moved';parent.rename(dest);parent.symlink_to(dest,target_is_directory=True)
    with pytest.raises(SmokeError) as exc:gate.run()
    assert (exc.value.phase,exc.value.reason)==('preflight','symlink')


def test_inaccessible_namespace_does_not_masquerade_as_child_failure(monkeypatch):
    import types
    import timing_v2_workload_gate as gate
    assert gate.run()['child']['recipe']=='vkl.depth_convert.depthConvert'
    fake=types.SimpleNamespace(returncode=1,stderr='unshare: permission denied',stdout='')
    monkeypatch.setattr(gate.subprocess,'run',lambda *args,**kwargs:fake)
    with pytest.raises(SmokeError) as exc:gate.run('late')
    assert (exc.value.phase,exc.value.reason)==('launch','unstructured_failure')
