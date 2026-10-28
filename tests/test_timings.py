"""A timing reference must cover the catalog without inventing missing results."""
import importlib
import math
import pytest


def timing_module():
    try:return importlib.import_module('sintergpu.timings')
    except ImportError:pytest.fail('CPU/GPU timing registry missing')


def catalog():
    return [{'id':'test.kernel','variant':'portable','sha256':'a'*64},{'id':'test.other','variant':'portable','sha256':'b'*64}]


def measurement():
    return {**timing_module().dependency_identities(),'operation':'test.kernel','variant':'portable','source_sha256':'a'*64,'reference_sha256':'c'*64,'workload':{'id':'fixture-16','sha256':'d'*64,'description':'16 scalar elements'},'environment':{'cpu':'test CPU','gpu':'test GPU','compiler_sha256':'e'*64},'metrics':{'cpu_reference_wall_ms':[1.,2.,3.],'gpu_kernel_ms':[.1,.2,.3]},'iterations':3,'warmups':2,'scope':'Independent CPU reference and GPU fixture parity, no performance qualification'}


def test_registry_covers_every_catalog_entry_and_missing_times_remain_pending():
    m=timing_module();db=m.build_registry(catalog(),[measurement()]);assert len(db['operations'])==2
    table=m.render_markdown(db)
    assert 'test.kernel' in table and 'test.other' in table and 'pending' in table
    assert '2.000000' in table and '0.200000' in table
    assert db['coverage']['cpu_measured']==1 and db['coverage']['gpu_measured']==1
    assert db['coverage']['both_measured']==1 and db['coverage']['pending']==1


@pytest.mark.parametrize('samples',[[],[0.],[-1.],[math.nan],[math.inf]])
def test_registry_rejects_invalid_or_zero_raw_durations(samples):
    item=measurement();item['metrics']['cpu_reference_wall_ms']=samples
    with pytest.raises(ValueError):timing_module().build_registry(catalog(),[item])


def test_registry_never_reuses_old_source_timings_as_current():
    item=measurement();item['source_sha256']='f'*64
    db=timing_module().build_registry(catalog(),[item]);table=timing_module().render_markdown(db)
    assert db['coverage']['both_measured']==0 and 'stale' in table
    assert db['measurements'][0]['metrics']==item['metrics'] # Historical evidence retained.


def test_cpu_and_gpu_workload_records_are_not_combined_or_pipeline_times_relabelled():
    a=measurement();b=measurement();a['metrics']={'cpu_reference_wall_ms':[1.,2.,3.]};b['metrics']={'gpu_pipeline_ms':[.1,.2,.3]};b['workload']['id']='different-workload';b['workload']['sha256']='f'*64
    db=timing_module().build_registry(catalog(),[a,b]);assert db['coverage']['both_measured']==0
    table=timing_module().render_markdown(db)
    assert 'different-workload' in table and '0.200000' in table and 'GPU kernel' in table


def test_timings_are_bound_to_complete_source_and_environment_identities():
    for field in ('source_sha256','reference_sha256','environment','scope','workload'):
        item=measurement();item.pop(field)
        with pytest.raises(ValueError):timing_module().build_registry(catalog(),[item])


def test_cpu_samples_exclude_warmup_and_keep_independent_invocations(monkeypatch):
    try:module=importlib.import_module('benchmarks.locus_timings')
    except ImportError:pytest.fail('reusable Locus CPU/GPU timing recipe missing')
    calls=[];ticks=iter([0,1000000,2000000,5000000,6000000,8000000])
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:next(ticks))
    samples=module.sample_cpu(lambda:calls.append(1),iterations=3,warmups=2)
    assert samples==[1.,3.,2.] and len(calls)==5


@pytest.mark.parametrize('field',['shader_tree_sha256','reference_tree_sha256'])
def test_dependency_only_changes_make_timings_stale(field):
    item=measurement();item[field]='f'*64
    db=timing_module().build_registry(catalog(),[item])
    assert db['coverage']['both_measured']==0 and not db['measurements'][0]['current_source']


def test_cli_exposes_packaged_timing_reference_without_gpu(capsys):
    from sintergpu.cli import main
    import json
    assert main(['timings','--json'])==0
    db=json.loads(capsys.readouterr().out)
    from sintergpu.catalog import list_operations
    assert {o['operation'] for o in db['operations']}=={o['id'] for o in list_operations()}
    assert main(['timings','depth.bilateral','--json'])==0
    selected=json.loads(capsys.readouterr().out)
    assert [o['operation'] for o in selected['operations']]==['depth.bilateral']
    assert all(r['operation']=='depth.bilateral' for r in selected['measurements'])


@pytest.mark.parametrize('metric',['gpu_kernel_ms','gpu_pipeline_ms','host_submit_complete_ms','transfer_inclusive_ms'])
def test_all_gpu_path_metrics_require_gpu_and_compiler_identity(metric):
    item=measurement();item['metrics']={metric:[1.,2.,3.]};item['environment']={'cpu':'CPU'}
    with pytest.raises(ValueError,match='GPU'):timing_module().build_registry(catalog(),[item])


@pytest.mark.parametrize('compiler',[None,123])
def test_invalid_compiler_identity_is_a_validation_error(compiler):
    item=measurement();item['environment']['compiler_sha256']=compiler
    with pytest.raises(ValueError):timing_module().build_registry(catalog(),[item])


def test_cpu_only_samples_do_not_require_a_gpu_runtime_identity():
    item=measurement();item['metrics']={'cpu_reference_wall_ms':[1.,2.,3.]};item['environment']={'cpu':'CPU'}
    assert timing_module().build_registry(catalog(),[item])['coverage']['cpu_measured']==1


@pytest.mark.parametrize('diagnostic',['missing','timeout'])
def test_missing_optional_driver_diagnostic_preserves_collection_and_closes_device(tmp_path,monkeypatch,diagnostic):
    import subprocess,sys,types
    from benchmarks import locus_timings as module
    import sintergpu.fixtures.locus.cases as fixtures
    monkeypatch.setenv('SINTERGPU_GPU_TESTS','1')
    class Device:
        closed=False
        @staticmethod
        def enumerate_adapters(kind):return [types.SimpleNamespace(vendor_id=0x1002,device_id=0x1586,luid=None)]
        def __init__(self,**kwargs):self.info=types.SimpleNamespace(adapter_name='Test GPU',api_name='Vulkan',timestamp_frequency=1000)
        def close(self):Device.closed=True
    spy=types.ModuleType('slangpy');spy.__file__=str(tmp_path/'__init__.py');spy.Device=Device;spy.DeviceType=types.SimpleNamespace(vulkan='vulkan');spy.SlangFloatingPointMode=types.SimpleNamespace(precise='precise')
    (tmp_path/'libslang-compiler.so').write_bytes(b'test compiler')
    monkeypatch.setitem(sys.modules,'slangpy',spy)
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:'test-version')
    monkeypatch.setattr(fixtures,'cases',lambda:[])
    def fail(*args,**kwargs):
        if diagnostic=='missing':raise FileNotFoundError('vulkaninfo')
        raise subprocess.TimeoutExpired(['vulkaninfo'],30)
    monkeypatch.setattr(subprocess,'run',fail)
    report=module.run(gpu=True,iterations=1,warmups=1)
    assert Device.closed and report['measurements']==[]
    assert 'unavailable' in report['environment']['vulkaninfo_summary']


def test_current_release_qualification_counts_agree_with_timing_database():
    import json,re
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    db=json.loads((root/'src/sintergpu/data/timings.json').read_text())
    qualification=json.loads((root/'qualification'/f"{db['library_version']}.json").read_text())
    reference=qualification['timing_reference']
    assert reference['catalog_entries']==db['coverage']['catalog_entries']
    assert reference['paired_cpu_gpu_entries']==db['coverage']['both_measured']
    assert reference['pending_portable_entries']==db['coverage']['pending']
    assert reference['additional_native_pending']==len(db['additional_pending_targets'])
    for gate in qualification['pending_gates']:
        count=re.search(r'CPU/GPU baseline timing coverage for (\d+) remaining portable',gate)
        if count:assert int(count[1])==db['coverage']['pending']


def test_timing_registry_serialization_is_compact_and_semantics_preserving():
    import json
    from tools.update_timings import serialize_registry
    registry = {'measurements': [{'metrics': {'cpu_reference_wall_ms': {'samples': [1.0, 2.0, 3.0]}}, 'environment': {'gpu': 'example'}}]}
    encoded = serialize_registry(registry)
    assert json.loads(encoded) == registry
    assert len(encoded) < len(json.dumps(registry, indent=2))
    assert encoded == serialize_registry(registry)
