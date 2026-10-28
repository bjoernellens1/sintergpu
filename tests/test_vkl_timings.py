"""VKL baselines must cover existing entries without inventing pipeline times."""
import importlib
import numpy as np
import pytest


def recipe():
    try:return importlib.import_module('benchmarks.vkl_timings')
    except ImportError:pytest.fail('remaining VKL per-entry timing recipe missing')


def test_cases_cover_every_existing_nonfeature_vkl_entry_and_full_output_tails():
    from sintergpu.catalog import list_operations
    m=recipe();cases=list(m.cases())
    expected={o['id'] for o in list_operations() if o['id'].startswith('vkl.') and not o['id'].startswith(('vkl.brief.','vkl.binary_match.'))}
    # The established per-entry CPU recipe retains its 13-operation contract;
    # new staged reprojection measurements live in their own complete-graph recipe.
    reprojection=importlib.import_module('benchmarks.vkl_reprojection')
    graphs=list(reprojection.cases())
    new_entries={stage['operation'] for case in graphs for stage in case['graph']['stages']
                 if stage['operation'].startswith('vkl.reprojection.')}
    established=expected-{op for op in expected if op.startswith('vkl.reprojection.')}
    assert len(established)==13
    assert {c['operation'] for c in cases}==established
    assert {c['operation'] for c in cases}|new_entries==expected
    assert {case['family'] for case in graphs}=={'projection','raster','temporal','reference'}
    assert len(cases)==38
    for c in cases:
        actual=m.reference_call(c)
        assert actual and set(actual)==set(c['outputs'])
        assert c['thread_count'][0]>256 and c['thread_count'][1:]==(1,1)
        for name,a in actual.items():
            assert a.dtype==c['arrays'][name].dtype and a.shape==c['arrays'][name].shape
            np.testing.assert_array_equal(a[c['live_counts'][name]:],c['arrays'][name][c['live_counts'][name]:])
        m.check_parity(c,actual,actual)
        name=c['outputs'][0];bad={k:v.copy() for k,v in actual.items()};bad[name][-1]+=1
        with pytest.raises(AssertionError):m.check_parity(c,bad,actual)


def test_vkl_workload_identity_binds_stage_geometry_masks_and_output_seeds():
    m=recipe();c=next(m.cases());original=m.workload_identity(c)
    assert original==m.workload_identity(next(m.cases()))
    c['thread_count']=(c['thread_count'][0]+256,1,1)
    assert m.workload_identity(c)['sha256']!=original['sha256']
    c=next(m.cases());c['arrays'][c['outputs'][0]][-1]+=1
    assert m.workload_identity(c)['sha256']!=original['sha256']


def test_initialize_and_direct_label_oracles_preserve_distinct_invalid_contracts():
    m=recipe();cases=list(m.cases())
    init=next(c for c in cases if c['operation'].endswith('.initializeDepth'))
    label=next(c for c in cases if c['operation'].endswith('.labelDirectDepth'))
    init['arrays']['depth'][0]=np.nan;init['arrays']['valid'][0]=7
    label['arrays']['valid'][0]=7
    a=m.reference_call(init);b=m.reference_call(label)
    assert a['validOut'][0]==0 and a['depthOut'][0]==0
    assert b['direct_validOut'][0]==1 and b['provenanceOut'][0]==1 and b['propagated_validOut'][0]==0


@pytest.mark.parametrize('iterations,warmups',[(0,1),(1,0),(True,1),(1,True)])
def test_vkl_timing_collector_rejects_invalid_counts_before_device_access(iterations,warmups):
    with pytest.raises(ValueError):recipe().run(gpu=True,iterations=iterations,warmups=warmups)


def test_cpu_only_collection_retains_source_identities_without_importing_gpu(monkeypatch):
    m=recipe();one=next(m.cases());monkeypatch.setattr(m,'cases',lambda:iter([one]))
    report=m.run(gpu=False,iterations=2,warmups=1)
    assert len(report['measurements'])==1
    row=report['measurements'][0]
    assert set(row['metrics'])=={'cpu_reference_wall_ms'}
    assert len(row['metrics']['cpu_reference_wall_ms'])==2
    assert len(row['setup_source_sha256'])==64 and len(row['reference_sha256'])==64
    assert row['environment']['gpu'] is None
    from sintergpu.timings import build_registry
    from sintergpu.catalog import list_operations
    assert build_registry(list_operations(),report['measurements'])['coverage']['cpu_measured']==1


def test_temporal_best_workload_shape_describes_rows_not_an_unrelated_image():
    for c in recipe().cases():
        if c['kind']=='best':assert c['shape']==(c['params']['batches'],1,1,1)
