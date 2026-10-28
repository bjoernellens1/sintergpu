"""Per-stage baselines must exercise real primitive contracts and retain provenance."""
import importlib
import importlib.util
import copy
import numpy as np
import pytest


def recipe():
    assert importlib.util.find_spec('benchmarks.primitive_timings'), 'primitive timing recipe missing'
    return importlib.import_module('benchmarks.primitive_timings')


def test_each_existing_primitive_entry_has_nonempty_representative_workloads():
    m=recipe();cases=list(m.cases())
    assert {c['operation'] for c in cases}=={
        'primitives.'+family+'.'+entry
        for family,entries in [('scan',('scanBlocks','scanCarry','scanExclusive')),('scan_uint',('scanBlocks','scanCarry','scanExclusive')),('compact',('compactPrepare','compactScatter'))]
        for entry in entries}
    assert {c['params']['count'] for c in cases if c['kind']=='scanBlocks'}>={255,256,257,70001}
    empty=[c for c in cases if c['params']['count']==0]
    assert len(empty)==1 and empty[0]['kind']=='compactPrepare' and empty[0]['thread_count']==[256,1,1]
    for c in cases:
        expected=m.reference_call(c);m.check_parity(c,expected,expected)
        assert set(expected)==set(c['outputs'])
        for name in c['outputs']:
            assert c['arrays'][name].size>c['live_counts'][name]


def test_block_prefix_uses_independent_serial_oracle_and_retains_padding():
    m=recipe();c=m.make_scan_case('scanBlocks',np.uint32,4)
    c['arrays']['values'][:4]=[0xffffffff,2,3,4];c['arrays']['segmentIds'][:4]=[0,0,2,2]
    result=m.reference_call(c)
    np.testing.assert_array_equal(result['output'][:4],[0xffffffff,1,3,7])
    assert result['blockValues'][0]==7 and result['blockIds'][0]==2
    np.testing.assert_array_equal(result['output'][4:],c['arrays']['output'][4:])
    c=m.make_scan_case('scanBlocks',np.float32,257,product=True)
    c['arrays']['values'][:257]=1;c['arrays']['values'][127]=0;c['arrays']['segmentIds'][:257]=0
    r=m.reference_call(c)
    assert r['output'][126]==1 and r['output'][255]==0 and r['output'][256]==1


def test_carry_only_modifies_matching_segment_and_exclusive_uses_identity():
    m=recipe();c=m.make_scan_case('scanCarry',np.uint32,259)
    c['arrays']['segmentIds'][:259]=0;c['arrays']['segmentIds'][258]=3
    c['arrays']['blockIds'][0]=0;c['arrays']['scannedBlocks'][0]=11;c['arrays']['output'][:259]=2
    r=m.reference_call(c)['output'];np.testing.assert_array_equal(r[:256],np.full(256,2))
    np.testing.assert_array_equal(r[256:259],[13,13,2])
    c=m.make_scan_case('scanExclusive',np.float32,4,product=True)
    c['arrays']['segmentIds'][:4]=[0,0,2,2];c['arrays']['inclusiveValues'][:4]=[3,6,4,20]
    np.testing.assert_array_equal(m.reference_call(c)['output'][:4],[1,3,1,4])


def test_compaction_reference_preserves_duplicate_payload_order_and_empty_count():
    m=recipe();c=m.make_compact_case('compactScatter',5,'sparse')
    c['arrays']['predicate'][:5]=[7,0,3,0,1];c['arrays']['prefix'][:5]=[1,1,2,2,3];c['arrays']['payload'][:5]=[9,2,9,3,4]
    r=m.reference_call(c)
    np.testing.assert_array_equal(r['compacted'][:3],[9,9,4]);assert r['selectedCount'][0]==3
    c=m.make_compact_case('compactPrepare',0,'none');assert m.reference_call(c)['selectedCount'][0]==0


def test_workload_identity_covers_uniforms_launch_bindings_inputs_and_output_seeds():
    m=recipe();original=m.make_scan_case('scanCarry',np.float32,257);identity=m.workload_identity(original)['sha256']
    for mutate in [lambda c:c['params'].__setitem__('product',1),lambda c:c['thread_count'].__setitem__(0,768),lambda c:c['arrays']['output'].__setitem__(-1,73),lambda c:c['arrays']['scannedBlocks'].__setitem__(0,9),lambda c:c.__setitem__('arrays',dict(reversed(list(c['arrays'].items()))))]:
        c=copy.deepcopy(original);mutate(c);assert m.workload_identity(c)['sha256']!=identity


def test_parity_checks_partial_block_summaries_and_untouched_output_tails():
    m=recipe();c=m.make_scan_case('scanBlocks',np.float32,257);e=m.reference_call(c)
    for name,index in [('output',-1),('blockValues',-1),('blockIds',1)]:
        a={k:v.copy() for k,v in e.items()};a[name][index]=73
        with pytest.raises(AssertionError):m.check_parity(c,a,e)


def test_cpu_only_collection_has_actual_samples_and_source_bound_identities(monkeypatch):
    m=recipe();c=m.make_scan_case('scanBlocks',np.float32,257)
    monkeypatch.setattr(m,'cases',lambda:iter([c]));report=m.run(iterations=2,warmups=1)
    row,=report['measurements'];assert len(row['metrics']['cpu_reference_wall_ms'])==2
    assert row['operation']==c['operation'] and row['workload']['sha256']==m.workload_identity(c)['sha256']
    for key in ('source_sha256','reference_sha256','runner_sha256','runner_helper_sha256','shader_tree_sha256','reference_tree_sha256'):
        assert len(row[key])==64
    assert row['environment']['gpu'] is None
    for kwargs in ({'iterations':0},{'warmups':0},{'iterations':True},{'device_index':True}):
        with pytest.raises(ValueError):m.run(**kwargs)


def test_cpu_stage_composition_matches_global_serial_reference():
    from sintergpu.references.scan import segmented_scan
    m=recipe()
    for dtype in (np.float32,np.uint32):
        for product in (False,True):
            block=m.make_scan_case('scanBlocks',dtype,1025,product)
            b=m.reference_call(block);n=1025;blocks=(n+255)//256
            carry=m.make_scan_case('scanCarry',dtype,n,product)
            carry['arrays']['segmentIds'][:n]=block['arrays']['segmentIds'][:n]
            carry['arrays']['output'][:n]=b['output'][:n]
            carry['arrays']['blockIds'][:blocks]=b['blockIds'][:blocks]
            with np.errstate(over='ignore'):
                carry['arrays']['scannedBlocks'][:blocks]=segmented_scan(b['blockValues'][:blocks],b['blockIds'][:blocks],product=product)
                expected=segmented_scan(block['arrays']['values'][:n],block['arrays']['segmentIds'][:n],product=product)
            actual=m.reference_call(carry)['output'][:n]
            if dtype==np.uint32:np.testing.assert_array_equal(actual,expected)
            else:np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)


def test_measured_irregular_fixtures_exercise_matching_and_rejected_later_carry():
    m=recipe();fixtures=[c for c in m.cases() if c['params']['count']==1025]
    assert len(fixtures)==12
    for c in fixtures:
        ids=c['arrays']['segmentIds'][:1025]
        assert np.any(np.diff(np.unique(ids))>1)
        if c['kind']!='scanCarry':continue
        index=np.arange(1025);later=index>=256;block=index//256
        matching=ids[later]==c['arrays']['blockIds'][block[later]-1]
        assert np.any(matching) and np.any(~matching)
        assert ids[767]==ids[768] and ids[999]!=ids[1000]
        expected=m.reference_call(c)
        rejected=index[later][~matching]
        np.testing.assert_array_equal(expected['output'][rejected],c['arrays']['output'][rejected])
        corrupted={k:v.copy() for k,v in expected.items()}
        corrupted['output'][rejected[0]]+=1
        with pytest.raises(AssertionError):m.check_parity(c,corrupted,expected)


def test_fixture_selection_is_explicit_and_recorded(monkeypatch):
    m=recipe();legacy=m.make_scan_case('scanBlocks',np.uint32,257)
    irregular=m.make_scan_case('scanBlocks',np.uint32,1025)
    monkeypatch.setattr(m,'cases',lambda:iter([legacy,irregular]))
    for selection,expected in [('all',2),('legacy',1),('irregular',1)]:
        report=m.run(iterations=1,warmups=1,fixture_set=selection)
        assert len(report['measurements'])==expected
        assert report['environment']['fixture_set']==selection
        assert all(r['environment']['fixture_set']==selection for r in report['measurements'])
    with pytest.raises(ValueError):m.run(fixture_set='typo')
