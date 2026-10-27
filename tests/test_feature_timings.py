"""Per-entry feature timing must preserve chunk state and independent semantics."""
import numpy as np
from benchmarks import vkl_features as timing
from sintergpu.references import features_numeric as reference


def test_stage_cpu_oracles_compose_to_independent_match_and_cover_tail_chunks():
    assert hasattr(timing,'cases'), 'feature stage timing cases missing'
    cases=list(timing.cases())
    assert len(cases)==10
    assert len({c['stage']['operation'] for c in cases})==7
    assert sum(c['stage']['params'].get('chunkCount')==1 for c in cases)==3
    for case in cases:
        actual=timing.stage_reference(case['stage'],case['arrays'])
        assert set(actual)==set(case['outputs'])
        for name,value in actual.items():
            assert value.shape==case['arrays'][name].shape
            assert value.dtype==case['arrays'][name].dtype
            np.testing.assert_array_equal(value,case['expected'][name])
    final=next(c for c in cases if c['stage']['entrypoint']=='finalizeMatch')
    p=final['stage']['params'];a=final['original']
    expected=reference.binary_local_match(a['landmarkDescriptor'].reshape(p['batches'],p['landmarks'],p['bits']).astype(bool),a['candidateDescriptor'].reshape(p['batches'],p['candidates'],p['bits']).astype(bool),a['projected'].reshape(p['batches'],p['landmarks'],2),a['candidateXY'].reshape(p['batches'],p['candidates'],2),radius_px=5.,max_hamming_distance=p['maxDistance'],mutual=True,landmark_valid=a['landmarkValid'].reshape(p['batches'],p['landmarks']).astype(bool),candidate_valid=a['candidateValid'].reshape(p['batches'],p['candidates']).astype(bool),candidate_chunk_size=129)
    for out,key in [('matchedIndexOut','matched_index'),('distanceOut','distance'),('matchValidOut','valid')]:
        np.testing.assert_array_equal(final['expected'][out][:p['batches']*p['landmarks']],expected[key].reshape(-1))
    assert expected['valid'].any(), 'timing fixture must perform successful matching'


def test_feature_case_identity_binds_seeded_outputs_and_inputs():
    assert hasattr(timing,'workload_identity'), 'feature workload hashing missing'
    case=next(timing.cases());first=timing.workload_identity(case)
    assert first==timing.workload_identity(next(timing.cases()))
    name=case['outputs'][0];case['arrays'][name][0]^=np.uint32(1)
    assert first['sha256']!=timing.workload_identity(case)['sha256']


def test_feature_workload_identity_includes_dispatch_geometry_and_binding_order():
    case=next(timing.cases());first=timing.workload_identity(case)
    original=case['stage']['thread_count'];case['stage']['thread_count']=(original[0]+256,1,1)
    assert first['sha256']!=timing.workload_identity(case)['sha256']
    case['stage']['thread_count']=original;case['stage']['bindings']=tuple(reversed(case['stage']['bindings']))
    assert first['sha256']!=timing.workload_identity(case)['sha256']
