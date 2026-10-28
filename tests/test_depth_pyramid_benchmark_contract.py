# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import numpy as np
import pytest
from benchmarks.depth_pyramid_consolidation import (ROOT, DONORS, WORKLOADS, BOUNDARIES, collect_run, evaluate, source_manifest, make_inputs, cpu_reference, VKL_DONOR_EPSILON)


def identities(profile='vkl'):
    paths = {'candidate': DONORS[profile + '_incumbent'], 'incumbent': DONORS[profile + '_incumbent'],
             'donor': DONORS[profile + '_donor'], 'reference': 'src/sintergpu/references/depth_pyramid_profiles.py',
             'harness': 'benchmarks/depth_pyramid_consolidation.py'}
    return {'sources': {k: {'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for k, p in paths.items()},
            'environment': {'compiler_sha256': 'a'*64, 'device': 'synthetic device', 'device_uuid': 'synthetic uuid', 'driver': 'synthetic driver', 'environment': 'synthetic test'}}

CONTROLS = {'exclusive': True, 'interference_free': True, 'receipt_sha256': 'b'*64}

class Fake:
    def __init__(self, profile='vkl', delta=0, levels=3, threshold=None, extra=None):
        if threshold is None: threshold = VKL_DONOR_EPSILON if profile == 'vkl' else .01
        self.profile, self.delta, self.levels, self.threshold, self.extra = profile, delta, levels, threshold, extra or {}
        self.events = []
    def reset(self, variant, inputs): self.events.append(('reset', variant))
    def output(self, inputs):
        result = cpu_reference(inputs, profile=self.profile, levels=self.levels, threshold=self.threshold)
        result.update(copy.deepcopy(self.extra)); return result
    def execute(self, variant, inputs, boundary):
        self.events.append(('execute', variant, boundary)); return self.output(inputs)
    def measure(self, variant, inputs, boundary):
        self.events.append(('measure', variant, boundary))
        return 1.0 + (self.delta if variant == 'candidate' else 0), self.output(inputs)


def fixture():
    return {'depth': np.ones((1,8,8), np.float32), 'valid': np.ones((1,8,8), np.uint32), 'confidence': np.ones((1,8,8), np.float32)}


def run(f, inputs=None, **kwargs):
    args = dict(profile=f.profile, boundary='isolated_gpu', samples=30, warmups=0, levels=f.levels, threshold=f.threshold, identities=identities(f.profile), expected_extra=f.extra, controls=CONTROLS)
    args.update(kwargs)
    return collect_run(f, fixture() if inputs is None else inputs, **args)


def test_preregistered_workloads_and_boundaries():
    assert any(w['profile'] == 'vkl' and w['batch'] > 1 for w in WORKLOADS)
    assert any(w['profile'] == 'vkl' and (w['height'] == 1 or w['width'] == 1) for w in WORKLOADS)
    assert any(w['profile'] == 'vkl' and w['height'] % 2 for w in WORKLOADS)
    assert any(w['profile'] == 'slam' and (w['width'], w['height']) == (160, 120) for w in WORKLOADS)
    assert {'cpu_reference', 'isolated_gpu', 'resident_pyramid', 'host_completion', 'transfer_inclusive', 'slam_tracking'} <= set(BOUNDARIES)
    assert source_manifest()['vkl_donor']['sha256'] != source_manifest()['slam_donor']['sha256']


def test_interleaves_resets_and_checks_all_outputs():
    f = Fake(); result = run(f, samples=3, warmups=1)
    assert result['order'] == ['old_before', 'candidate', 'old_after'] * 3
    assert all(f.events[i][0] == 'reset' for i in range(0, len(f.events), 2))


def test_jointly_wrong_or_truncated_incumbent_and_candidate_rejected():
    class Truncated(Fake):
        def output(self, inputs):
            x = super().output(inputs); x['levels'].pop(); return x
    with pytest.raises(ValueError, match='incomplete'): run(Truncated())
    with pytest.raises(ValueError, match='complete native'): run(Fake(levels=1))
    class Wrong(Fake):
        def output(self, inputs):
            x = super().output(inputs); x['levels'][1][0][0,0,0] = 8; return x
    with pytest.raises(AssertionError): run(Wrong())


def test_measured_output_and_input_mutation_rejected():
    class BadMeasured(Fake):
        def measure(self, variant, inputs, boundary):
            value, output = super().measure(variant, inputs, boundary)
            if variant == 'candidate': output['levels'][1][0][0,0,0] += 1
            return value, output
    with pytest.raises(AssertionError): run(BadMeasured())
    class Mutating(Fake):
        def measure(self, variant, inputs, boundary): inputs['depth'][0,0,0] = 9; return 1
    with pytest.raises(ValueError, match='input'): run(Mutating())


def test_identity_and_order_checks():
    with pytest.raises(ValueError): run(Fake(), identities={})
    bad = identities(); bad['environment']['driver'] = 'unknown'
    with pytest.raises(ValueError): run(Fake(), identities=bad)
    runs = [run(Fake()) for _ in range(3)]
    assert evaluate(runs)['status'] == 'pass'
    assert evaluate(runs[:2])['status'] == 'inconclusive'
    changed = copy.deepcopy(runs); changed[1]['identities']['environment']['driver'] = 'different'
    assert evaluate(changed)['status'] == 'inconclusive'
    changed = copy.deepcopy(runs); changed[1]['order'][0] = 'candidate'
    assert evaluate(changed)['status'] == 'inconclusive'
    changed = copy.deepcopy(runs); changed[1]['controls'] = None
    assert evaluate(changed)['status'] == 'inconclusive'


def test_per_run_regression_not_hidden_by_pooling():
    runs = [run(Fake(), samples=100) for _ in range(3)]
    for r in runs[:2]: r['samples_ms'] = {'old_before': [10.]*100, 'old_after': [10.]*100, 'candidate': [9.]*100}
    runs[2]['samples_ms'] = {'old_before': [1.]*80+[10.]*20, 'old_after': [1.]*80+[10.]*20, 'candidate': [2.]*80+[9.]*20}
    assert evaluate(runs)['status'] == 'inconclusive'


def test_real_profile_fixtures_and_vkl_native_mask():
    vkl = next(w for w in WORKLOADS if w['profile'] == 'vkl' and w['height'] == 1)
    slam = next(w for w in WORKLOADS if w['profile'] == 'slam')
    a = make_inputs(vkl); np.testing.assert_array_equal(a['depth'], make_inputs(vkl)['depth'])
    assert len(cpu_reference(a, profile='vkl', threshold=VKL_DONOR_EPSILON)['levels']) == 1
    result = cpu_reference(make_inputs(slam), profile='slam', threshold=.01)
    assert [x.shape for x in result['levels']] == [(120,160), (60,80), (30,40), (15,20)]
    x = {'depth': np.array([[[5.]]], np.float32), 'valid': np.array([[[0]]], np.uint32), 'confidence': np.array([[[.5]]], np.float32)}
    result = cpu_reference(x, profile='vkl', threshold=VKL_DONOR_EPSILON)
    assert len(result['levels']) == 1 and result['levels'][0][0].item() == result['levels'][0][2].item() == 0


def test_tracking_requires_independent_pose_and_status():
    f = Fake(profile='slam'); x = {'depth': np.ones((8,8), np.float32)}
    with pytest.raises(ValueError, match='pose and status'): run(f, x, boundary='slam_tracking')
    extra = {'pose': np.eye(4, dtype=np.float32), 'status': np.array([1,0], np.uint32)}
    f = Fake(profile='slam', extra=extra)
    assert run(f, x, boundary='slam_tracking')['actual_levels'] == 4


def test_vkl_fixed_donor_epsilon_changes_discontinuity_branch():
    from sintergpu.references.depth_pyramid_profiles import vkl_level
    depth = np.array([[[1., 1.0001], [1., 1.0001]]], np.float32)
    valid = np.ones_like(depth, dtype=np.uint32)
    confidence = np.ones_like(depth)
    donor = vkl_level(depth, valid, confidence, np.finfo(np.float32).eps)[0]
    alternate = vkl_level(depth, valid, confidence, .01)[0]
    assert donor.item() == 1.0 and alternate.item() > donor.item()
    with pytest.raises(ValueError, match='donor epsilon'):
        run(Fake(threshold=.01))
