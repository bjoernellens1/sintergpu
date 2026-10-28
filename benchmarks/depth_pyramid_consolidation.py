# SPDX-License-Identifier: Apache-2.0
"""Preregistered depth pyramid comparison protocol; no GPU backend is supplied.

A backend must reset state outside its measurement, validate all intermediate
levels, and define each boundary precisely. This module does not acquire device
locks or establish controlled conditions; callers must supply and retain those
receipts before using results for qualification.
"""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VKL_DONOR_EPSILON = float(np.finfo(np.float32).eps)
WORKLOADS = (
    {'profile': 'vkl', 'batch': 1, 'height': 1, 'width': 1},
    {'profile': 'vkl', 'batch': 2, 'height': 1, 'width': 17},
    {'profile': 'vkl', 'batch': 2, 'height': 17, 'width': 1},
    {'profile': 'vkl', 'batch': 3, 'height': 121, 'width': 161},
    {'profile': 'vkl', 'batch': 1, 'height': 480, 'width': 640},
    {'profile': 'slam', 'batch': 1, 'height': 120, 'width': 160},
    {'profile': 'slam', 'batch': 1, 'height': 480, 'width': 640},
)
BOUNDARIES = {
    'cpu_reference': 'independent CPU pyramid only',
    'isolated_gpu': 'one dispatch, GPU timestamps',
    'resident_pyramid': 'all pyramid dispatches, GPU timestamps',
    'host_completion': 'host recording, submission and completion',
    'transfer_inclusive': 'upload, execution, required readbacks and completion',
    'slam_tracking': 'resident tracking GPU pipeline with pose and status validation',
}
DONORS = {
    'vkl_donor': 'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py',
    'vkl_incumbent': 'src/sintergpu/kernels/vkl/depth_pyramid.slang',
    'slam_donor': 'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/shaders/tracker.slang',
    'slam_incumbent': 'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang',
}


def source_manifest():
    return {key: {'path': path, 'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
            for key, path in DONORS.items()}



def make_inputs(workload, *, seed=613):
    """Create deterministic, shared input buffers for a declared workload."""
    if workload not in WORKLOADS:
        raise ValueError('workload is not preregistered')
    rng = np.random.default_rng(seed)
    shape = (workload['batch'], workload['height'], workload['width'])
    depth = rng.uniform(.2, 4., size=shape).astype(np.float32)
    if workload['profile'] == 'slam':
        depth = depth[0]
        depth[rng.random(depth.shape) < .1] = 0
        return {'depth': depth}
    return {'depth': depth, 'valid': (rng.random(shape) >= .1).astype(np.uint32),
            'confidence': rng.uniform(0., 1., size=shape).astype(np.float32)}


def cpu_reference(inputs, *, profile, threshold, levels=3):
    """Evaluate distinct donor profiles; output includes input and each level.

    SLAM's literal scalar reference has not yet passed frozen GPU parity.
    """
    from sintergpu.references.depth_pyramid_profiles import vkl_level, slam_level
    if profile == 'vkl' and threshold != VKL_DONOR_EPSILON:
        raise ValueError('VKL donor epsilon must be float32 machine epsilon')
    if levels < 1: raise ValueError('levels must be positive')
    if profile == 'vkl':
        valid = (inputs['valid'] != 0) & np.isfinite(inputs['depth']) & (inputs['depth'] > 0)
        current = (np.where(valid, inputs['depth'], np.float32(0)), valid.astype(np.uint32), np.where(valid, inputs['confidence'], np.float32(0)))
        result = [tuple(x.copy() for x in current)]
        for _ in range(levels):
            if current[0].shape[-2:] == (1, 1): break
            current = vkl_level(*current, threshold)
            result.append(current)
    elif profile == 'slam':
        current = inputs['depth']
        result = [current.copy()]
        for _ in range(levels):
            if any(n % 2 for n in current.shape): break
            current = slam_level(current, threshold)
            result.append(current)
    else: raise ValueError('unknown profile')
    return {'levels': result}

def _digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x): h.update(k.encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b'list' if isinstance(x, list) else b'tuple')
            for item in x: add(item)
        elif isinstance(x, np.ndarray):
            h.update(x.dtype.str.encode()); h.update(repr(x.shape).encode()); h.update(x.tobytes(order='C'))
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def _equal(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys()
        for k in expected: _equal(expected[k], actual[k])
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual) and len(expected) == len(actual)
        for a, b in zip(expected, actual): _equal(a, b)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected, actual, strict=True)
    else:
        assert expected == actual


def _validate_identities(identities, profile):
    """Reject absent or unverifiable source and environment identities."""
    if not isinstance(identities, dict) or set(identities) != {'sources', 'environment'}:
        raise ValueError('source and environment identities required')
    sources, env = identities['sources'], identities['environment']
    required = {'candidate', 'incumbent', 'donor', 'reference', 'harness'}
    if not isinstance(sources, dict) or set(sources) != required:
        raise ValueError('complete source identities required')
    expected = {'incumbent': DONORS[profile + '_incumbent'], 'donor': DONORS[profile + '_donor'],
                'reference': 'src/sintergpu/references/depth_pyramid_profiles.py',
                'harness': 'benchmarks/depth_pyramid_consolidation.py'}
    for key, entry in sources.items():
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
            raise ValueError('source path and sha256 required')
        path = (ROOT / entry['path']).resolve()
        if key in expected and path != (ROOT / expected[key]).resolve():
            raise ValueError('unexpected source identity')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('source identity does not match file')
    keys = {'compiler_sha256', 'device', 'device_uuid', 'driver', 'environment'}
    if not isinstance(env, dict) or set(env) != keys or any(not isinstance(env[k], str) or not env[k].strip() or env[k].lower() in {'unknown', 'unavailable', 'none'} for k in keys):
        raise ValueError('complete observed environment identities required')
    if len(env['compiler_sha256']) != 64 or any(c not in '0123456789abcdef' for c in env['compiler_sha256']):
        raise ValueError('compiler SHA256 required')


def collect_run(backend, inputs, *, profile, boundary, samples, warmups, levels, threshold, identities, expected_extra=None, controls=None):
    """Collect measured snapshots against independent CPU references.

    GPU backends, controlled receipts and frozen-donor GPU validation remain
    unimplemented. Caller-supplied controls are claims, not verified evidence.
    """
    if profile not in ('vkl', 'slam') or boundary not in BOUNDARIES:
        raise ValueError('unknown profile or boundary')
    if boundary == 'slam_tracking' and profile != 'slam': raise ValueError('tracking requires SLAM profile')
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (samples, warmups, levels)) or samples < 1 or warmups < 0 or levels < 1:
        raise ValueError('invalid sample or level counts')
    if levels != 3: raise ValueError('complete native/half/quarter/eighth pyramid required')
    if not np.isfinite(threshold) or threshold < 0: raise ValueError('invalid threshold')
    _validate_identities(identities, profile)
    expected = cpu_reference(inputs, profile=profile, threshold=threshold, levels=levels)
    if expected_extra:
        if 'levels' in expected_extra: raise ValueError('independent levels cannot be overridden')
        expected.update(copy.deepcopy(expected_extra))
    if boundary == 'slam_tracking' and not {'pose', 'status'} <= expected.keys():
        raise ValueError('independent tracking pose and status required')
    input_hash = _digest(inputs)
    def check_input():
        if _digest(inputs) != input_hash: raise ValueError('input changed during collection')
    def check_output(output):
        if not isinstance(output, dict) or not isinstance(output.get('levels'), list) or len(output['levels']) != len(expected['levels']):
            raise ValueError('incomplete pyramid levels')
        _equal(expected, output)
    def execute(variant):
        backend.reset(variant, inputs); check_input()
        result = backend.execute(variant, inputs, boundary); check_input()
        check_output(result)
    execute('old'); execute('candidate')
    def measure(variant):
        backend.reset(variant, inputs); check_input()
        measurement = backend.measure(variant, inputs, boundary); check_input()
        value, observed = measurement
        if not np.isfinite(value) or value <= 0: raise ValueError('invalid timing')
        check_output(observed)
        return float(value)
    for _ in range(warmups):
        for variant in ('old', 'candidate', 'old'): measure(variant)
    timings = {k: [] for k in ('old_before', 'candidate', 'old_after')}
    order = []
    for _ in range(samples):
        for label, variant in (('old_before', 'old'), ('candidate', 'candidate'), ('old_after', 'old')):
            timings[label].append(measure(variant)); order.append(label)
    return {'profile': profile, 'boundary': boundary, 'input_sha256': input_hash,
            'levels': levels, 'actual_levels': len(expected['levels']), 'threshold': float(threshold),
            'identities': copy.deepcopy(identities), 'samples_ms': timings, 'order': order,
            'warmups': warmups, 'controls': copy.deepcopy(controls),
            'scope': 'protocol only; hardware and control receipts are not independently verified'}


def evaluate(runs, *, bootstrap_samples=10000, seed=613):
    """Test each run independently; a statistical pass is not qualification."""
    def inconclusive(reason): return {'status': 'inconclusive', 'reason': reason}
    if len(runs) != 3: return inconclusive('three runs required')
    keys = ('profile', 'boundary', 'input_sha256', 'levels', 'actual_levels', 'threshold', 'identities', 'warmups')
    if any(any(k not in r for k in keys) for r in runs): return inconclusive('missing identity')
    if any(any(r[k] != runs[0][k] for k in keys) for r in runs[1:]): return inconclusive('mismatched workload or identity')
    if runs[0]['profile'] == 'vkl' and runs[0]['threshold'] != VKL_DONOR_EPSILON:
        return inconclusive('VKL donor epsilon mismatch')
    try: _validate_identities(runs[0]['identities'], runs[0]['profile'])
    except (ValueError, KeyError, TypeError, OSError): return inconclusive('invalid source or environment identity')
    if bootstrap_samples < 1000: return inconclusive('insufficient bootstrap samples')
    rng = np.random.default_rng(seed); summaries = []
    for run in runs:
        control = run.get('controls')
        if not isinstance(control, dict) or control.get('exclusive') is not True or control.get('interference_free') is not True:
            return inconclusive('controlled run and interference evidence required')
        receipt = control.get('receipt_sha256')
        if not isinstance(receipt, str) or len(receipt) != 64 or any(c not in '0123456789abcdef' for c in receipt):
            return inconclusive('control receipt identity required')
        try:
            s = run['samples_ms']
            a, b, c = (np.asarray(s[k], dtype=float) for k in ('old_before', 'candidate', 'old_after'))
        except (KeyError, ValueError, TypeError): return inconclusive('invalid samples')
        if len(a) < 30 or a.ndim != 1 or len(a) != len(b) or len(a) != len(c) or not all(np.isfinite(x).all() and (x > 0).all() for x in (a, b, c)):
            return inconclusive('insufficient or invalid samples')
        if run.get('order') != ['old_before', 'candidate', 'old_after'] * len(a):
            return inconclusive('invalid paired order')
        old = (a + c) / 2; d = b - old
        boots = np.array([np.median(rng.choice(d, len(d), replace=True)) for _ in range(bootstrap_samples)])
        ci = np.percentile(boots, [2.5, 97.5]); old95 = np.percentile(old, 95); new95 = np.percentile(b, 95)
        summaries.append({'old_median_ms': float(np.median(old)), 'candidate_median_ms': float(np.median(b)),
                          'old_p95_ms': float(old95), 'candidate_p95_ms': float(new95),
                          'median_difference_95ci_ms': ci.tolist()})
        if ci[1] > 0 or new95 > old95:
            return {'status': 'inconclusive', 'reason': 'per-run no-regression gate failed', 'runs': summaries}
    return {'status': 'pass', 'runs': summaries,
            'scope': 'statistical protocol only; hardware, control receipts, and backend remain unqualified'}
