"""Experimental point-plane dispatch ABI and frozen CPU fixture, no GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / 'src/sintergpu/kernels'
ADAPTER = KERNELS / 'optimization/point_plane_aosoa_256.slang'
FIXTURE = ROOT / 'tests/fixtures/point_plane_aosoa_1025.npz'
CANDIDATE = ROOT / 'tests/fixtures/point_plane_aosoa_1025_candidate.slang'
NAMES = ['points', 'targets', 'normals', 'out_0', 'out_1', 'points_indices', 'params']
FIELDS = ['problem_size', 'points_num_alloc', 'targets_num_alloc', 'normals_num_alloc', 'out_0_num_alloc', 'out_1_num_alloc']


def test_frozen_candidate_fixture_and_layout():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == '3a9e10f09ee3a312c52c00fe683bae8af09cf7e43b026c51c5ffa974c6edae9f'
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == '349f631d9d523409a5f8859de3e0480913b4a23ff7455306174c165db9391b4e'
    with np.load(FIXTURE, allow_pickle=False) as f:
        assert set(f.files) == set(NAMES[:-1])
        assert f['points'].shape == (512, 4) and f['targets'].shape == f['normals'].shape == (1032, 4)
        assert f['points_indices'].shape == (1025,) and f['points_indices'].dtype == np.uint32
        assert np.all(f['points_indices'] < 512) and np.all(f['points_indices'][:4] == 0)
        assert f['out_0'].shape == (1039,) and f['out_1'].shape == (6199,)
        assert np.all(f['out_0'] == -901) and np.all(f['out_1'] == -902)
        p = f['points'][f['points_indices'], :3].astype(np.float64)
        n = f['normals'][:1025, :3].astype(np.float64)
        q = f['targets'][:1025, :3].astype(np.float64)
        assert np.sum(n[0] * (p[0] - q[0])) == 0
        assert np.array_equal(np.cross(p[0], n[0]), np.zeros(3))
        assert np.isfinite(np.sum(n * (p-q), axis=1)).all()


def test_adapter_body_is_the_frozen_candidate_with_shared_import():
    core = (KERNELS / 'optimization/point_plane.slang').read_text()
    candidate = CANDIDATE.read_text()
    assert candidate.count(core) == 1
    expected = '// Experimental 1025-element AoSoA profile; caller owns all resources.\n' + candidate.replace(core, 'import optimization.point_plane;\n')
    assert ADAPTER.read_text() == expected


@pytest.mark.parametrize('version', ('2026.12', '2026.19'))
def test_reflection_matches_frozen_dispatch(version, tmp_path):
    key = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    override = os.environ.get(key)
    compiler = Path(override) if override else ROOT / '.tools' / ('slang-' + version) / 'bin/slangc'
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        if key in os.environ: pytest.fail(f'invalid explicit compiler {key}')
        pytest.skip(f'Slang {version} unavailable')
    result = subprocess.run([str(compiler), '-version'], capture_output=True, text=True, check=True)
    assert (result.stdout + result.stderr).strip() == version
    reflections = []
    for index, shader in enumerate((CANDIDATE, ADAPTER)):
        spv = tmp_path / f'{index}.spv'; reflection = tmp_path / f'{index}.json'
        run = subprocess.run([str(compiler), str(shader), '-entry', 'main', '-target', 'spirv', '-profile', 'glsl_460', '-I', str(KERNELS), '-o', str(spv), '-reflection-json', str(reflection)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        data = json.loads(reflection.read_text()); ep = data['entryPoints'][0]
        assert ep['threadGroupSize'] == [256, 1, 1]
        bindings = {b['name']: b['binding']['index'] for b in ep['bindings'] if b['binding'].get('used') == 1}
        assert bindings == dict(zip(NAMES, range(7)))
        params = next(x for x in data['parameters'] if x['name'] == 'params')
        fields = params['type']['elementType']['fields']
        assert [f['name'] for f in fields] == FIELDS
        assert [f['binding']['offset'] for f in fields] == list(range(0, 24, 4))
        reflections.append((bindings, fields))
    assert reflections[0] == reflections[1]


def valid_ranges():
    sizes = {'points': 8192, 'targets': 16512, 'normals': 16512, 'out_0': 4156,
             'out_1': 24796, 'points_indices': 4100, 'params': 32}
    return {name: {'allocation_id': name, 'offset': 0, 'length': size} for name, size in sizes.items()}


def valid_limits():
    return {'max_workgroups_x': 5, 'max_group_size_x': 256, 'max_threads_per_group': 256,
            'max_buffer_bytes': 24796, 'max_uniform_buffer_bytes': 32}


def test_checked_fixture_plan_and_guards():
    from sintergpu.point_plane_aosoa_1025 import fixture_plan
    limits, ranges = valid_limits(), valid_ranges()
    kwargs = dict(limits=limits, ranges=ranges, allocations_verified=True, fixture_contents_verified=True)
    plan = fixture_plan(**kwargs)
    assert plan['groups'] == (5, 1, 1) and plan['thread_count'] == (1280, 1, 1)
    assert plan['params'] == (1025, 512, 1032, 1032, 1032, 1032, 0, 0)
    for key in ('allocations_verified', 'fixture_contents_verified'):
        with pytest.raises(ValueError): fixture_plan(**dict(kwargs, **{key: False}))
    for key in limits:
        changed = dict(limits); changed[key] -= 1
        with pytest.raises(ValueError): fixture_plan(**dict(kwargs, limits=changed))
    for name in ranges:
        changed = {k: dict(v) for k, v in ranges.items()}; changed[name]['length'] -= 4
        with pytest.raises(ValueError): fixture_plan(**dict(kwargs, ranges=changed))
    changed = {k: dict(v) for k, v in ranges.items()}; changed['out_1']['allocation_id'] = 'points'
    with pytest.raises(ValueError, match='overlap'): fixture_plan(**dict(kwargs, ranges=changed))
    changed = {k: dict(v) for k, v in ranges.items()}; changed['points']['offset'] = (1 << 64) - 4
    with pytest.raises(ValueError, match='overflow'): fixture_plan(**dict(kwargs, ranges=changed))
    changed = {k: dict(v) for k, v in ranges.items()}; changed['points']['offset'] = 2
    with pytest.raises(ValueError, match='alignment'): fixture_plan(**dict(kwargs, ranges=changed))
    # Read-only sharing is allowed only when the attested byte ranges do not conflict with outputs.
    changed = {k: dict(v) for k, v in ranges.items()}; changed['targets']['allocation_id'] = 'points'
    assert fixture_plan(**dict(kwargs, ranges=changed))['groups'] == (5, 1, 1)
