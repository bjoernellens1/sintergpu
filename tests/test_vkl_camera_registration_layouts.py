import ast
from pathlib import Path

import pytest

from sintergpu.camera_registration import (
    camera_rectify_grid_plan, camera_rectify_from_grid_plan, camera_rectify_plan,
    depth_camera_rays_plan, depth_register_plan,
)


def entries(plan):
    return [stage['entrypoint'] for stage in plan['stages']]


def test_rectification_composes_same_stages_and_complete_outputs():
    grid = camera_rectify_grid_plan(2, 3, 4, 5, 6, max_workgroups_x=20, distortion=True, debug=True)
    sample = camera_rectify_from_grid_plan(2, 3, 3, 4, 5, 6, max_workgroups_x=20, sampling='nearest')
    direct = camera_rectify_plan(2, 3, 3, 4, 5, 6, max_workgroups_x=20, distortion=True, debug=True, sampling='nearest')
    assert entries(direct) == entries(grid) + entries(sample) == ['rectifyGrid', 'rectifyFromGrid']
    assert set(grid['outputs']) == {'source_uv', 'valid', 'target_ray', 'displacement'}
    assert set(sample['outputs']) == {'image', 'valid'}
    assert set(direct['outputs']) == {'image', 'valid', 'source_uv', 'target_ray', 'displacement'}
    assert grid['resources']['source_uv']['shape'] == (2, 5, 6, 2)
    assert grid['resources']['target_ray']['shape'] == (2, 5, 6, 3)
    assert direct['stages'][1]['barriers'] >= ['source_uv', 'valid']
    assert direct['profile']['sampling'] == 'nearest'
    assert direct['profile']['cache']['caller_calibration_identity_required'] is True


def test_rays_and_registration_output_layouts_and_order():
    rays = depth_camera_rays_plan(2, 3, 4, max_workgroups_x=10, distortion=True)
    assert rays['resources'][rays['outputs']['depth_rays']]['shape'] == (2, 3, 4, 2)
    plan = depth_register_plan(2, 3, 4, 5, 6, max_workgroups_x=10, depth_distortion=True, rgb_distortion=True, debug=True)
    assert entries(plan) == ['depthCameraRays', 'clearRegistration', 'projectRegistration', 'registrationMin', 'registrationWinnerCount', 'finalizeRegistrationDebug']
    assert set(plan['outputs']) == {'depth', 'valid', 'winner_source_index', 'collision_count', 'source_uv', 'projected_z', 'source_valid'}
    for key in ('winner_source_index', 'collision_count'):
        resource = plan['resources'][plan['outputs'][key]]
        assert resource['dtype'] == 'int64' and resource['element_size'] == 8
        assert resource['shape'] == (2, 1, 5, 6)
    assert plan['resources']['source_uv']['shape'] == (2, 12, 2)
    assert plan['resources']['projected_z']['shape'] == (2, 12)
    assert 'shaderInt64' in plan['features']
    stages = {s['entrypoint']: s for s in plan['stages']}
    assert {'min_depth_bits', 'winner_index', 'collision_count_scratch'} <= set(stages['clearRegistration']['barriers'])
    assert {'depth_rays'} <= set(stages['projectRegistration']['barriers'])
    assert {'source_uv', 'projected_z', 'source_valid', 'destination_index', 'min_depth_bits'} <= set(stages['registrationMin']['barriers'])
    assert {'min_depth_bits'} <= set(stages['registrationWinnerCount']['barriers'])
    assert {'min_depth_bits', 'winner_index', 'collision_count_scratch'} <= set(stages['finalizeRegistrationDebug']['barriers'])
    cached = depth_register_plan(2, 3, 4, 5, 6, max_workgroups_x=10, cached_rays=True)
    assert entries(cached) == entries(plan)[1:-1] + ['finalizeRegistration']
    assert cached['resources']['depth_rays']['role'] == 'input'
    assert cached['features'] == []
    assert cached['profile']['cache']['caller_calibration_identity_required']


@pytest.mark.parametrize('args', [(True, 1, 1, 1, 1), (0, 1, 1, 1, 1), (1, 0, 1, 1, 1), (1, 1, 1, 1, -1)])
def test_invalid_dimensions(args):
    with pytest.raises(ValueError):
        camera_rectify_grid_plan(*args, max_workgroups_x=1)


@pytest.mark.parametrize('kw', [dict(debug=1), dict(distortion=1), dict(sampling='cubic'), dict(max_workgroups_x=True)])
def test_invalid_flags(kw):
    options = dict(max_workgroups_x=10)
    options.update(kw)
    with pytest.raises(ValueError):
        camera_rectify_plan(1, 3, 3, 4, 3, 4, **options)


@pytest.mark.parametrize('minimum,maximum', [(0, None), (True, None), (float('nan'), None), (1e-100, None), (1, 1), (1, float('inf')), (1, 1e100)])
def test_invalid_depth_intervals(minimum, maximum):
    with pytest.raises(ValueError):
        depth_register_plan(1, 3, 4, 5, 6, max_workgroups_x=10, min_depth=minimum, max_depth=maximum)


def test_overflow_and_dispatch_fail_before_resource_creation():
    with pytest.raises(ValueError):
        depth_register_plan(1, 65535, 65537, 1, 1, max_workgroups_x=2**32-1)
    with pytest.raises(ValueError):
        camera_rectify_plan(1, 2**32-1, 1, 1, 1, 2, max_workgroups_x=100)
    with pytest.raises(ValueError):
        depth_camera_rays_plan(1, 17, 17, max_workgroups_x=1)
    with pytest.raises(ValueError, match='float32 pixel coordinates'):
        depth_camera_rays_plan(1, 1, 2**24 + 1, max_workgroups_x=2**32-1)


def test_recording_uses_prepared_bindings_and_caller_encoder_only():
    plan = depth_register_plan(1, 3, 4, 5, 6, max_workgroups_x=10, debug=True)
    prepared = [(s['entrypoint'], s['thread_count'], s['bindings'], s['barriers']) for s in plan['stages']]
    class Encoder:
        def __init__(self): self.calls = []
        def barrier(self, resources): self.calls.append(('barrier', resources))
        def dispatch(self, entry, count, bindings): self.calls.append(('dispatch', entry, count, bindings))
    encoder = Encoder()
    def record(encoder):
        for entry, count, bindings, barriers in prepared:
            encoder.barrier(barriers)
            encoder.dispatch(entry, count, bindings)
    record(encoder)
    record(encoder)
    assert [call[1] for call in encoder.calls if call[0] == 'dispatch'] == entries(plan) * 2
    assert all(encoder.calls[i + 1][3] is prepared[(i // 2) % len(prepared)][2] for i in range(0, len(encoder.calls), 2))
    source = Path('src/sintergpu/camera_registration.py').read_text()
    tree = ast.parse(source)
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imports & {'numpy', 'torch', 'slangpy'}


def test_cache_identity_requires_calibration_and_shader_and_rejects_changes():
    from sintergpu.camera_registration import cache_identity, validate_cache
    p = depth_camera_rays_plan(1, 3, 4, max_workgroups_x=10, distortion=True)
    identity = cache_identity(p, 'calibration-v1', 'shader-sha256')
    validate_cache(p, identity, 'calibration-v1', 'shader-sha256')
    for changed in ('calibration-v2', ''):
        with pytest.raises(ValueError):
            validate_cache(p, identity, changed, 'shader-sha256')
    with pytest.raises(ValueError):
        validate_cache(p, identity, 'calibration-v1', 'another-shader')
    changed = depth_camera_rays_plan(1, 4, 3, max_workgroups_x=10, distortion=True)
    with pytest.raises(ValueError):
        validate_cache(changed, identity, 'calibration-v1', 'shader-sha256')
    changed = depth_camera_rays_plan(1, 3, 4, max_workgroups_x=10, distortion=False)
    with pytest.raises(ValueError):
        validate_cache(changed, identity, 'calibration-v1', 'shader-sha256')


def test_every_bound_resource_exists_and_sizes_match_shapes():
    plans = [camera_rectify_grid_plan(1, 3, 4, 5, 6, max_workgroups_x=10),
             camera_rectify_from_grid_plan(1, 4, 3, 4, 5, 6, max_workgroups_x=10),
             camera_rectify_plan(1, 4, 3, 4, 5, 6, max_workgroups_x=10, debug=True, distortion=True),
             depth_camera_rays_plan(1, 3, 4, max_workgroups_x=10),
             depth_register_plan(1, 3, 4, 5, 6, max_workgroups_x=10, debug=True, rgb_distortion=True)]
    import math
    for plan in plans:
        for resource in plan['resources'].values():
            assert resource['elements'] == math.prod(resource['shape'])
            assert resource['element_size'] == (8 if resource['dtype'] == 'int64' else 4)
        for stage in plan['stages']:
            assert set(stage['bindings'].values()) <= set(plan['resources'])
            assert set(stage['barriers']) <= set(plan['resources'])


def test_calibration_cache_keys_match_composed_graphs():
    from sintergpu.camera_registration import cache_identity, validate_cache
    grid = camera_rectify_grid_plan(1, 3, 4, 5, 6, max_workgroups_x=10, distortion=True)
    direct = camera_rectify_plan(1, 3, 3, 4, 5, 6, max_workgroups_x=10, distortion=True, sampling='nearest')
    validate_cache(direct, cache_identity(grid, 'cal', 'shader'), 'cal', 'shader')
    rays = depth_camera_rays_plan(1, 3, 4, max_workgroups_x=10, distortion=True)
    register = depth_register_plan(1, 3, 4, 5, 6, max_workgroups_x=10, depth_distortion=True, cached_rays=True)
    validate_cache(register, cache_identity(rays, 'cal', 'shader'), 'cal', 'shader')
