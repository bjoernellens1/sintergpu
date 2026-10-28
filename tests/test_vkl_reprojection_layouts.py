"""Setup-only graph contracts: explicit scalar32 storage and dispatch limits."""
import math
import pytest

from sintergpu.reprojection import reprojection_plan, temporal_reprojection_plan

LIMIT = 65535


def test_projection_graph_resources_and_borrowed_bindings():
    graph = reprojection_plan(2, 3, 5, 7, 9, max_workgroups_x=LIMIT, target_depth=True)
    assert set(graph) == {'resources', 'stages', 'outputs'}
    assert graph['resources']['source_depth'] == dict(elements=30, dtype='float32', role='input')
    assert graph['resources']['target_depth']['elements'] == 126
    assert graph['resources']['projected_xy']['elements'] == 60
    assert graph['resources']['visible']['dtype'] == 'uint32'
    assert graph['outputs']['overlap'] == 'overlap'
    assert set(graph['outputs']) == {'projected_xy', 'projected_depth', 'visible', 'displacement', 'overlap', 'parallax', 'depth_consistency', 'depth_consistency_valid', 'occluded', 'occlusion_valid'}
    assert [s['entrypoint'] for s in graph['stages']] == ['depthGeometry', 'projectPoints', 'reduceProjection']
    projection = graph['stages'][1]
    assert projection['thread_count'] == (512, 1, 1)
    assert projection['params']['geometryRepeat'] == 1
    for stage in graph['stages']:
        assert set(stage) == {'operation', 'entrypoint', 'thread_count', 'params', 'bindings', 'barriers'}
        assert all(name in graph['resources'] for name in stage['bindings'].values())
    assert 'camera_xyz' in projection['barriers']


def test_raster_passes_and_max_count_storage():
    graph = reprojection_plan(1, 17, 19, 3, 7, max_workgroups_x=LIMIT, raster=True)
    assert [s['entrypoint'] for s in graph['stages']][-4:] == ['clearRaster', 'rasterMin', 'rasterSupport', 'finalizeRaster']
    assert graph['resources']['raster_min']['dtype'] == 'uint32'
    assert graph['resources']['support_count']['dtype'] == 'uint32'
    assert set(graph['outputs']) == {'depth', 'valid', 'confidence', 'support_count'}
    assert graph['resources']['projected_xy']['role'] == 'scratch'
    assert graph['resources']['source_confidence']['elements'] == 323
    assert 'depth_consistency' not in graph['outputs']


def test_temporal_reference_uses_pair_repeats_and_source_major_fusion():
    graph = temporal_reprojection_plan(2, 3, 5, 7, max_workgroups_x=LIMIT, reference=True)
    assert set(graph['outputs']) == {'rgb', 'depth', 'valid', 'confidence', 'support_count', 'temporal_support_count', 'source_disagreement'}
    assert graph['resources']['source_depth']['elements'] == 210
    assert graph['resources']['packed_depth']['elements'] == 280
    stages = graph['stages']
    projects = [s for s in stages if s['entrypoint'] == 'projectPoints']
    assert len(projects) == 2
    assert (projects[0]['params']['geometryRepeat'], projects[0]['params']['sourceCameraRepeat'], projects[0]['params']['targetCameraRepeat']) == (3, 3, 1)
    assert (projects[1]['params']['geometryRepeat'], projects[1]['params']['sourceCameraRepeat'], projects[1]['params']['targetCameraRepeat']) == (1, 1, 3)
    assert [s['entrypoint'] for s in stages][-3:] == ['packTemporal', 'temporalDepthFuse', 'finalizeReference']
    fusion = next(s for s in stages if s['entrypoint'] == 'temporalDepthFuse')
    assert fusion['params']['sources'] == 4
    assert fusion['bindings']['visibility'] == fusion['bindings']['valids']
    assert fusion['bindings']['consistency'] == fusion['bindings']['valids']


@pytest.mark.parametrize('depth,confidence', [(False, False), (True, False), (False, True), (True, True)])
def test_temporal_optional_inputs(depth, confidence):
    graph = temporal_reprojection_plan(1, 2, 3, 5, max_workgroups_x=LIMIT, source_depth=depth, source_confidence=confidence)
    assert set(graph['outputs']) == {'aligned_rgb', 'aligned_valid', 'alignment_confidence'}
    assert ('source_depth' in graph['resources']) == depth
    assert ('source_confidence' in graph['resources']) == confidence


@pytest.mark.parametrize('function,args', [(reprojection_plan, (0, 3, 5, 7, 9)), (temporal_reprojection_plan, (0, 2, 3, 5))])
def test_zero_batch_has_no_dispatches(function, args):
    graph = function(*args, max_workgroups_x=LIMIT)
    assert graph['stages'] == []
    assert all(graph['resources'][name]['elements'] == 0 for name in graph['outputs'].values())


@pytest.mark.parametrize('args,kwargs', [
    ((True, 3, 5, 7, 9), {}), ((1, 0, 5, 7, 9), {}), ((1, 3, 5, 0, 9), {}),
    ((1, 3, 5, 7, 9), {'raster': 1}), ((1, 3, 5, 7, 9), {'target_depth': 1}),
    ((1, 3, 5, 7, 9), {'consistency_threshold': 0}),
    ((1, 3, 5, 7, 9), {'consistency_threshold': math.inf}),
    ((1, 3, 5, 7, 9), {'consistency_threshold': 1e-50}),
    ((1, 3, 5, 7, 9), {'consistency_threshold': True}),
    ((1, 2**24 + 1, 1, 1, 1), {}), ((1, 65536, 65536, 1, 1), {}),
    ((2**28, 1, 1, 1, 1), {}), ((1, 1, 1, 65536, 65536), {}),
])
def test_reprojection_invalid_shapes_flags_and_capacity(args, kwargs):
    with pytest.raises(ValueError):
        reprojection_plan(*args, max_workgroups_x=LIMIT, **kwargs)


@pytest.mark.parametrize('kwargs', [{'neighbors': 0}, {'neighbors': True}, {'source_depth': 1}, {'reference': 1}, {'source_confidence': 1}, {'min_support': True}, {'min_support': 0}])
def test_temporal_invalid_flags_and_counts(kwargs):
    base = dict(batches=1, neighbors=2, height=3, width=5, max_workgroups_x=LIMIT)
    base.update(kwargs)
    with pytest.raises(ValueError):
        temporal_reprojection_plan(**base)


def test_dispatch_limit_checks_include_padded_pair_projection_and_reduce():
    with pytest.raises(ValueError):
        reprojection_plan(2, 1, 257, 1, 1, max_workgroups_x=3)
    with pytest.raises(ValueError):
        temporal_reprojection_plan(1, 4, 1, 1, max_workgroups_x=3)
    with pytest.raises(ValueError):
        temporal_reprojection_plan(1, 2**30, 1, 1, max_workgroups_x=LIMIT)


def test_reused_geometry_and_raster_reset_declare_write_barriers():
    graph = reprojection_plan(1, 3, 5, 3, 5, max_workgroups_x=LIMIT, raster=True)
    first = graph['stages'][0]
    assert 'camera_xyz' in first['barriers']
    clear = next(s for s in graph['stages'] if s['entrypoint'] == 'clearRaster')
    assert set(clear['barriers']) == {'raster_min', 'raster_confidence', 'raster_support'}


def test_threshold_preserves_explicit_float32_rounding():
    graph = reprojection_plan(1, 1, 1, 1, 1, max_workgroups_x=LIMIT, consistency_threshold=.05)
    assert graph['stages'][1]['params']['threshold'] == 0.05000000074505806


def test_setup_module_is_cpu_only():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/sintergpu/reprojection.py').read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names]
    assert not any(name and any(token in name for token in ('slangpy', 'torch', 'numpy', 'vulkan')) for name in modules)


def test_padded_projection_thread_product_overflow_is_rejected():
    with pytest.raises(ValueError, match='padded'):
        reprojection_plan(20_000_000, 1, 1, 1, 1, max_workgroups_x=0xffffffff)
