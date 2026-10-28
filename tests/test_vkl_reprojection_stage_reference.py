"""Independent per-dispatch NumPy oracles; no pipeline substitution in timing."""
import numpy as np
import pytest

from sintergpu.references.reprojection_stage_numeric import run_stage


def params(**updates):
    values = dict(pairs=1, height=1, width=1, targetHeight=1, targetWidth=1,
                  geometryRepeat=1, sourceCameraRepeat=1, targetCameraRepeat=1,
                  targetDepthRepeat=1, useTargetDepth=0, useConfidence=0,
                  groupsPerPair=1, neighbors=1, batches=1, threshold=.05)
    values.update(updates)
    return values


def f(*values): return np.array(values, np.float32)
def u(*values): return np.array(values, np.uint32)


def test_clear_writes_preallocated_active_elements_and_preserves_tails():
    bindings = {key: u(7, 8, 73) for key in ('minDepthBits', 'maxConfidenceBits', 'supportBits')}
    out = run_stage('clearRaster', params(targetWidth=2), bindings)
    assert out.keys() == bindings.keys()
    for key in bindings:
        assert out[key] is bindings[key]
        np.testing.assert_array_equal(out[key], [0x7f800000 if key == 'minDepthBits' else 0] * 2 + [73])


def test_raster_min_and_support_preserve_existing_inout_and_count_every_close_contributor():
    eps = np.finfo(np.float32).eps
    bindings = dict(projected_xyOut=f(0, .01, .02, 0, 0, 0), projected_depthOut=f(1, 1 + eps, 2),
                    visibleOut=u(1, 1, 1), sourceConfidence=f(-0., .9, 1),
                    minDepthBits=f(3, 73).view(np.uint32), maxConfidenceBits=f(.2, 73).view(np.uint32), supportBits=u(4, 73))
    p = params(width=3, useConfidence=1)
    out = run_stage('rasterMin', p, bindings)
    assert set(out) == {'minDepthBits'}
    np.testing.assert_array_equal(out['minDepthBits'].view(np.float32), [1, 73])
    out = run_stage('rasterSupport', p, bindings)
    assert set(out) == {'maxConfidenceBits', 'supportBits'}
    np.testing.assert_array_equal(out['maxConfidenceBits'].view(np.float32), f(.9, 73))
    np.testing.assert_array_equal(out['supportBits'], [6, 73])


def test_project_uses_precomputed_xyz_and_unconditional_invalid_translation():
    pose = np.eye(4, dtype=np.float32); pose[:3, 3] = [2, 3, 4]
    bindings = dict(cameraXYZ=f(0, 0, 0), geometryValid=u(0), sourcePose=pose.reshape(-1),
                    targetPose=np.eye(4, dtype=np.float32).reshape(-1), targetIntrinsics=np.eye(3, dtype=np.float32).reshape(-1),
                    projected_xyOut=f(73, 73, 73), projected_depthOut=f(73, 73), visibleOut=u(73, 73),
                    displacementOut=f(73, 73), partialCount=u(73, 73), partialParallax=f(73, 73))
    out = run_stage('projectPoints', params(targetHeight=3, targetWidth=4), bindings)
    np.testing.assert_array_equal(out['projected_xyOut'], [.5, .75, 73])
    np.testing.assert_array_equal(out['projected_depthOut'], [4, 73])
    np.testing.assert_array_equal(out['visibleOut'], [0, 73])
    np.testing.assert_array_equal(out['partialCount'], [0, 73])
    np.testing.assert_array_equal(out['partialParallax'], [0, 73])


def test_reduce_consumes_only_partials_and_keeps_nonfinite_diagnostic():
    bindings = dict(partialCount=u(2, 1, 73), partialParallax=f(4, np.nan, 73), overlapOut=f(73, 73), parallaxOut=f(73, 73))
    out = run_stage('reduceProjection', params(width=257, groupsPerPair=2), bindings)
    np.testing.assert_allclose(out['overlapOut'], f(3 / 257, 73), rtol=1e-7)
    assert np.isnan(out['parallaxOut'][0]) and out['parallaxOut'][1] == 73


def test_project_optional_nearest_half_tie_and_invalid_nonfinite_partial():
    bindings = dict(cameraXYZ=f(.5, 0, 1), geometryValid=u(1), sourcePose=np.eye(4, dtype=np.float32).reshape(-1),
                    targetPose=np.eye(4, dtype=np.float32).reshape(-1), targetIntrinsics=np.eye(3, dtype=np.float32).reshape(-1),
                    targetDepth=f(11, 22, 33, 44), projected_xyOut=f(73, 73, 73), projected_depthOut=f(73, 73),
                    visibleOut=u(73, 73), displacementOut=f(73, 73), partialCount=u(73, 73), partialParallax=f(73, 73),
                    depth_consistencyOut=f(73, 73), depth_consistency_validOut=u(73, 73), occludedOut=u(73, 73), occlusion_validOut=u(73, 73))
    p = params(targetWidth=4, useTargetDepth=1)
    out = run_stage('projectPoints', p, bindings)
    np.testing.assert_array_equal(out['depth_consistencyOut'], [21, 73])
    bindings['cameraXYZ'][:] = f(1e30, 0, 1)
    out = run_stage('projectPoints', p, bindings)
    assert np.isinf(out['displacementOut'][0]) and np.isnan(out['partialParallax'][0])
    assert out['visibleOut'][0] == out['depth_consistency_validOut'][0] == 0


def test_sampler_unmasked_singletons_optional_consistency_and_nonfinite_grid():
    p = params(pairs=2, useTargetDepth=1, useConfidence=1)
    bindings = dict(sourceRGB=f(2, 3, 4, 5, 6, 7), sourceConfidence=f(.5, .7),
                    projected_xyOut=f(100, -100, np.nan, 0), visibleOut=u(1, 0),
                    depth_consistencyOut=f(.05, 0), depth_consistency_validOut=u(1, 0), occludedOut=u(0, 0),
                    aligned_rgbOut=np.full(8, 73, np.float32), aligned_validOut=u(73, 73, 73), alignment_confidenceOut=f(73, 73, 73))
    out = run_stage('sampleTemporal', p, bindings)
    np.testing.assert_array_equal(out['aligned_rgbOut'][:3], [2, 3, 4])
    assert np.isnan(out['aligned_rgbOut'][3:6]).all()
    np.testing.assert_allclose(out['alignment_confidenceOut'][0], np.exp(-np.float32(1)) * np.float32(.5), rtol=1e-7)
    assert np.isnan(out['alignment_confidenceOut'][1])
    np.testing.assert_array_equal(out['aligned_validOut'], [1, 0, 73])
    np.testing.assert_array_equal(out['aligned_rgbOut'][6:], [73, 73])


def test_pack_batch_neighbor_layout_zero_target_and_trust_and_preserve_tails():
    p = params(pairs=6, batches=2, neighbors=3, width=2)
    count = 12; packed_count = 16
    bindings = dict(depthOut=np.arange(count, dtype=np.float32), confidenceOut=np.linspace(0, 1, count, dtype=np.float32),
                    validOut=np.ones(count, np.uint32),
                    packedDepth=np.full(packed_count + 2, 73, np.float32), packedConfidence=np.full(packed_count + 2, 73, np.float32),
                    packedValid=np.full(packed_count + 2, 73, np.uint32), packedTrusted=np.full(packed_count + 2, 73, np.uint32))
    out = run_stage('packTemporal', p, bindings)
    np.testing.assert_array_equal(out['packedDepth'][:packed_count], [0, 0, 0, 0, 0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
    np.testing.assert_array_equal(out['packedTrusted'][:packed_count], 0)
    for value in out.values(): np.testing.assert_array_equal(value[packed_count:], [73, 73])


def test_finalize_reference_uses_every_rgb_weight_and_masks_only_four_outputs():
    p = params(pairs=3, neighbors=3)
    bindings = dict(aligned_rgbOut=f(0, 0, 0, 0, 0, 0, 1, 1, 1), aligned_validOut=u(1, 1, 1),
                    alignment_confidenceOut=f(1, 1, .25), rasterConfidence=f(1, 1, 1), rasterValid=u(1, 1, 1),
                    fusedDepth=f(1), fusedValid=u(1), fusedConfidence=f(.9),
                    rgbOut=np.full(4, 73, np.float32), depthOut=f(73, 73), validOut=u(73, 73), confidenceOut=f(73, 73))
    out = run_stage('finalizeReference', p, bindings)
    assert set(out) == {'rgbOut', 'depthOut', 'validOut', 'confidenceOut'}
    np.testing.assert_allclose(out['rgbOut'][:3], 1 / 9, rtol=1e-7)
    np.testing.assert_array_equal(out['confidenceOut'], f(.9, 73))
    bindings['aligned_validOut'][:] = 0
    out = run_stage('finalizeReference', p, bindings)
    np.testing.assert_array_equal(out['rgbOut'], [0, 0, 0, 73])
    np.testing.assert_array_equal(out['depthOut'], [0, 73])


def execute_graph_cpu(graph, inputs):
    """Prepare geometry and D9 prefixes outside the nine new dispatch oracles."""
    from sintergpu.references.geometry_depth_numeric import depth_geometry, temporal_depth_fuse
    dtypes = {'float32': np.float32, 'uint32': np.uint32, 'int32': np.int32}
    buffers = {name: np.full(resource['elements'] + 13, 73, dtypes[resource['dtype']]) for name, resource in graph['resources'].items()}
    for name, value in inputs.items(): buffers[name][:value.size] = value.reshape(-1)
    written = set()
    for stage in graph['stages']:
        p = stage['params']; bindings = {name: buffers[resource] for name, resource in stage['bindings'].items()}
        if stage['entrypoint'] == 'depthGeometry':
            b, h, w = p['batches'], p['height'], p['width']
            expected = depth_geometry(bindings['depth'][:b*h*w].reshape(b, 1, h, w), bindings['intrinsics'][:9*b].reshape(b, 3, 3))
            for key, value in expected.items(): bindings[key + 'Out'][:value.size] = value.reshape(-1)
        elif stage['entrypoint'] == 'temporalDepthFuse':
            count, sources = p['count'], p['sources']
            shape = (sources, 1, 1, 1, count)
            arrays = [bindings[name][:sources*count].reshape(shape) for name in ('depths', 'valids', 'confidences', 'visibility', 'consistency', 'trusted')]
            expected = temporal_depth_fuse(*arrays, consistency_threshold=p['threshold'], min_support=p['minSupport'], min_temporal_support=p['minTemporalSupport'])
            for key, value in expected.items(): bindings[key + 'Out'][:value.size] = value.reshape(-1)
        else:
            before = {name: value.copy() for name, value in bindings.items()}
            output = run_stage(stage['entrypoint'], p, bindings)
            written.add(stage['entrypoint'])
            for key, array in output.items(): assert array is bindings[key]
            # Readonly bindings stay byte-identical, including NaN payloads.
            for key in set(bindings) - set(output):
                np.testing.assert_array_equal(bindings[key].view(np.uint32), before[key].view(np.uint32), err_msg=stage['entrypoint'] + ':' + key)
    for name, resource in graph['resources'].items(): np.testing.assert_array_equal(buffers[name][resource['elements']:], 73, err_msg=name)
    return buffers, written


def compare_graph_outputs(graph, buffers, expected):
    assert graph['outputs'].keys() == expected.keys()
    for key, array in expected.items():
        actual = buffers[graph['outputs'][key]][:array.size].reshape(array.shape)
        if array.dtype.kind in 'biu': np.testing.assert_array_equal(actual, array, err_msg=key)
        else: np.testing.assert_allclose(actual, array, rtol=3e-6, atol=2e-6, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('b,h,w,th,tw', [(1, 3, 5, 5, 7), (2, 17, 19, 13, 23), (1, 1, 1, 1, 1)])
@pytest.mark.parametrize('target_depth,raster', [(False, False), (True, False), (False, True), (True, True)])
def test_stage_composition_matches_full_d4_oracles(b, h, w, th, tw, target_depth, raster):
    from sintergpu.reprojection import reprojection_plan
    from sintergpu.references.reprojection_numeric import reproject_rgbd, rasterize_reprojection
    rng = np.random.default_rng(57)
    k = np.tile(np.array([[4.3, 0, .31], [0, 3.7, .17], [0, 0, 1]], np.float32), (b, 1, 1))
    tk = k.copy(); tk[:, 0, 0] = .2
    pose = np.tile(np.eye(4, dtype=np.float32), (b, 1, 1)); source = pose.copy()
    angle = np.float32(.017)
    source[:, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    source[:, :3, 3] = [.13, -.07, .21]
    depth = rng.uniform(.9, 1.1, (b, 1, h, w)).astype(np.float32)
    depth.reshape(-1)[:min(4, depth.size)] = [0, np.nan, np.inf, -1][:min(4, depth.size)]
    confidence = rng.random(depth.shape, dtype=np.float32)
    td = rng.uniform(.5, 2, (b, 1, th, tw)).astype(np.float32)
    inputs = dict(source_depth=depth, source_intrinsics=k, source_pose_c2w=source, target_intrinsics=tk, target_pose_c2w=pose)
    if target_depth: inputs['target_depth'] = td
    if raster: inputs['source_confidence'] = confidence
    graph = reprojection_plan(b, h, w, th, tw, max_workgroups_x=65535, target_depth=target_depth, raster=raster)
    buffers, written = execute_graph_cpu(graph, inputs)
    expected = rasterize_reprojection(depth, confidence, k, source, tk, pose, target_hw=(th, tw)) if raster else reproject_rgbd(depth, k, source, tk, pose, td if target_depth else None, target_hw=(th, tw))
    compare_graph_outputs(graph, buffers, expected)
    assert {'projectPoints', 'reduceProjection'} <= written


@pytest.mark.parametrize('b,n,h,w', [(1, 1, 1, 1), (2, 3, 3, 5), (8, 3, 3, 5), (1, 2, 17, 19)])
@pytest.mark.parametrize('source_depth,source_confidence,reference', [(False, False, False), (True, False, False), (False, True, False), (True, True, False), (False, False, True), (True, True, True)])
def test_stage_composition_matches_full_t0_oracles(b, n, h, w, source_depth, source_confidence, reference):
    from sintergpu.reprojection import temporal_reprojection_plan
    from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb, temporal_rgbd_reference
    rng = np.random.default_rng(81)
    k = np.tile(np.array([[4.3, 0, .31], [0, 3.7, .17], [0, 0, 1]], np.float32), (b, 1, 1))
    pose = np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))
    sk = np.repeat(k[:, None], n, axis=1); sp = np.repeat(pose[:, None], n, axis=1)
    angle = np.float32(.017)
    sp[..., :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    sp[..., :3, 3] = [.012, -.007, .021]
    td = rng.uniform(.9, 1.1, (b, 1, h, w)).astype(np.float32); td.reshape(-1)[0] = 0
    sd = rng.uniform(.9, 1.1, (b, n, 1, h, w)).astype(np.float32)
    rgb = rng.random((b, n, 3, h, w), dtype=np.float32); conf = rng.random(sd.shape, dtype=np.float32)
    inputs = dict(target_depth=td, source_rgb=rgb, target_intrinsics=k, target_pose_c2w=pose, source_intrinsics=sk, source_pose_c2w=sp)
    if source_depth or reference: inputs['source_depth'] = sd
    if source_confidence: inputs['source_confidence'] = conf
    graph = temporal_reprojection_plan(b, n, h, w, max_workgroups_x=65535, source_depth=source_depth, source_confidence=source_confidence, reference=reference)
    buffers, written = execute_graph_cpu(graph, inputs)
    expected = temporal_rgbd_reference(**inputs) if reference else temporal_reproject_rgb(target_rgb=np.zeros((b, 3, h, w), np.float32), **inputs)
    compare_graph_outputs(graph, buffers, expected)
    if reference: assert {'packTemporal', 'finalizeReference'} <= written


@pytest.mark.parametrize('dtype', [np.float64, np.int64, np.uint8])
def test_typed_capacity_validation_and_unknown_stage(dtype):
    bindings = {key: np.zeros(1, dtype=dtype) for key in ('minDepthBits', 'maxConfidenceBits', 'supportBits')}
    with pytest.raises(ValueError): run_stage('clearRaster', params(), bindings)
    with pytest.raises(ValueError): run_stage('notAStage', params(), {})


def test_project_partial_boundaries_are_per_pair_and_include_odd_tail():
    pairs, plane = 2, 257
    xyz = np.zeros((pairs, 3, 1, plane), np.float32)
    xyz[:, 0, 0] = np.arange(plane, dtype=np.float32)
    xyz[:, 2] = 1
    bindings = dict(cameraXYZ=xyz.reshape(-1), geometryValid=np.ones(pairs * plane, np.uint32),
                    sourcePose=np.tile(np.eye(4, dtype=np.float32), (pairs, 1, 1)).reshape(-1),
                    targetPose=np.tile(np.eye(4, dtype=np.float32), (pairs, 1, 1)).reshape(-1),
                    targetIntrinsics=np.tile(np.eye(3, dtype=np.float32), (pairs, 1, 1)).reshape(-1),
                    projected_xyOut=np.full(2 * pairs * plane + 1, 73, np.float32),
                    projected_depthOut=np.full(pairs * plane + 1, 73, np.float32),
                    visibleOut=np.full(pairs * plane + 1, 73, np.uint32),
                    displacementOut=np.full(pairs * plane + 1, 73, np.float32),
                    partialCount=np.full(pairs * 2 + 1, 73, np.uint32), partialParallax=np.full(pairs * 2 + 1, 73, np.float32))
    bindings['geometryValid'][plane:] = 0
    out = run_stage('projectPoints', params(pairs=pairs, width=plane, targetWidth=plane, groupsPerPair=2), bindings)
    np.testing.assert_array_equal(out['partialCount'], [256, 1, 0, 0, 73])
    np.testing.assert_array_equal(out['partialParallax'], [0, 0, 0, 0, 73])
    for value in out.values(): assert value[-1] == 73


def test_finalize_raster_consumes_existing_scratch_without_implicitly_clearing_it():
    bindings = dict(minDepthBits=f(1, np.inf).view(np.uint32), maxConfidenceBits=f(.9, .4).view(np.uint32), supportBits=u(3, 2),
                    depthOut=f(73, 73, 73), validOut=u(73, 73, 73), confidenceOut=f(73, 73, 73), support_countOut=u(73, 73, 73))
    out = run_stage('finalizeRaster', params(targetWidth=2), bindings)
    np.testing.assert_array_equal(out['depthOut'], [1, 0, 73])
    np.testing.assert_array_equal(out['validOut'], [1, 0, 73])
    np.testing.assert_array_equal(out['confidenceOut'], f(.9, .4, 73))
    np.testing.assert_array_equal(out['support_countOut'], [3, 2, 73])
    np.testing.assert_array_equal(bindings['supportBits'], [3, 2])


def test_reduction_uint_count_addition_retains_scalar32_wrap_semantics():
    bindings = dict(partialCount=u(0xffffffff, 2), partialParallax=f(0, 0), overlapOut=f(73), parallaxOut=f(73))
    out = run_stage('reduceProjection', params(width=257, groupsPerPair=2), bindings)
    np.testing.assert_array_equal(out['overlapOut'], f(1 / 257))


def test_stages_do_not_import_or_call_geometry_fusion_or_complete_pipeline():
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'src/sintergpu/references/reprojection_stage_numeric.py'
    tree = ast.parse(path.read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {'reprojection_numeric'}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {'depth_geometry', 'temporal_depth_fuse', 'reproject_rgbd', 'rasterize_reprojection', 'temporal_reproject_rgb', 'temporal_rgbd_reference'}


def test_short_capacity_and_noncontiguous_inputs_are_rejected():
    bindings = {key: u(0) for key in ('minDepthBits', 'maxConfidenceBits', 'supportBits')}
    with pytest.raises(ValueError): run_stage('clearRaster', params(targetWidth=2), bindings)
    bindings = {key: np.zeros(4, np.uint32)[::2] for key in bindings}
    with pytest.raises(ValueError): run_stage('clearRaster', params(targetWidth=2), bindings)


def test_project_repeated_geometry_cameras_and_shared_target_depth_batches():
    pairs = 6
    bindings = dict(cameraXYZ=f(0, 0, 1, 0, 0, 1), geometryValid=u(1, 1),
                    sourcePose=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)).reshape(-1),
                    targetPose=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)).reshape(-1),
                    targetIntrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)).reshape(-1), targetDepth=f(2, 3))
    for key in ('projected_xyOut', 'projected_depthOut', 'displacementOut', 'partialParallax', 'depth_consistencyOut'):
        bindings[key] = np.full(pairs * (2 if key == 'projected_xyOut' else 1) + 1, 73, np.float32)
    for key in ('visibleOut', 'partialCount', 'depth_consistency_validOut', 'occludedOut', 'occlusion_validOut'):
        bindings[key] = np.full(pairs + 1, 73, np.uint32)
    p = params(pairs=pairs, batches=2, neighbors=3, geometryRepeat=3, sourceCameraRepeat=3,
               targetCameraRepeat=3, targetDepthRepeat=3, useTargetDepth=1)
    out = run_stage('projectPoints', p, bindings)
    np.testing.assert_array_equal(out['depth_consistencyOut'], [1, 1, 1, 2, 2, 2, 73])
    np.testing.assert_array_equal(out['partialCount'], [1, 1, 1, 1, 1, 1, 73])
