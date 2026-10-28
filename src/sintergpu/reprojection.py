# SPDX-License-Identifier: Apache-2.0
"""Setup-only caller-owned D4/T0 resource graphs; no GPU imports or execution.

All sizes are scalar32 elements. Logical bool is uint32, D4 support uint32 and
D9 signed counts int32 (bounded below INT_MAX by the RGB scalar capacity bound).
Stages require storage barriers on the named resources before recording. Resources
must not alias except the explicitly repeated read bindings in D9. Cameras/depth
contents follow the independent float32 reference profile and are caller-validated.
"""
import math
import numbers
import struct

from .vkl import geometry_layout, fusion_layout, _integer, UINT_MAX, INT_MAX


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError('consistency_threshold must be positive finite float32')
    try:
        rounded = struct.unpack('f', struct.pack('f', float(value)))[0]
    except (OverflowError, struct.error):
        raise ValueError('consistency_threshold exceeds float32') from None
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError('consistency_threshold must be positive finite float32')
    return rounded


class _Graph:
    def __init__(self, limit):
        self.resources = {}
        self.stages = []
        self.limit = limit

    def resource(self, name, elements, dtype='float32', role='scratch'):
        if elements > UINT_MAX:
            raise ValueError(name + ' scalar32 offsets exceed uint32')
        if name in self.resources:
            if self.resources[name] != dict(elements=elements, dtype=dtype, role=role):
                raise ValueError('incompatible resource reuse: ' + name)
        else:
            self.resources[name] = dict(elements=elements, dtype=dtype, role=role)
        return name

    def stage(self, entry, count, params, bindings, barriers=(), module='reprojection'):
        if (count + 255) // 256 > self.limit or count > UINT_MAX:
            raise ValueError(entry + ' exceeds device X dispatch or uint32 thread count')
        if count:
            self.stages.append(dict(operation='vkl.' + module + '.' + entry,
                                    entrypoint=entry, thread_count=(count, 1, 1),
                                    params=params.copy(), bindings=bindings,
                                    barriers=list(dict.fromkeys(barriers))))

    def finish(self, outputs):
        for resource in outputs.values():
            self.resources[resource]['role'] = 'output'
        return dict(resources=self.resources, stages=self.stages, outputs=outputs)


def _validate(batches, height, width, th, tw, limit):
    _integer('batches', batches)
    _integer('max_workgroups_x', limit, 1)
    for name, value in [('height', height), ('width', width), ('target_height', th), ('target_width', tw)]:
        _integer(name, value, 1, 2**24)
    if height * width > UINT_MAX or th * tw > UINT_MAX:
        raise ValueError('image area exceeds uint32 indexing')
    if 3 * batches * height * width > UINT_MAX or 3 * batches * th * tw > UINT_MAX or 16 * batches > UINT_MAX:
        raise ValueError('camera/RGB scalar32 offsets overflow')


def _geometry(g, batches, h, w, depth, intrinsics, prefix):
    layout = geometry_layout(batches, h, w, max_workgroups_x=g.limit)
    count = batches * h * w
    outputs = dict(validOut=g.resource(prefix + 'valid', count, 'uint32'),
                   rays_cameraOut=g.resource(prefix + 'rays', 3 * count),
                   xyz_cameraOut=g.resource(prefix + 'xyz', 3 * count),
                   normalsOut=g.resource(prefix + 'normals', 3 * count),
                   geometry_confidenceOut=g.resource(prefix + 'geometry_confidence', count),
                   xyz_worldOut=g.resource('dummy_float', 1))
    bindings = dict(depth=depth, intrinsics=intrinsics,
                    pose=g.resource('dummy_float', 1), confidence='dummy_float', **outputs)
    g.stage('depthGeometry', count, layout['params'], bindings, tuple(outputs.values()), module='geometry')
    return outputs['xyz_cameraOut'], outputs['validOut']


def _project(g, pairs, h, w, th, tw, xyz, geometry_valid, source_pose,
             target_k, target_pose, *, prefix='', geometry_repeat=1,
             source_repeat=1, target_repeat=1, depth_repeat=1,
             target_depth=None, threshold=.05, neighbors=1, batches=None):
    plane = h * w
    count = pairs * plane
    groups = (plane + 255) // 256
    if 3 * count > UINT_MAX or pairs * groups * 256 > UINT_MAX:
        raise ValueError('projection RGB or padded partial offsets overflow')
    params = dict(pairs=pairs, height=h, width=w, targetHeight=th, targetWidth=tw,
                  geometryRepeat=geometry_repeat, sourceCameraRepeat=source_repeat,
                  targetCameraRepeat=target_repeat, targetDepthRepeat=depth_repeat,
                  useTargetDepth=int(target_depth is not None), useConfidence=0,
                  groupsPerPair=groups, neighbors=neighbors,
                  batches=pairs if batches is None else batches, threshold=threshold)
    counts = dict(projected_xy=2 * count, projected_depth=count, visible=count,
                  displacement=count, overlap=pairs, parallax=pairs,
                  depth_consistency=count, depth_consistency_valid=count,
                  occluded=count, occlusion_valid=count)
    outputs = {}
    for key, size in counts.items():
        if key.startswith('depth_consistency') or key in ('occluded', 'occlusion_valid'):
            if target_depth is None:
                continue
        dtype = 'uint32' if key in ('visible', 'depth_consistency_valid', 'occluded', 'occlusion_valid') else 'float32'
        outputs[key] = g.resource(prefix + key, size, dtype)
    partial_count = g.resource(prefix + 'partial_count', pairs * groups, 'uint32')
    partial_parallax = g.resource(prefix + 'partial_parallax', pairs * groups)
    bindings = dict(cameraXYZ=xyz, geometryValid=geometry_valid, sourcePose=source_pose,
                    targetIntrinsics=target_k, targetPose=target_pose,
                    targetDepth=target_depth or g.resource('dummy_float', 1),
                    partialCount=partial_count, partialParallax=partial_parallax)
    bindings.update({key + 'Out': name for key, name in outputs.items() if key not in ('overlap', 'parallax')})
    for key in ('depth_consistency', 'depth_consistency_valid', 'occluded', 'occlusion_valid'):
        if key not in outputs:
            dtype = 'float32' if key == 'depth_consistency' else 'uint32'
            bindings[key + 'Out'] = g.resource('dummy_float' if dtype == 'float32' else 'dummy_uint', 1, dtype)
    g.stage('projectPoints', pairs * groups * 256, params, bindings, (xyz, geometry_valid))
    g.stage('reduceProjection', pairs * 256, params,
            dict(partialCount=partial_count, partialParallax=partial_parallax,
                 overlapOut=outputs['overlap'], parallaxOut=outputs['parallax']),
            (partial_count, partial_parallax))
    return params, outputs


def _raster(g, params, projection, confidence, prefix=''):
    pixels = params['pairs'] * params['targetHeight'] * params['targetWidth']
    source_pixels = params['pairs'] * params['height'] * params['width']
    scratch = {name: g.resource(prefix + name, pixels, 'uint32') for name in ('raster_min', 'raster_confidence', 'raster_support')}
    bindings = dict(minDepthBits=scratch['raster_min'], maxConfidenceBits=scratch['raster_confidence'], supportBits=scratch['raster_support'])
    g.stage('clearRaster', pixels, params, bindings, tuple(scratch.values()))
    source = dict(projected_xyOut=projection['projected_xy'], projected_depthOut=projection['projected_depth'],
                  visibleOut=projection['visible'], sourceConfidence=confidence)
    g.stage('rasterMin', source_pixels, params, dict(**bindings, **source), tuple(scratch.values()) + tuple(projection.values()))
    g.stage('rasterSupport', source_pixels, params, dict(**bindings, **source), (scratch['raster_min'],))
    outputs = {key: g.resource(prefix + key, pixels, 'uint32' if key in ('valid', 'support_count') else 'float32') for key in ('depth', 'valid', 'confidence', 'support_count')}
    g.stage('finalizeRaster', pixels, params,
            dict(**bindings, **{key + 'Out': value for key, value in outputs.items()}), tuple(scratch.values()))
    return outputs


def reprojection_plan(batches, height, width, target_height, target_width, *,
                      max_workgroups_x, target_depth=False, raster=False,
                      consistency_threshold=.05):
    """D4 projection graph, or the four frozen raster outputs when raster=True."""
    _validate(batches, height, width, target_height, target_width, max_workgroups_x)
    if type(target_depth) is not bool or type(raster) is not bool:
        raise ValueError('target_depth/raster must be booleans')
    threshold = _threshold(consistency_threshold)
    g = _Graph(max_workgroups_x)
    count = batches * height * width
    depth = g.resource('source_depth', count, role='input')
    sk = g.resource('source_intrinsics', 9 * batches, role='input')
    sp = g.resource('source_pose_c2w', 16 * batches, role='input')
    tk = g.resource('target_intrinsics', 9 * batches, role='input')
    tp = g.resource('target_pose_c2w', 16 * batches, role='input')
    td = g.resource('target_depth', batches * target_height * target_width, role='input') if target_depth else None
    xyz, valid = _geometry(g, batches, height, width, depth, sk, 'camera_')
    params, outputs = _project(g, batches, height, width, target_height, target_width,
                               xyz, valid, sp, tk, tp, target_depth=td, threshold=threshold)
    if raster:
        params['useConfidence'] = 1
        confidence = g.resource('source_confidence', count, role='input')
        outputs = _raster(g, params, outputs, confidence)
    return g.finish(outputs)


def temporal_reprojection_plan(batches, neighbors, height, width, *,
                               max_workgroups_x, source_depth=False,
                               source_confidence=False, reference=False,
                               consistency_threshold=.05, min_support=1):
    """T0 graph; reference=True requires prior depth regardless of source_depth."""
    _validate(batches, height, width, height, width, max_workgroups_x)
    _integer('neighbors', neighbors, 1, INT_MAX - 1)
    _integer('min_support', min_support, 1, INT_MAX)
    if any(type(value) is not bool for value in (source_depth, source_confidence, reference)):
        raise ValueError('source_depth/source_confidence/reference must be booleans')
    threshold = _threshold(consistency_threshold)
    pairs = batches * neighbors
    count, pair_count = batches * height * width, pairs * height * width
    if 3 * pair_count > UINT_MAX or 16 * pairs > UINT_MAX or (neighbors + 1) * count > UINT_MAX:
        raise ValueError('temporal camera/RGB/packed scalar32 offsets overflow')
    g = _Graph(max_workgroups_x)
    td = g.resource('target_depth', count, role='input')
    rgb = g.resource('source_rgb', 3 * pair_count, role='input')
    tk = g.resource('target_intrinsics', 9 * batches, role='input')
    tp = g.resource('target_pose_c2w', 16 * batches, role='input')
    sk = g.resource('source_intrinsics', 9 * pairs, role='input')
    sp = g.resource('source_pose_c2w', 16 * pairs, role='input')
    sd = g.resource('source_depth', pair_count, role='input') if source_depth or reference else None
    sc = g.resource('source_confidence', pair_count, role='input') if source_confidence else None
    xyz, valid = _geometry(g, batches, height, width, td, tk, 'target_camera_')
    params, projected = _project(g, pairs, height, width, height, width, xyz, valid, tp, sk, sp,
                                 prefix='warp_', geometry_repeat=neighbors,
                                 source_repeat=neighbors, target_depth=sd, threshold=threshold,
                                 neighbors=neighbors, batches=batches)
    params['useConfidence'] = int(sc is not None)
    aligned = {key: g.resource(key, 3 * pair_count if key == 'aligned_rgb' else pair_count,
                              'uint32' if key == 'aligned_valid' else 'float32')
               for key in ('aligned_rgb', 'aligned_valid', 'alignment_confidence')}
    sample_bindings = dict(sourceRGB=rgb, sourceConfidence=sc or 'dummy_float',
                           projected_xyOut=projected['projected_xy'], visibleOut=projected['visible'],
                           **{key + 'Out': value for key, value in aligned.items()})
    for key in ('depth_consistency', 'depth_consistency_valid', 'occluded'):
        sample_bindings[key + 'Out'] = projected.get(key, 'dummy_float' if key == 'depth_consistency' else 'dummy_uint')
    g.stage('sampleTemporal', pair_count, params, sample_bindings, tuple(projected.values()))
    if not reference:
        return g.finish(aligned)
    xyz, valid = _geometry(g, pairs, height, width, sd, sk, 'prior_camera_')
    prior_params, prior_projected = _project(g, pairs, height, width, height, width, xyz, valid, sp, tk, tp,
                                             prefix='prior_', target_repeat=neighbors,
                                             threshold=threshold, neighbors=neighbors, batches=batches)
    prior_params['useConfidence'] = int(sc is not None)
    raster = _raster(g, prior_params, prior_projected, sc or 'dummy_float', prefix='prior_raster_')
    packed = {key: g.resource('packed_' + key, (neighbors + 1) * count,
                              'uint32' if key in ('valid', 'trusted') else 'float32') for key in ('depth', 'valid', 'confidence', 'trusted')}
    g.stage('packTemporal', (neighbors + 1) * count, params,
            dict(depthOut=raster['depth'], validOut=raster['valid'], confidenceOut=raster['confidence'],
                 packedDepth=packed['depth'], packedValid=packed['valid'], packedConfidence=packed['confidence'], packedTrusted=packed['trusted']), tuple(raster.values()))
    fusion = fusion_layout(batches, height, width, neighbors + 1, max_workgroups_x=max_workgroups_x,
                           temporal=True, threshold=threshold, min_support=min_support, min_temporal_support=1)
    fusion_bindings = dict(depths=packed['depth'], valids=packed['valid'], confidences=packed['confidence'],
                           visibility=packed['valid'], consistency=packed['valid'], trusted=packed['trusted'])
    # Aliasing read-only evidence with valid is equivalent: the temporal usable
    # predicate is valid & visible & consistent; every prior raster-valid is true
    # in all three, and every invalid prior is already excluded by valid.
    for name, size in fusion['elements'].items():
        if name.endswith('Out'):
            dtype = 'uint32' if name == 'validOut' else 'int32' if any(token in name for token in ('count', 'source', 'contributor')) and name != 'source_disagreementOut' else 'float32'
            fusion_bindings[name] = g.resource('fused_' + name[:-3], size, dtype)
    g.stage('temporalDepthFuse', count, fusion['params'], fusion_bindings, tuple(packed.values()), module='depth_fusion')
    outputs = {key: g.resource(key, 3 * count if key == 'rgb' else count,
                               'uint32' if key == 'valid' else 'float32') for key in ('rgb', 'depth', 'valid', 'confidence')}
    outputs.update({key: fusion_bindings[key + 'Out'] for key in ('support_count', 'temporal_support_count', 'source_disagreement')})
    g.stage('finalizeReference', count, params,
            dict(aligned_rgbOut=aligned['aligned_rgb'], aligned_validOut=aligned['aligned_valid'],
                 alignment_confidenceOut=aligned['alignment_confidence'], rasterConfidence=raster['confidence'], rasterValid=raster['valid'],
                 fusedDepth=fusion_bindings['depthOut'], fusedValid=fusion_bindings['validOut'], fusedConfidence=fusion_bindings['confidenceOut'],
                 rgbOut=outputs['rgb'], depthOut=outputs['depth'], validOut=outputs['valid'], confidenceOut=outputs['confidence']),
            tuple(aligned.values()) + tuple(raster.values()) + tuple(fusion_bindings.values()))
    return g.finish(outputs)
