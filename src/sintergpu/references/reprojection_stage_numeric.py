# SPDX-License-Identifier: Apache-2.0
"""Independent CPU float32 oracles for the nine D4/T0 dispatch boundaries.

run_stage mutates only the caller-preallocated written/inout buffers and returns
those SAME flat arrays. Readonly inputs and inactive capacity tails are preserved.
The caller snapshots/resets persistent state outside the timed interval. No stage
runs backprojection, D9 fusion or a complete public D4/T0 pipeline on its behalf.
Sampler work uses the shared independent CPU sampling contract. CPU allocation
of its temporary sampled pixels is part of this explicitly NumPy reference cost.
"""
import numpy as np

from .reprojection_numeric import sample_pixels

_F = np.float32
_U = np.uint32
_EPS = np.finfo(np.float32).eps


def _buffer(bindings, name, count, dtype=np.float32):
    value = bindings[name]
    if (not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype
            or not value.flags.c_contiguous or value.size < count):
        raise ValueError(name + ' must be a contiguous flat ' + np.dtype(dtype).name + ' array with sufficient capacity')
    return value


def _outputs(bindings, sizes, *, uints=()):
    return {name: _buffer(bindings, name, size, np.uint32 if name in uints else np.float32)
            for name, size in sizes.items()}


def _dot(row, xyz):
    return _F(_F(_F(row[0] * xyz[0]) + _F(row[1] * xyz[1])) + _F(row[2] * xyz[2]))


def _project(p, b):
    pairs, h, w = p['pairs'], p['height'], p['width']
    th, tw, groups = p['targetHeight'], p['targetWidth'], p['groupsPerPair']
    plane = h * w; count = pairs * plane
    if groups != (plane + 255) // 256:
        raise ValueError('groupsPerPair must describe every padded 256-pixel partial')
    geometry_count = (pairs + p['geometryRepeat'] - 1) // p['geometryRepeat']
    source_count = (pairs + p['sourceCameraRepeat'] - 1) // p['sourceCameraRepeat']
    target_count = (pairs + p['targetCameraRepeat'] - 1) // p['targetCameraRepeat']
    xyz = _buffer(b, 'cameraXYZ', 3 * geometry_count * plane)
    valid = _buffer(b, 'geometryValid', geometry_count * plane, np.uint32)
    sp = _buffer(b, 'sourcePose', 16 * source_count)
    tp = _buffer(b, 'targetPose', 16 * target_count)
    tk = _buffer(b, 'targetIntrinsics', 9 * target_count)
    sizes = dict(projected_xyOut=2 * count, projected_depthOut=count, visibleOut=count,
                 displacementOut=count, partialCount=pairs * groups, partialParallax=pairs * groups)
    uints = ['visibleOut', 'partialCount']
    if p['useTargetDepth']:
        sizes.update(depth_consistencyOut=count, depth_consistency_validOut=count, occludedOut=count, occlusion_validOut=count)
        uints += ['depth_consistency_validOut', 'occludedOut', 'occlusion_validOut']
        depth_count = (pairs + p['targetDepthRepeat'] - 1) // p['targetDepthRepeat']
        depth = _buffer(b, 'targetDepth', depth_count * th * tw)
    out = _outputs(b, sizes, uints=uints)
    for pair in range(pairs):
        g = pair // p['geometryRepeat']; source = pair // p['sourceCameraRepeat']; target = pair // p['targetCameraRepeat']
        source_pose = sp[16 * source:16 * (source + 1)].reshape(4, 4)
        target_pose = tp[16 * target:16 * (target + 1)].reshape(4, 4)
        intrinsics = tk[9 * target:9 * (target + 1)].reshape(3, 3)
        for group in range(groups):
            partial_count = 0; partial_parallax = _F(0)
            for q in range(256 * group, min(256 * (group + 1), plane)):
                camera = xyz[g * 3 * plane + q:g * 3 * plane + q + 3 * plane:plane]
                world = np.array([_F(_dot(source_pose[i, :3], camera) + source_pose[i, 3]) for i in range(3)], np.float32)
                offset = world - target_pose[:3, 3]
                rel = np.array([_dot(target_pose[:3, i], offset) for i in range(3)], np.float32)
                denominator = np.maximum(rel[2], _EPS)
                u = _F(_F(_F(intrinsics[0, 0] * rel[0]) / denominator) + intrinsics[0, 2])
                v = _F(_F(_F(intrinsics[1, 1] * rel[1]) / denominator) + intrinsics[1, 2])
                visible = bool(valid[g * plane + q] and rel[2] > 0 and 0 <= u <= tw - 1 and 0 <= v <= th - 1)
                i = pair * plane + q
                dx, dy = _F(u - _F(q % w)), _F(v - _F(q // w))
                displacement = np.sqrt(_F(_F(dx * dx) + _F(dy * dy)))
                out['projected_xyOut'][pair * 2 * plane + q] = u
                out['projected_xyOut'][(pair * 2 + 1) * plane + q] = v
                out['projected_depthOut'][i] = rel[2]
                out['visibleOut'][i] = visible
                out['displacementOut'][i] = displacement
                partial_count += visible
                partial_parallax = _F(partial_parallax + _F(displacement * _F(visible)))
            out['partialCount'][pair * groups + group] = partial_count
            out['partialParallax'][pair * groups + group] = partial_parallax
        if p['useTargetDepth']:
            d = pair // p['targetDepthRepeat']
            field = depth[d * th * tw:(d + 1) * th * tw].reshape(1, 1, th, tw)
            xy = out['projected_xyOut'][pair * 2 * plane:(pair + 1) * 2 * plane].reshape(1, 2, h, w)
            sampled = sample_pixels(field, xy).reshape(-1)
            for q in range(plane):
                i = pair * plane + q; z = out['projected_depthOut'][i]
                usable = bool(out['visibleOut'][i] and np.isfinite(sampled[q]) and sampled[q] > 0)
                out['depth_consistencyOut'][i] = abs(_F(sampled[q] - z)) if usable else _F(0)
                out['depth_consistency_validOut'][i] = usable
                out['occlusion_validOut'][i] = usable
                out['occludedOut'][i] = usable and sampled[q] < _F(z - _F(p['threshold']))
    return out


def _reduce(p, b):
    pairs, groups = p['pairs'], p['groupsPerPair']
    count = _buffer(b, 'partialCount', pairs * groups, np.uint32)
    parallax = _buffer(b, 'partialParallax', pairs * groups)
    out = _outputs(b, dict(overlapOut=pairs, parallaxOut=pairs))
    for pair in range(pairs):
        start = pair * groups; stop = start + groups
        total = int(count[start:stop].sum(dtype=np.uint64)) & 0xffffffff
        out['overlapOut'][pair] = _F(total) / _F(p['height'] * p['width'])
        out['parallaxOut'][pair] = parallax[start:stop].sum(dtype=np.float32) / _F(max(total, 1))
    return out


def _clear(p, b):
    count = p['pairs'] * p['targetHeight'] * p['targetWidth']
    sizes = {name: count for name in ('minDepthBits', 'maxConfidenceBits', 'supportBits')}
    out = _outputs(b, sizes, uints=sizes)
    out['minDepthBits'][:count] = 0x7f800000
    out['maxConfidenceBits'][:count] = 0
    out['supportBits'][:count] = 0
    return out


def _contributors(p, b):
    plane = p['height'] * p['width']; count = p['pairs'] * plane
    xy = _buffer(b, 'projected_xyOut', 2 * count)
    z = _buffer(b, 'projected_depthOut', count)
    visible = _buffer(b, 'visibleOut', count, np.uint32)
    th, tw = p['targetHeight'], p['targetWidth']
    for i in range(count):
        if not visible[i]:
            continue
        pair, q = divmod(i, plane)
        u, v = xy[pair * 2 * plane + q], xy[(pair * 2 + 1) * plane + q]
        x = min(max(int(np.rint(u)), 0), tw - 1)
        y = min(max(int(np.rint(v)), 0), th - 1)
        yield i, pair * th * tw + y * tw + x, z[i]


def _raster_min(p, b):
    count = p['pairs'] * p['targetHeight'] * p['targetWidth']
    out = _outputs(b, dict(minDepthBits=count), uints=('minDepthBits',))
    for _, destination, z in _contributors(p, b):
        bits = z.view(np.uint32)
        out['minDepthBits'][destination] = min(out['minDepthBits'][destination], bits)
    return out


def _raster_support(p, b):
    count = p['pairs'] * p['targetHeight'] * p['targetWidth']
    minimum = _buffer(b, 'minDepthBits', count, np.uint32).view(np.float32)
    out = _outputs(b, dict(maxConfidenceBits=count, supportBits=count), uints=('maxConfidenceBits', 'supportBits'))
    if p['useConfidence']:
        confidence = _buffer(b, 'sourceConfidence', p['pairs'] * p['height'] * p['width'])
    for i, destination, z in _contributors(p, b):
        if abs(_F(z - minimum[destination])) <= _F(_EPS * max(abs(minimum[destination]), _F(1))):
            value = confidence[i] if p['useConfidence'] else _F(1)
            bits = _U(0) if value == 0 else value.view(np.uint32)
            out['maxConfidenceBits'][destination] = max(out['maxConfidenceBits'][destination], bits)
            out['supportBits'][destination] = _U((int(out['supportBits'][destination]) + 1) & 0xffffffff)
    return out


def _finalize_raster(p, b):
    count = p['pairs'] * p['targetHeight'] * p['targetWidth']
    depth = _buffer(b, 'minDepthBits', count, np.uint32).view(np.float32)
    confidence = _buffer(b, 'maxConfidenceBits', count, np.uint32).view(np.float32)
    support = _buffer(b, 'supportBits', count, np.uint32)
    out = _outputs(b, dict(depthOut=count, validOut=count, confidenceOut=count, support_countOut=count), uints=('validOut', 'support_countOut'))
    for i in range(count):
        valid = np.isfinite(depth[i])
        out['depthOut'][i] = depth[i] if valid else _F(0)
        out['validOut'][i] = valid
        out['confidenceOut'][i] = confidence[i]
        out['support_countOut'][i] = support[i]
    return out


def _sample(p, b):
    pairs, h, w = p['pairs'], p['height'], p['width']; plane = h * w; count = pairs * plane
    th, tw = p['targetHeight'], p['targetWidth']
    xy = _buffer(b, 'projected_xyOut', 2 * count)[:2 * count].reshape(pairs, 2, h, w)
    rgb = _buffer(b, 'sourceRGB', 3 * count)[:3 * count].reshape(pairs, 3, th, tw)
    visible = _buffer(b, 'visibleOut', count, np.uint32)
    out = _outputs(b, dict(aligned_rgbOut=3 * count, aligned_validOut=count, alignment_confidenceOut=count), uints=('aligned_validOut',))
    sampled = sample_pixels(rgb, xy, mode='bilinear').reshape(-1)
    out['aligned_rgbOut'][:3 * count] = sampled
    if p['useTargetDepth']:
        consistency = _buffer(b, 'depth_consistencyOut', count)
        consistency_valid = _buffer(b, 'depth_consistency_validOut', count, np.uint32)
        occluded = _buffer(b, 'occludedOut', count, np.uint32)
    if p['useConfidence']:
        confidence_field = _buffer(b, 'sourceConfidence', count)[:count].reshape(pairs, 1, th, tw)
        sampled_confidence = sample_pixels(confidence_field, xy, mode='bilinear').reshape(-1)
    for i in range(count):
        valid = bool(visible[i]); confidence = _F(valid)
        if p['useTargetDepth']:
            consistent = bool(consistency_valid[i] and not occluded[i])
            valid &= consistent
            confidence = _F(_F(confidence * np.exp(_F(-consistency[i] / _F(p['threshold'])))) * _F(consistent))
        if p['useConfidence']:
            confidence = _F(confidence * sampled_confidence[i])
        out['aligned_validOut'][i] = valid
        out['alignment_confidenceOut'][i] = confidence
    return out


def _pack(p, b):
    batches, neighbors, plane = p['batches'], p['neighbors'], p['height'] * p['width']
    count = batches * plane; prior_count = neighbors * count; packed_count = (neighbors + 1) * count
    depth = _buffer(b, 'depthOut', prior_count)
    valid = _buffer(b, 'validOut', prior_count, np.uint32)
    confidence = _buffer(b, 'confidenceOut', prior_count)
    out = _outputs(b, dict(packedDepth=packed_count, packedValid=packed_count, packedConfidence=packed_count, packedTrusted=packed_count), uints=('packedValid', 'packedTrusted'))
    for source in range(neighbors + 1):
        for i in range(count):
            destination = source * count + i
            if source == 0:
                out['packedDepth'][destination] = 0
                out['packedConfidence'][destination] = 0
                out['packedValid'][destination] = 0
            else:
                batch, q = divmod(i, plane)
                prior = (batch * neighbors + source - 1) * plane + q
                out['packedDepth'][destination] = depth[prior]
                out['packedConfidence'][destination] = confidence[prior]
                out['packedValid'][destination] = valid[prior]
            out['packedTrusted'][destination] = 0
    return out


def _finalize_reference(p, b):
    batches, neighbors, plane = p['batches'], p['neighbors'], p['height'] * p['width']
    count = batches * plane; prior_count = neighbors * count
    rgb = _buffer(b, 'aligned_rgbOut', 3 * prior_count)
    aligned_valid = _buffer(b, 'aligned_validOut', prior_count, np.uint32)
    aligned_confidence = _buffer(b, 'alignment_confidenceOut', prior_count)
    raster_confidence = _buffer(b, 'rasterConfidence', prior_count)
    raster_valid = _buffer(b, 'rasterValid', prior_count, np.uint32)
    fused_depth = _buffer(b, 'fusedDepth', count)
    fused_valid = _buffer(b, 'fusedValid', count, np.uint32)
    fused_confidence = _buffer(b, 'fusedConfidence', count)
    out = _outputs(b, dict(rgbOut=3 * count, depthOut=count, validOut=count, confidenceOut=count), uints=('validOut',))
    for i in range(count):
        batch, q = divmod(i, plane); total = _F(0); numerator = np.zeros(3, np.float32)
        for neighbor in range(neighbors):
            pair = batch * neighbors + neighbor; j = pair * plane + q
            weight = _F(aligned_confidence[j] * raster_confidence[j])
            weight = _F(_F(weight * _F(raster_valid[j] != 0)) * _F(aligned_valid[j] != 0))
            total = _F(total + weight)
            for channel in range(3):
                numerator[channel] = _F(numerator[channel] + _F(rgb[(pair * 3 + channel) * plane + q] * weight))
        valid = bool(fused_valid[i] and total > 0)
        out['validOut'][i] = valid
        out['depthOut'][i] = fused_depth[i] if valid else _F(0)
        out['confidenceOut'][i] = _F(fused_confidence[i] * min(total, _F(1))) if valid else _F(0)
        for channel in range(3):
            out['rgbOut'][(batch * 3 + channel) * plane + q] = _F(numerator[channel] / max(total, _EPS)) if valid else _F(0)
    return out


_STAGES = dict(projectPoints=_project, reduceProjection=_reduce, clearRaster=_clear,
               rasterMin=_raster_min, rasterSupport=_raster_support,
               finalizeRaster=_finalize_raster, sampleTemporal=_sample,
               packTemporal=_pack, finalizeReference=_finalize_reference)


def run_stage(entrypoint, params, bindings):
    """Execute exactly one independent dispatch oracle on prepared flat buffers.

    params follows ReprojectionParams; bindings uses the shader binding names,
    including caller-owned writable arrays. Returned values alias those arrays.
    Masks/counts retain the GPU uint32 ABI; no bool/int64 byte reinterpretation.
    """
    if entrypoint not in _STAGES:
        raise ValueError('unknown reprojection stage: ' + str(entrypoint))
    with np.errstate(all='ignore'):
        return _STAGES[entrypoint](params, bindings)
