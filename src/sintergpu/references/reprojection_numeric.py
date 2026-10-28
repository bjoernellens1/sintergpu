# SPDX-License-Identifier: Apache-2.0
"""Independent scalar NumPy D4 oracle, using the frozen CPU float32 profile.

Pixels pass through the normalized float32 grid coordinate roundtrip used by
align_corners=True sampling. Nearest ties round to even; padding is zero.
The characterized CPU profile returns zero for nonfinite nearest coordinates
and NaN for nonfinite bilinear coordinates. No Torch code is imported here.
"""
import math

import numpy as np

_F = np.float32
_EPS = np.finfo(np.float32).eps


def _array(value):
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError('the numerical reference profile requires float32 arrays')
    return array


def _sample_coordinate(pixel, size):
    # Preserve each rounding boundary, including singleton axes: multiplying a
    # finite normalized coordinate by zero yields coordinate zero on that axis.
    with np.errstate(over='ignore', invalid='ignore'):
        normalized = _F(_F(_F(2) * pixel) / _F(max(size - 1, 1)) - _F(1))
        return _F(_F(_F(normalized + _F(1)) / _F(2)) * _F(size - 1))


def sample_pixels(field, projected_xy, *, mode='nearest'):
    """Sample [B,C,H,W] at [B,2,OH,OW], with frozen grid_sample semantics."""
    field = _array(field)
    projected_xy = _array(projected_xy)
    if (field.ndim != 4 or projected_xy.ndim != 4 or projected_xy.shape[1] != 2
            or projected_xy.shape[0] != field.shape[0] or min(field.shape[-2:]) < 1):
        raise ValueError('field [B,C,H,W] and projected_xy [B,2,OH,OW] must match')
    if mode not in ('nearest', 'bilinear'):
        raise ValueError('mode must be nearest or bilinear')
    batches, channels, h, w = field.shape
    oh, ow = projected_xy.shape[-2:]
    result = np.zeros((batches, channels, oh, ow), np.float32)
    for b in range(batches):
        for y in range(oh):
            for x in range(ow):
                u = _sample_coordinate(projected_xy[b, 0, y, x], w)
                v = _sample_coordinate(projected_xy[b, 1, y, x], h)
                if not (np.isfinite(u) and np.isfinite(v)):
                    if mode == 'bilinear':
                        result[b, :, y, x] = np.nan
                    continue
                if mode == 'nearest':
                    xx, yy = int(np.rint(u)), int(np.rint(v))
                    if 0 <= xx < w and 0 <= yy < h:
                        result[b, :, y, x] = field[b, :, yy, xx]
                    continue
                left, top = math.floor(u), math.floor(v)
                # Bilinear weights retain the float32 arithmetic, and the four
                # additions follow image row order. Out-of-bounds taps vanish.
                for yy in (top, top + 1):
                    wy = _F(_F(top + 1) - v) if yy == top else _F(v - _F(top))
                    for xx in (left, left + 1):
                        wx = _F(_F(left + 1) - u) if xx == left else _F(u - _F(left))
                        if 0 <= xx < w and 0 <= yy < h:
                            for c in range(channels):
                                term = _F(_F(field[b, c, yy, xx] * wx) * wy)
                                result[b, c, y, x] = _F(result[b, c, y, x] + term)
    return result


def _dot3(row, xyz):
    return _F(_F(_F(row[0] * xyz[0]) + _F(row[1] * xyz[1])) + _F(row[2] * xyz[2]))


def reproject_rgbd(source_depth, source_intrinsics, source_pose_c2w,
                   target_intrinsics, target_pose_c2w, target_depth=None, *,
                   target_hw=None, consistency_threshold=0.05):
    """Project depth, preserving raw diagnostics even for invalid source depth."""
    depth, sk, sp, tk, tp = map(_array, (source_depth, source_intrinsics,
                                      source_pose_c2w, target_intrinsics, target_pose_c2w))
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError('source_depth must have shape [B,1,H,W]')
    b, _, h, w = depth.shape
    if sk.shape != (b, 3, 3) or tk.shape != sk.shape or sp.shape != (b, 4, 4) or tp.shape != sp.shape:
        raise ValueError('source and target geometry must have matching batch shapes')
    if not np.isfinite(sk).all() or np.any(sk[:, (0, 1), (0, 1)] <= 0):
        raise ValueError('source intrinsics must be finite with positive focal lengths')
    if consistency_threshold <= 0:
        raise ValueError('consistency_threshold must be positive')
    if target_hw is None and target_depth is None:
        raise ValueError('target_hw is required when target_depth is absent')
    th, tw = target_hw or (h, w)
    if target_depth is not None:
        target_depth = _array(target_depth)
        if target_depth.ndim != 4 or target_depth.shape[:2] != (b, 1):
            raise ValueError('target_depth must have shape [B,1,H,W]')
        if target_hw is not None and tuple(target_hw) != target_depth.shape[-2:]:
            raise ValueError('target_hw must match target_depth spatial shape')
        th, tw = target_depth.shape[-2:]
    xy = np.empty((b, 2, h, w), np.float32)
    raw_z = np.empty_like(depth)
    visible = np.zeros(depth.shape, bool)
    displacement = np.empty_like(depth)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        for batch in range(b):
            for y in range(h):
                for x in range(w):
                    d = depth[batch, 0, y, x]
                    valid = np.isfinite(d) and d > 0
                    d = d if valid else _F(0)
                    camera = np.array([
                        _F(_F(_F(x) - sk[batch, 0, 2]) / sk[batch, 0, 0]) * d,
                        _F(_F(_F(y) - sk[batch, 1, 2]) / sk[batch, 1, 1]) * d,
                        d], np.float32)
                    world = np.array([_F(_dot3(sp[batch, i, :3], camera) + sp[batch, i, 3]) for i in range(3)], np.float32)
                    offset = world - tp[batch, :3, 3]
                    rel = np.array([_dot3(tp[batch, :3, i], offset) for i in range(3)], np.float32)
                    z = np.maximum(rel[2], _EPS)
                    u = _F(_F(_F(tk[batch, 0, 0] * rel[0]) / z) + tk[batch, 0, 2])
                    v = _F(_F(_F(tk[batch, 1, 1] * rel[1]) / z) + tk[batch, 1, 2])
                    xy[batch, :, y, x] = (u, v)
                    raw_z[batch, 0, y, x] = rel[2]
                    visible[batch, 0, y, x] = valid and rel[2] > 0 and 0 <= u <= tw - 1 and 0 <= v <= th - 1
                    dx, dy = _F(u - _F(x)), _F(v - _F(y))
                    displacement[batch, 0, y, x] = np.sqrt(_F(_F(dx * dx) + _F(dy * dy)))
        counts = visible.sum(axis=(1, 2, 3))
        parallax = (displacement * visible).sum(axis=(1, 2, 3), dtype=np.float32) / np.maximum(counts, 1).astype(np.float32)
    overlap = visible.astype(np.float32).mean(axis=(1, 2, 3))
    out = dict(projected_xy=xy, projected_depth=raw_z, visible=visible,
               displacement=displacement, overlap=overlap, parallax=parallax)
    if target_depth is not None:
        sampled = sample_pixels(target_depth, xy)
        valid = visible & np.isfinite(sampled) & (sampled > 0)
        out.update(depth_consistency=np.where(valid, np.abs(sampled - raw_z), _F(0)),
                   depth_consistency_valid=valid,
                   occluded=valid & (sampled < raw_z - _F(consistency_threshold)),
                   occlusion_valid=valid.copy())
    return out


def rasterize_reprojection(source_depth, source_confidence, source_intrinsics,
                           source_pose_c2w, target_intrinsics, target_pose_c2w, *, target_hw):
    """Z-buffer positive visible depths; collect every epsilon-close supporter."""
    confidence = _array(source_confidence)
    if (confidence.shape != np.shape(source_depth) or not np.isfinite(confidence).all()
            or np.any(confidence < 0) or np.any(confidence > 1)):
        raise ValueError('source_confidence must be finite [0,1] and match source_depth')
    th, tw = target_hw
    if th < 1 or tw < 1:
        raise ValueError('target_hw must be positive')
    projection = reproject_rgbd(source_depth, source_intrinsics, source_pose_c2w,
                              target_intrinsics, target_pose_c2w, target_hw=target_hw)
    batches, _, h, w = np.shape(source_depth)
    shape = (batches, 1, th, tw)
    depth = np.full(shape, np.inf, np.float32)
    out_confidence = np.zeros(shape, np.float32)
    support = np.zeros(shape, np.int64)
    contributors = []
    for b in range(batches):
        for y in range(h):
            for x in range(w):
                if not projection['visible'][b, 0, y, x]:
                    continue
                u, v = projection['projected_xy'][b, :, y, x]
                xx, yy = min(max(int(np.rint(u)), 0), tw - 1), min(max(int(np.rint(v)), 0), th - 1)
                z = projection['projected_depth'][b, 0, y, x]
                destination = (b, 0, yy, xx)
                depth[destination] = min(depth[destination], z)
                contributors.append((destination, z, confidence[b, 0, y, x]))
    for destination, z, c in contributors:
        minimum = depth[destination]
        if abs(_F(z - minimum)) <= _F(_EPS * max(abs(minimum), _F(1))):
            out_confidence[destination] = max(out_confidence[destination], c)
            support[destination] += 1
    valid = np.isfinite(depth)
    return dict(depth=np.where(valid, depth, _F(0)), valid=valid,
                confidence=out_confidence, support_count=support)


depth_reproject = reproject_rgbd
