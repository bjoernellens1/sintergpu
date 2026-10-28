# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy T0 composition of the scalar D4 and D9 oracles.

These functions preserve unmasked aligned RGB and depth-fusion diagnostics.
Only final RGB-D/confidence is masked by the complete reference validity.
"""
import numpy as np

from .geometry_depth_numeric import temporal_depth_fuse
from .reprojection_numeric import _array, rasterize_reprojection, reproject_rgbd, sample_pixels

_F = np.float32
_EPS = np.finfo(np.float32).eps


def temporal_reproject_rgb(target_rgb, target_depth, source_rgb,
                           target_intrinsics, target_pose_c2w,
                           source_intrinsics, source_pose_c2w, *,
                           source_depth=None, source_confidence=None,
                           consistency_threshold=0.05):
    """Sample [B,N] prior RGB with target depth geometry and optional evidence."""
    rgb, depth, source, tk, tp, sk, sp = map(_array, (
        target_rgb, target_depth, source_rgb, target_intrinsics,
        target_pose_c2w, source_intrinsics, source_pose_c2w))
    if rgb.ndim != 4 or rgb.shape[1] != 3 or source.ndim != 5:
        raise ValueError('target RGB-D and source RGB [B,N,3,H,W] must align')
    b, n, _, h, w = source.shape
    if rgb.shape != (b, 3, h, w) or depth.shape != (b, 1, h, w):
        raise ValueError('target RGB-D and source RGB [B,N,3,H,W] must align')
    if tk.shape != (b, 3, 3) or tp.shape != (b, 4, 4) or sk.shape != (b, n, 3, 3) or sp.shape != (b, n, 4, 4):
        raise ValueError('target and source camera geometry must match RGB batches')
    if consistency_threshold <= 0:
        raise ValueError('consistency_threshold must be positive')
    expected = (b, n, 1, h, w)
    if source_depth is not None:
        source_depth = _array(source_depth)
        if source_depth.shape != expected:
            raise ValueError('source_depth must match [B,N,1,H,W]')
    if source_confidence is not None:
        source_confidence = _array(source_confidence)
        if (source_confidence.shape != expected or not np.isfinite(source_confidence).all()
                or np.any(source_confidence < 0) or np.any(source_confidence > 1)):
            raise ValueError('source_confidence must be finite [0,1] and match source RGB')
    # Pair ordering is batch first then neighbor. D4 consumes each pair as a
    # separate camera observation; this is CPU reference storage only.
    paired = lambda value: np.repeat(value[:, None], n, axis=1).reshape(b * n, *value.shape[1:])
    projected = reproject_rgbd(paired(depth), paired(tk), paired(tp),
                              sk.reshape(b * n, 3, 3), sp.reshape(b * n, 4, 4),
                              None if source_depth is None else source_depth.reshape(b * n, 1, h, w),
                              target_hw=(h, w), consistency_threshold=consistency_threshold)
    xy = projected['projected_xy']
    aligned = sample_pixels(source.reshape(b * n, 3, h, w), xy, mode='bilinear')
    valid = projected['visible'].copy()
    confidence = valid.astype(np.float32)
    if source_depth is not None:
        consistent = projected['depth_consistency_valid'] & ~projected['occluded']
        valid &= consistent
        confidence *= np.exp(-projected['depth_consistency'] / _F(consistency_threshold))
        confidence *= consistent
    if source_confidence is not None:
        confidence *= sample_pixels(source_confidence.reshape(b * n, 1, h, w), xy, mode='bilinear')
    return dict(aligned_rgb=aligned.reshape(b, n, 3, h, w),
                aligned_valid=valid.reshape(expected),
                alignment_confidence=confidence.reshape(expected))


def temporal_rgbd_reference(*, target_depth, source_rgb, source_depth,
                            target_intrinsics, target_pose_c2w,
                            source_intrinsics, source_pose_c2w,
                            source_confidence=None, consistency_threshold=0.05,
                            min_support=1):
    """Build a reference whose depth observations come exclusively from priors."""
    source_rgb = _array(source_rgb)
    source_depth = _array(source_depth)
    target_depth = _array(target_depth)
    if source_rgb.ndim != 5 or source_rgb.shape[2] != 3:
        raise ValueError('source RGB-D must be [B,N,3,H,W]/[B,N,1,H,W]')
    b, n, _, h, w = source_rgb.shape
    if source_depth.shape != (b, n, 1, h, w) or target_depth.shape != (b, 1, h, w) or n < 1:
        raise ValueError('target and source RGB-D camera geometry must match')
    if not isinstance(min_support, int) or isinstance(min_support, bool) or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    if source_confidence is None:
        source_confidence = np.ones_like(source_depth)
    else:
        source_confidence = _array(source_confidence)
    aligned = temporal_reproject_rgb(np.zeros((b, 3, h, w), np.float32), target_depth,
                                     source_rgb, target_intrinsics, target_pose_c2w,
                                     source_intrinsics, source_pose_c2w,
                                     source_depth=source_depth, source_confidence=source_confidence,
                                     consistency_threshold=consistency_threshold)
    pair_camera = lambda value: np.repeat(np.asarray(value)[:, None], n, axis=1).reshape(b * n, *np.shape(value)[1:])
    raster = rasterize_reprojection(source_depth.reshape(b * n, 1, h, w),
                                   source_confidence.reshape(b * n, 1, h, w),
                                   np.asarray(source_intrinsics).reshape(b * n, 3, 3),
                                   np.asarray(source_pose_c2w).reshape(b * n, 4, 4),
                                   pair_camera(target_intrinsics), pair_camera(target_pose_c2w),
                                   target_hw=(h, w))
    pair_shape = (b, n, 1, h, w)
    raster_depth = raster['depth'].reshape(pair_shape)
    raster_valid = raster['valid'].reshape(pair_shape)
    raster_confidence = raster['confidence'].reshape(pair_shape)
    # D9 input is source-major. Slot zero is deliberately unavailable; neither
    # target depth nor target confidence contributes to the reference depth.
    pack = lambda prior: np.concatenate((np.zeros((1, b, 1, h, w), prior.dtype), prior.transpose(1, 0, 2, 3, 4)), axis=0)
    depths, valids, confidences = pack(raster_depth), pack(raster_valid), pack(raster_confidence)
    evidence = np.ones_like(valids)
    fused = temporal_depth_fuse(depths, valids, confidences, evidence, evidence,
                                np.zeros_like(valids), consistency_threshold=consistency_threshold,
                                min_support=min_support, min_temporal_support=1)
    weight = aligned['alignment_confidence'] * raster_confidence
    weight *= raster_valid
    weight *= aligned['aligned_valid']
    weight_sum = weight.sum(axis=1, dtype=np.float32)
    valid = fused['valid'] & (weight_sum > 0)
    rgb = (aligned['aligned_rgb'] * weight).sum(axis=1, dtype=np.float32) / np.maximum(weight_sum, _EPS)
    return dict(rgb=np.where(valid, rgb, _F(0)),
                depth=np.where(valid, fused['depth'], _F(0)), valid=valid,
                confidence=np.where(valid, fused['confidence'] * np.minimum(weight_sum, _F(1)), _F(0)),
                support_count=fused['support_count'],
                temporal_support_count=fused['temporal_support_count'],
                source_disagreement=fused['source_disagreement'])
