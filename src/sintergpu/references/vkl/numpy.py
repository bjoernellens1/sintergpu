"""Independent float32 CPU oracles for experimental VKL forward profiles.

These routines never allocate devices or submit GPU work. Masks use bool on CPU;
``mask_u32`` explicitly converts them for the GPU ABI, never reinterprets bytes.
Camera validity is a caller transaction-boundary obligation for GPU dispatch.
"""
from __future__ import annotations

import numpy as np


def mask_u32(value):
    value = np.asarray(value)
    if value.dtype != np.bool_:
        raise ValueError("mask conversion requires boolean values")
    return np.ascontiguousarray(value, dtype=np.uint32)


def project_depth_evidence(xyz, visible, provisional, depth_candidate,
                           world_to_camera, intrinsics, sensor_depth, *,
                           support_threshold, conflict_threshold,
                           epsilon=np.finfo(np.float32).eps):
    xyz = np.asarray(xyz, dtype=np.float32)
    b, n, channels = xyz.shape
    if channels != 3 or any(np.asarray(m).shape != (b, n) for m in (visible, provisional, depth_candidate)):
        raise ValueError("expected [B,N,3] and matching masks")
    for threshold in (support_threshold, conflict_threshold):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("thresholds must be finite and positive")
    if np.shape(world_to_camera) != (b, 4, 4) or np.shape(intrinsics) != (b, 3, 3):
        raise ValueError("expected batched row-major matrices")
    sensor_depth = np.asarray(sensor_depth, dtype=np.float32)
    if sensor_depth.ndim != 4 or sensor_depth.shape[:2] != (b, 1) or min(sensor_depth.shape[2:]) < 1:
        raise ValueError("expected nonempty [B,1,H,W] depth")
    names = ("eligible", "projected_valid", "sensor_valid", "support", "conflict")
    result = {key: np.zeros((b, n), dtype=bool) for key in names}
    result["depth_residual"] = np.zeros((b, n), dtype=np.float32)
    h, w = sensor_depth.shape[-2:]
    # Scalar point loop is independent of the vectorized shader implementation.
    with np.errstate(all="ignore"):
        for batch in range(b):
            matrix = np.asarray(world_to_camera[batch], dtype=np.float32)
            k = intrinsics[batch]
            for point in range(n):
                p = xyz[batch, point]
                eligible = bool(visible[batch, point] and (provisional[batch, point] or depth_candidate[batch, point]) and np.isfinite(p).all())
                result["eligible"][batch, point] = eligible
                if not eligible:
                    continue
                camera = np.asarray([sum((p[j] * matrix[i, j] for j in range(3)), np.float32(0)) + matrix[i, 3] for i in range(3)], dtype=np.float32)
                z = camera[2]
                if not np.isfinite(z) or z <= epsilon:
                    continue
                u = camera[0] / z * k[0, 0] + k[0, 2]
                v = camera[1] / z * k[1, 1] + k[1, 2]
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                # Python ints do not overflow for huge finite projected coordinates.
                ui, vi = int(np.rint(u)), int(np.rint(v))
                if not (0 <= ui < w and 0 <= vi < h):
                    continue
                result["projected_valid"][batch, point] = True
                lookup = sensor_depth[batch, 0, vi, ui]
                if not np.isfinite(lookup) or lookup <= 0:
                    continue
                result["sensor_valid"][batch, point] = True
                residual = abs(z - lookup)
                result["depth_residual"][batch, point] = residual
                result["support"][batch, point] = residual < np.float32(support_threshold)
                result["conflict"][batch, point] = bool(depth_candidate[batch, point] and z < lookup - np.float32(conflict_threshold))
    return result


def photometric_isp(rgb_linear, *, exposure_ev=None, rgb_gain=None,
                    color_matrix=None, vignette=None, response_gamma=None,
                    epsilon=np.finfo(np.float32).eps):
    """Forward observed RGB only; no autodiff or aggregate diagnostics profile."""
    rgb = np.asarray(rgb_linear, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("expected [B,3,H,W]")
    b, _, h, w = rgb.shape
    raw = rgb.copy()
    if exposure_ev is not None:
        raw *= np.exp2(np.asarray(exposure_ev, dtype=np.float32).reshape(b, 1, 1, 1))
    if rgb_gain is not None:
        raw *= np.asarray(rgb_gain, dtype=np.float32).reshape(b, 3, 1, 1)
    if vignette is not None:
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        r = ys[:, None] ** 2 + xs[None, :] ** 2
        powers = np.stack((r, r*r, r*r*r))
        raw *= np.exp(-np.einsum('bck,khw->bchw', np.asarray(vignette, dtype=np.float32), powers))
    if color_matrix is not None:
        raw = np.einsum('bij,bjhw->bihw', np.asarray(color_matrix, dtype=np.float32), raw)
    raw = np.maximum(raw, np.float32(0))
    if response_gamma is not None:
        gamma = np.asarray(response_gamma, dtype=np.float32).reshape(b, 1, 1, 1)
        raw = (raw + np.float32(epsilon)) ** gamma - np.float32(epsilon) ** gamma
    return np.clip(raw, 0, 1).astype(np.float32)


def depth_convert(values, valid, confidence, *, representation=0, scale=1.0, focal_baseline=1.0):
    """Numeric profile: 0=depth, 1=inverse depth, 2=disparity; status stays host policy."""
    if representation not in (0, 1, 2):
        raise ValueError("unknown representation")
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(all="ignore"):
        depth = values * np.float32(scale) if representation == 0 else (1 / (values * np.float32(scale)) if representation == 1 else np.reciprocal(values) * np.float32(focal_baseline))
        usable = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values > 0) & np.isfinite(depth) & (depth > 0)
        depth = np.where(usable, depth, 0).astype(np.float32)
        inverse = np.where(usable, 1 / depth, 0).astype(np.float32)
    return dict(depth=depth, inverse_depth=inverse, valid=usable, confidence=np.where(usable, confidence, 0).astype(np.float32))


def depth_pyramid_level(depth, valid, confidence, *, epsilon=np.finfo(np.float32).eps):
    """One donor level, including its masked-INF maximum behavior at invalid edges."""
    depth = np.asarray(depth, dtype=np.float32)
    b, _, h, w = depth.shape
    shape = (b, 1, (h+1)//2, (w+1)//2)
    out = dict(depth=np.zeros(shape, np.float32), valid=np.zeros(shape, bool), confidence=np.zeros(shape, np.float32))
    for batch in range(b):
        for y in range(shape[2]):
            for x in range(shape[3]):
                samples = []
                for dy in range(2):
                    for dx in range(2):
                        yy, xx = 2*y+dy, 2*x+dx
                        if yy < h and xx < w and valid[batch, 0, yy, xx] and np.isfinite(depth[batch, 0, yy, xx]) and depth[batch, 0, yy, xx] > 0:
                            samples.append((depth[batch, 0, yy, xx], confidence[batch, 0, yy, xx]))
                if samples:
                    closest = min(samples, key=lambda p: p[0])  # first tie wins
                    same = len(samples) == 4 and max(p[0] for p in samples) - closest[0] <= np.float32(epsilon*32)
                    idx = (batch, 0, y, x)
                    out['valid'][idx] = True
                    out['depth'][idx] = np.mean([p[0] for p in samples], dtype=np.float32) if same else closest[0]
                    out['confidence'][idx] = np.mean([p[1] for p in samples], dtype=np.float32) if same else closest[1]
    return out
