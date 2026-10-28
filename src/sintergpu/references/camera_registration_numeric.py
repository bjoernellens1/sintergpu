"""Independent NumPy float32 oracle for VKL camera registration's eager profile."""
from __future__ import annotations

import math
import ctypes
import ctypes.util
import struct
import numpy as np

_F = np.float32
# The pinned Torch 2.14.1+rocm7.2 CPU profile uses FMA per bilinear tap.
def _load_fmaf():
    try:
        library = ctypes.CDLL(ctypes.util.find_library('m') or 'libm.so.6')
        function = library.fmaf
        function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        function.restype = ctypes.c_float
        return function
    except (OSError, AttributeError):
        return None


def _fma_fallback(a, b, c):
    """Correctly rounded binary32 fused multiply-add, including special values."""
    def bits(x):
        return struct.unpack('<I', struct.pack('<f', float(x)))[0]
    def decode(x):
        raw = bits(x); sign = raw >> 31; exp = (raw >> 23) & 255; frac = raw & 0x7fffff
        if exp == 255:
            return sign, None, frac
        return sign, (frac if exp == 0 else frac | 0x800000), (-149 if exp == 0 else exp - 150)
    sa, ma, ea = decode(a); sb, mb, eb = decode(b); sc, mc, ec = decode(c)
    def result(raw):
        return _F(struct.unpack('<f', struct.pack('<I', raw))[0])
    if (ma is None and ea) or (mb is None and eb) or (mc is None and ec):
        return result(0x7fc00000)
    ps = sa ^ sb
    if ma is None or mb is None:
        if ma == 0 or mb == 0 or (mc is None and sc != ps):
            return result(0x7fc00000)
        return result((ps << 31) | 0x7f800000)
    if mc is None:
        return result((sc << 31) | 0x7f800000)
    pm = ma * mb; pe = ea + eb
    exponent = min(pe, ec)
    total = ((-pm if ps else pm) << (pe-exponent)) + ((-mc if sc else mc) << (ec-exponent))
    if total == 0:
        return result((ps & sc) << 31 if pm == 0 and mc == 0 else 0)
    sign = int(total < 0); magnitude = abs(total)
    top = magnitude.bit_length() - 1 + exponent
    if top > 127:
        return result((sign << 31) | 0x7f800000)
    shift = max(magnitude.bit_length()-24, -149-exponent, 0)
    rounded = magnitude >> shift
    if shift:
        remainder = magnitude - (rounded << shift); halfway = 1 << (shift-1)
        if remainder > halfway or (remainder == halfway and rounded & 1):
            rounded += 1
    e = exponent + shift
    if rounded == 0:
        return result(sign << 31)
    if rounded >= 0x1000000:
        rounded >>= 1; e += 1
    top = rounded.bit_length()-1+e
    if top > 127:
        return result((sign << 31) | 0x7f800000)
    if top < -126:
        return result((sign << 31) | (rounded << (e+149)))
    # If the significand has fewer than 24 bits, normalize it exactly.
    left = 24 - rounded.bit_length()
    rounded <<= left; e -= left
    return result((sign << 31) | ((e+150) << 23) | (rounded & 0x7fffff))


_fmaf = _load_fmaf() or _fma_fallback


def _hw(hw):
    if not isinstance(hw, (tuple, list)) or len(hw) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in hw):
        raise ValueError('dimensions must be positive integers')
    return tuple(hw)


def _array(a, shape, name):
    if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != shape:
        raise ValueError(f'{name} must be float32 {shape}')


def _camera(k, batch, name, validate):
    _array(k, (batch, 3, 3), name)
    if validate and (not np.isfinite(k).all() or np.any(k[:, 0, 0] <= 0) or np.any(k[:, 1, 1] <= 0) or np.any(k[:, 0, 1] != 0) or np.any(k[:, 1, 0] != 0) or np.any(k[:, 2, :2] != 0) or np.any(k[:, 2, 2] != 1)):
        raise ValueError(f'{name} must be canonical pinhole intrinsics')


def _coeff(model, coeff, batch, validate):
    if model == 'pinhole' and coeff is None:
        return None
    if model != 'brown_conrady' or coeff is None:
        raise ValueError('invalid distortion model or coefficients')
    _array(coeff, (batch, 5), 'coefficients')
    if validate and not np.isfinite(coeff).all():
        raise ValueError('coefficients must be finite')
    return coeff


def _brown(x, y, coeff):
    if coeff is None:
        return x, y
    k1, k2, p1, p2, k3 = (coeff[:, i, None, None] for i in range(5))
    r = x*x + y*y
    radial = _F(1) + k1*r + k2*(r*r) + k3*(r*r*r)
    return x*radial + _F(2)*p1*x*y + p2*(r + _F(2)*x*x), y*radial + p1*(r + _F(2)*y*y) + _F(2)*p2*x*y


def _coordinates(k, hw):
    h, w = hw
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    return x, y, (x[None] - k[:, 0, 2, None, None])/k[:, 0, 0, None, None], (y[None] - k[:, 1, 2, None, None])/k[:, 1, 1, None, None]


def camera_rectify_grid(source_intrinsics, target_intrinsics, source_hw, output_hw, *, source_distortion_model='pinhole', source_distortion_coefficients=None, debug=False, validate=True):
    if not isinstance(source_intrinsics, np.ndarray) or source_intrinsics.ndim != 3 or source_intrinsics.shape[1:] != (3, 3) or source_intrinsics.shape[0] < 1:
        raise ValueError('source_intrinsics must be [B,3,3]')
    b = source_intrinsics.shape[0]
    _camera(source_intrinsics, b, 'source_intrinsics', validate)
    _camera(target_intrinsics, b, 'target_intrinsics', validate)
    sh, sw = _hw(source_hw)
    h, w = _hw(output_hw)
    coeff = _coeff(source_distortion_model, source_distortion_coefficients, b, validate)
    x, y, xn, yn = _coordinates(target_intrinsics, (h, w))
    xd, yd = _brown(xn, yn, coeff)
    u = xd*source_intrinsics[:, 0, 0, None, None] + source_intrinsics[:, 0, 2, None, None]
    v = yd*source_intrinsics[:, 1, 1, None, None] + source_intrinsics[:, 1, 2, None, None]
    tol = 16*np.finfo(np.float32).eps*max(sh, sw)
    valid = np.isfinite(u) & np.isfinite(v) & (u >= -tol) & (u <= sw-1+tol) & (v >= -tol) & (v <= sh-1+tol)
    out = {'source_uv': np.stack((u, v), axis=-1), 'valid': valid[:, None]}
    if debug:
        out.update(target_ray=np.stack((xn, yn, np.ones_like(xn)), axis=-1), displacement=np.sqrt((u-x)**2 + (v-y)**2)[:, None])
    return out


def camera_rectify_from_grid(image, source_uv, valid, *, sampling='bilinear'):
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim != 4 or sampling not in ('bilinear', 'nearest'):
        raise ValueError('invalid image or sampling')
    b, channels, h, w = image.shape
    if h < 1 or w < 1 or not isinstance(source_uv, np.ndarray) or source_uv.ndim != 4 or source_uv.shape[0] != b or source_uv.shape[-1] != 2 or source_uv.dtype != np.float32:
        raise ValueError('invalid source_uv')
    th, tw = source_uv.shape[1:3]
    if not isinstance(valid, np.ndarray) or valid.shape != (b, 1, th, tw) or valid.dtype != np.bool_:
        raise ValueError('invalid valid')
    result = np.zeros((b, channels, th, tw), dtype=np.float32)
    for bi in range(b):
        for j in range(th):
            for i in range(tw):
                u, v = source_uv[bi, j, i]
                u = _F(np.clip(u if np.isfinite(u) else 0, 0, w-1))
                v = _F(np.clip(v if np.isfinite(v) else 0, 0, h-1))
                # Reproduce normalized align-corners conversion and its float32 boundaries.
                gx = _F(_F(_F(2)*u)/_F(max(w-1, 1)) - _F(1))
                gy = _F(_F(_F(2)*v)/_F(max(h-1, 1)) - _F(1))
                x = _F(_F(gx + _F(1))*_F(w-1)/_F(2))
                y = _F(_F(gy + _F(1))*_F(h-1)/_F(2))
                if not valid[bi, 0, j, i]:
                    continue
                if sampling == 'nearest':
                    result[bi, :, j, i] = image[bi, :, int(np.rint(y)), int(np.rint(x))]
                else:
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    dx, dy = _F(x-x0), _F(y-y0)
                    for channel in range(channels):
                        value = _F(0)
                        for yy, wy in ((y0, _F(1)-dy), (y0+1, dy)):
                            for xx, wx in ((x0, _F(1)-dx), (x0+1, dx)):
                                if 0 <= xx < w and 0 <= yy < h:
                                    value = _F(_fmaf(image[bi, channel, yy, xx], _F(wx*wy), value))
                        result[bi, channel, j, i] = value
    return {'image': result, 'valid': valid}


def camera_rectify(image, source_intrinsics, target_intrinsics, output_hw, *, source_distortion_model='pinhole', source_distortion_coefficients=None, sampling='bilinear', debug=False, validate=True):
    if not isinstance(image, np.ndarray) or image.ndim != 4 or image.dtype != np.float32 or image.shape[0] != source_intrinsics.shape[0]:
        raise ValueError('invalid image')
    grid = camera_rectify_grid(source_intrinsics, target_intrinsics, image.shape[-2:], output_hw, source_distortion_model=source_distortion_model, source_distortion_coefficients=source_distortion_coefficients, debug=debug, validate=validate)
    out = camera_rectify_from_grid(image, grid['source_uv'], grid['valid'], sampling=sampling)
    if debug:
        out.update({key: grid[key] for key in ('source_uv', 'target_ray', 'displacement')})
    return out


def depth_camera_rays(depth_intrinsics, depth_hw, *, depth_distortion_model='pinhole', depth_distortion_coefficients=None, validate=True):
    if not isinstance(depth_intrinsics, np.ndarray) or depth_intrinsics.ndim != 3 or depth_intrinsics.shape[1:] != (3, 3) or depth_intrinsics.shape[0] < 1:
        raise ValueError('invalid depth_intrinsics')
    b = depth_intrinsics.shape[0]
    _camera(depth_intrinsics, b, 'depth_intrinsics', validate)
    coeff = _coeff(depth_distortion_model, depth_distortion_coefficients, b, validate)
    _, _, xd, yd = _coordinates(depth_intrinsics, _hw(depth_hw))
    x, y = xd, yd
    if coeff is not None:
        for _ in range(5):
            px, py = _brown(x, y, coeff)
            x, y = x + xd - px, y + yd - py
    return np.stack((x, y), axis=-1)


def depth_register(depth, depth_intrinsics, rgb_intrinsics, transform_rgb_depth, output_hw, *, depth_distortion_model='pinhole', depth_distortion_coefficients=None, rgb_distortion_model='pinhole', rgb_distortion_coefficients=None, min_depth=1e-6, max_depth=None, depth_rays=None, debug=False, validate=True):
    if not isinstance(depth, np.ndarray) or depth.dtype != np.float32 or depth.ndim != 4 or depth.shape[1] != 1 or min(depth.shape[-2:]) < 1:
        raise ValueError('depth must be float32 [B,1,H,W]')
    if not math.isfinite(min_depth) or min_depth <= 0 or (max_depth is not None and (not math.isfinite(max_depth) or max_depth <= min_depth)):
        raise ValueError('invalid depth interval')
    b, _, h, w = depth.shape
    th, tw = _hw(output_hw)
    _camera(depth_intrinsics, b, 'depth_intrinsics', validate)
    _camera(rgb_intrinsics, b, 'rgb_intrinsics', validate)
    _array(transform_rgb_depth, (b, 4, 4), 'transform')
    if validate:
        rotation = transform_rgb_depth[:, :3, :3]
        if not np.isfinite(transform_rgb_depth).all() or not np.allclose(rotation.transpose(0,2,1) @ rotation, np.eye(3), atol=1e-4, rtol=1e-4) or np.any(np.linalg.det(rotation) <= 0) or not np.allclose(transform_rgb_depth[:,3,:], [0,0,0,1], atol=1e-6, rtol=0):
            raise ValueError('transform must be rigid')
    dc = _coeff(depth_distortion_model, depth_distortion_coefficients, b, validate)
    rc = _coeff(rgb_distortion_model, rgb_distortion_coefficients, b, validate)
    if depth_rays is None:
        depth_rays = depth_camera_rays(depth_intrinsics, (h,w), depth_distortion_model=depth_distortion_model, depth_distortion_coefficients=dc, validate=False)
    else:
        _array(depth_rays, (b,h,w,2), 'depth_rays')
        if validate and not np.isfinite(depth_rays).all():
            raise ValueError('depth_rays must be finite')
    d = depth[:,0]
    dv = np.isfinite(d) & (d >= min_depth)
    if max_depth is not None:
        dv &= d <= max_depth
    safe = np.where(dv, d, _F(0))
    points = np.stack((depth_rays[...,0]*safe, depth_rays[...,1]*safe, safe), axis=-1).reshape(b,-1,3)
    xyz = points @ transform_rgb_depth[:, :3, :3].transpose(0,2,1) + transform_rgb_depth[:,None,:3,3]
    z = xyz[...,2]
    positive = np.isfinite(z) & (z > np.finfo(np.float32).eps)
    sz = np.where(positive, z, _F(1))
    xx, yy = _brown((xyz[...,0]/sz)[:,:,None], (xyz[...,1]/sz)[:,:,None], rc)
    u = xx[:,:,0]*rgb_intrinsics[:,None,0,0] + rgb_intrinsics[:,None,0,2]
    v = yy[:,:,0]*rgb_intrinsics[:,None,1,1] + rgb_intrinsics[:,None,1,2]
    finite = np.isfinite(u) & np.isfinite(v) & positive
    source_valid = np.zeros((b,h*w), dtype=bool)
    out_depth = np.zeros((b,1,th,tw), dtype=np.float32)
    winner = np.full((b,1,th,tw), -1, dtype=np.int64)
    counts = np.zeros((b,1,th,tw), dtype=np.int64)
    for bi in range(b):
        for index in range(h*w):
            if not finite[bi,index] or not dv.reshape(b,-1)[bi,index]:
                continue
            ui, vi = np.rint(u[bi,index]), np.rint(v[bi,index])
            if not (0 <= ui < tw and 0 <= vi < th):
                continue
            ix, iy = int(ui), int(vi)
            source_valid[bi,index] = True
            counts[bi,0,iy,ix] += 1
            if winner[bi,0,iy,ix] < 0 or z[bi,index] < out_depth[bi,0,iy,ix]:
                winner[bi,0,iy,ix] = index
                out_depth[bi,0,iy,ix] = z[bi,index]
    valid = winner >= 0
    out = {'depth': out_depth, 'valid': valid}
    if debug:
        out.update(winner_source_index=winner, collision_count=counts, source_uv=np.stack((u,v), axis=-1), projected_z=z, source_valid=source_valid)
    return out
