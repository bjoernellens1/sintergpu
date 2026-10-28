# SPDX-License-Identifier: Apache-2.0
"""Scalar float32 oracles for the existing VKL and SLAM downsample profiles.

VKL donor: 3384023a311651ea4260cf12ddab1567972f89f3;
SLAM donor: e35e52b35f692b8337696ea3c9d6721d1cc2bc6b.
These CPU models are not GPU qualification or a claim of profile equivalence.
"""
from __future__ import annotations

import numpy as np


def _array(value, ndim, dtype, name):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype != dtype or any(n == 0 for n in a.shape):
        raise ValueError(f'{name} must be a nonempty {ndim}D {np.dtype(dtype)} array')
    return a


def vkl_level(depth, valid, confidence, epsilon):
    """Return one ceil-half level for float32 [batch,height,width] inputs.

    Any nonzero uint32 mask is true. Partial blocks use their first minimum;
    only four usable, near-equal depths are averaged. Inputs are never modified.
    """
    d = _array(depth, 3, np.float32, 'depth')
    v = _array(valid, 3, np.uint32, 'valid')
    c = _array(confidence, 3, np.float32, 'confidence')
    if d.shape != v.shape or d.shape != c.shape:
        raise ValueError('input shapes must match')
    eps = np.float32(epsilon)
    b, h, w = d.shape
    shape = (b, (h + 1) // 2, (w + 1) // 2)
    result = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.uint32)
    conf = np.zeros(shape, np.float32)
    with np.errstate(invalid='ignore', over='ignore'):
        for batch in range(b):
            for row in range(shape[1]):
                for col in range(shape[2]):
                    samples = []
                    for sy in range(row * 2, min(row * 2 + 2, h)):
                        for sx in range(col * 2, min(col * 2 + 2, w)):
                            x = d[batch, sy, sx]
                            if v[batch, sy, sx] != 0 and np.isfinite(x) and x > 0:
                                samples.append((x, c[batch, sy, sx]))
                    if not samples:
                        continue
                    mask[batch, row, col] = 1
                    minimum = min(samples, key=lambda s: s[0])
                    if len(samples) == 4 and np.float32(max(s[0] for s in samples) - minimum[0]) <= np.float32(eps * np.float32(32)):
                        total = np.float32(0)
                        total_conf = np.float32(0)
                        for value, certainty in samples:
                            total = np.float32(total + value)
                            total_conf = np.float32(total_conf + certainty)
                        result[batch, row, col] = np.float32(total / np.float32(4))
                        conf[batch, row, col] = np.float32(total_conf / np.float32(4))
                    else:
                        result[batch, row, col], conf[batch, row, col] = minimum
    return result, mask, conf


def slam_level(depth, depth_jump):
    """Return the SLAM profile for an even float32 [height,width] image.

    Positive infinity is admitted, NaN and both zeros are not. The minimum
    starts at float32 1e20, including when every admitted sample is larger.
    """
    d = _array(depth, 2, np.float32, 'depth')
    h, w = d.shape
    if h % 2 or w % 2:
        raise ValueError('SLAM depth dimensions must be even')
    jump = np.float32(depth_jump)
    out = np.zeros((h // 2, w // 2), np.float32)
    with np.errstate(invalid='ignore', over='ignore'):
        for row in range(h // 2):
            for col in range(w // 2):
                total = np.float32(0)
                minimum = np.float32(1e20)
                maximum = np.float32(0)
                count = 0
                for sy in (row * 2, row * 2 + 1):
                    for sx in (col * 2, col * 2 + 1):
                        value = d[sy, sx]
                        if value > 0:
                            count += 1
                            total = np.float32(total + value)
                            minimum = min(minimum, value)
                            maximum = max(maximum, value)
                if count and np.float32(maximum - minimum) <= jump:
                    out[row, col] = np.float32(total / np.float32(count))
    return out
