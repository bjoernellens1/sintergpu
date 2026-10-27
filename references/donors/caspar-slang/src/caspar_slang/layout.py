"""Host packing for Caspar float32 vector-chunk AoSoA memory."""

from __future__ import annotations

import numpy as np


def _chunks(components: int, layout=None):
    if components <= 0:
        raise ValueError("components must be positive")
    chunks = (
        [list(range(i, min(i + 4, components))) for i in range(0, components, 4)]
        if not layout
        else layout
    )
    if sorted(j for chunk in chunks for j in chunk) != list(range(components)) or any(
        not 1 <= len(c) <= 4 for c in chunks
    ):
        raise ValueError("layout must contain each component exactly once in chunks of 1 to 4")
    return chunks


def storage_components(components: int, layout=None) -> int:
    return sum(4 if len(c) == 3 else len(c) for c in _chunks(components, layout))


def pack(array, layout=None) -> np.ndarray:
    data = np.asarray(array)
    if data.dtype != np.float32:
        raise TypeError("Caspar arrays must have dtype float32")
    if data.ndim != 2:
        raise ValueError("expected shape (N, D)")
    n, d = data.shape
    chunks = _chunks(d, layout)
    result = np.zeros(n * storage_components(d, chunks), dtype=np.float32)
    offset = 0
    for chunk in chunks:
        width = 4 if len(chunk) == 3 else len(chunk)
        result[offset : offset + n * width].reshape(n, width)[:, : len(chunk)] = data[:, chunk]
        offset += n * width
    return result


def unpack(array, n: int, components: int, layout=None) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be nonnegative")
    data = np.asarray(array)
    if data.dtype != np.float32:
        raise TypeError("Caspar arrays must have dtype float32")
    chunks = _chunks(components, layout)
    if data.ndim != 1 or data.size != n * storage_components(components, chunks):
        raise ValueError("packed buffer size does not match dimensions")
    result = np.empty((n, components), dtype=np.float32)
    offset = 0
    for chunk in chunks:
        width = 4 if len(chunk) == 3 else len(chunk)
        result[:, chunk] = data[offset : offset + n * width].reshape(n, width)[:, : len(chunk)]
        offset += n * width
    return result
