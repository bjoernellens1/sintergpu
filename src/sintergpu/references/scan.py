"""Independent serial segmented scan/compaction oracles (no runtime imports)."""
import numpy as np


def segmented_scan(values, segment_ids=None, *, product=False, exclusive=False):
    """Scan the last axis, or flattened nondecreasing uint segment IDs.

    Packed empty segments have no elements and need no special sentinel. Passing
    their expanded IDs is equivalent to nerfacc's (start,count) representation.
    """
    data = np.asarray(values)
    if data.ndim == 0:
        raise ValueError('values must have at least one dimension')
    if segment_ids is None:
        width = data.shape[-1]
        ids = np.repeat(np.arange(int(np.prod(data.shape[:-1]))), width)
    else:
        ids = np.asarray(segment_ids)
        if data.ndim != 1 or ids.shape != data.shape or np.any(ids[1:] < ids[:-1]):
            raise ValueError('flattened values require equal-shape nondecreasing segment IDs')
    flat = data.reshape(-1)
    result = np.empty_like(flat)
    accumulator = data.dtype.type(1 if product else 0)
    for i, value in enumerate(flat):
        if i == 0 or ids[i] != ids[i-1]:
            accumulator = data.dtype.type(1 if product else 0)
        if exclusive:
            result[i] = accumulator
        accumulator = accumulator * value if product else accumulator + value
        if not exclusive:
            result[i] = accumulator
    return result.reshape(data.shape)


def stable_compact(values, predicate):
    data, mask = np.asarray(values), np.asarray(predicate)
    if data.ndim != 1 or mask.shape != data.shape:
        raise ValueError('matching flat values and predicate required')
    return np.asarray([value for value, keep in zip(data, mask) if keep], dtype=data.dtype)
