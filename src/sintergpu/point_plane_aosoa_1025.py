# SPDX-License-Identifier: Apache-2.0
"""Checked host contract for one frozen experimental AoSoA fixture.

This validates caller-supplied attestations; it cannot inspect GPU memory or
prove physical allocation identity. No allocation, upload or dispatch occurs.
"""

_SIZES = {'points': 8192, 'targets': 16512, 'normals': 16512, 'out_0': 4156,
          'out_1': 24796, 'points_indices': 4100, 'params': 32}
_WRITES = {'out_0', 'out_1'}
_LIMITS = {'max_workgroups_x': 5, 'max_group_size_x': 256,
           'max_threads_per_group': 256, 'max_buffer_bytes': 24796, 'max_uniform_buffer_bytes': 32}


def fixture_plan(*, limits, ranges, allocations_verified, fixture_contents_verified):
    """Validate exact seeded 1025-element fixture; return caller-recorded dispatch.

    The caller independently verifies complete fixture uploads, index bounds,
    allocation IDs and byte ranges on the actual device. The boolean flags are
    attestations of those checks, not evidence produced by this function.
    """
    if allocations_verified is not True or fixture_contents_verified is not True:
        raise ValueError('caller verification attestations required')
    if not isinstance(limits, dict) or not all(type(limits.get(k)) is int and limits[k] >= v for k, v in _LIMITS.items()):
        raise ValueError('insufficient device limits')
    if not isinstance(ranges, dict) or set(ranges) != set(_SIZES):
        raise ValueError('complete fixture allocation ranges required')
    normalized = {}
    for name, needed in _SIZES.items():
        item = ranges[name]
        if not isinstance(item, dict) or set(item) != {'allocation_id', 'offset', 'length'}:
            raise ValueError('invalid allocation range')
        ident, offset, length = item['allocation_id'], item['offset'], item['length']
        if not isinstance(ident, str) or not ident or type(offset) is not int or type(length) is not int or offset < 0 or length < needed:
            raise ValueError('invalid allocation range or capacity')
        if offset % 4 or length % 4:
            raise ValueError('allocation range alignment')
        if offset >= 1 << 64 or length >= 1 << 64 or offset + length > (1 << 64) - 1:
            raise ValueError('allocation range overflow')
        limit = limits['max_uniform_buffer_bytes'] if name == 'params' else limits['max_buffer_bytes']
        if length > limit:
            raise ValueError('buffer range exceeds device limit')
        normalized[name] = (ident, offset, offset + length)
    for a in normalized:
        for b in normalized:
            if a >= b or not ({a, b} & _WRITES):
                continue
            ai, lo, hi = normalized[a]; bi, blo, bhi = normalized[b]
            if ai == bi and lo < bhi and blo < hi:
                raise ValueError(f'overlap involving writable resource: {a}, {b}')
    # Fixed values avoid uint32 multiply/add overflow in every shader index.
    return {'groups': (5, 1, 1), 'thread_count': (1280, 1, 1),
            'params': (1025, 512, 1032, 1032, 1032, 1032, 0, 0),
            'entrypoint': 'main', 'status': 'experimental-unqualified'}
