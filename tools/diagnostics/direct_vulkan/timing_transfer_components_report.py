# SPDX-License-Identifier: Apache-2.0
"""Offline validation of retained host subdivision records; not a GPU runner."""
import hashlib


def validate_components(samples: bytes, components: bytes, *, count: int = 50):
    if type(count) is not int or count != 50:
        raise ValueError('exactly 50 triples required')
    rows = [line.split() for line in samples.decode('ascii').splitlines() if not line.startswith('#')]
    parts = [line.split() for line in components.decode('ascii').splitlines()]
    if len(rows) != count * 3 or len(parts) != len(rows):
        raise ValueError('incomplete sample or component journal')
    result = []
    for index, (row, part) in enumerate(zip(rows, parts)):
        if len(row) != 10 or len(part) != 8:
            raise ValueError('record shape')
        if any(not token.isascii() or not token.isdecimal() for token in part):
            raise ValueError('noncanonical component number')
        values = list(map(int, part))
        if values[:3] != [0, index // 3, index % 3] or row[:3] != part[:3]:
            raise ValueError('record order')
        if values[3] != int(row[8]) or sum(values[4:]) != values[3]:
            raise ValueError('component sum or sample mismatch')
        result.append(values)
    return {'sample_sha256': hashlib.sha256(samples).hexdigest(),
            'components_sha256': hashlib.sha256(components).hexdigest(),
            'records': result, 'status': 'unqualified_host_clock_diagnostic'}
