# SPDX-License-Identifier: Apache-2.0
"""CPU-only point-plane fixtures and independent checks for the native adapter.

No SlangPy device is constructed. Source compilation remains the caller's job;
record compiler/version/flags and SHA256 of both SPIR-V files with the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from family_support import check_point_oracle
from inlined_point_support import point_case

NAMES = ('points', 'targets', 'normals', 'out_0', 'out_1', 'points_indices')
PARAM_NAMES = ('problem_size', 'points_num_alloc', 'targets_num_alloc',
               'normals_num_alloc', 'out_0_num_alloc', 'out_1_num_alloc')
COUNTS = (1, 63, 1023, 1024, 1025, 4099, 65539)


def export_fixture(directory: Path, count: int, profile: str, candidate_local_size=1024):
    if count <= 0:
        raise ValueError('count must be positive')
    case = point_case(None, count, profile,candidate_local_size=candidate_local_size)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in NAMES:
        array = case['arrays'][name]
        # Vulkan fixtures are explicitly little-endian 32-bit words.
        data = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'))).tobytes()
        (directory / f'{name}.bin').write_bytes(data)
        files[f'{name}.bin'] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                    dtype=array.dtype.str, shape=list(array.shape))
    params = case['params']['params']
    data = struct.pack('<8I', *(params[name] for name in PARAM_NAMES), 0, 0)
    (directory / 'params.bin').write_bytes(data)
    files['params.bin'] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    metadata = dict(schema='sintergpu.native.point_plane.fixture.v1', count=count,
                    allocation=case['allocation'], profile=profile, params=params,
                    candidate_local_size=candidate_local_size,
                    files=files,
                    source_sha256={name: hashlib.sha256(source.encode()).hexdigest()
                                   for name, source in case['sources'].items()},
                    source_compilation='external; attach pinned compiler identity, flags, and SPIR-V SHA256')
    (directory / 'fixture.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return case


def check_native_outputs(output_prefix: Path, count: int, profile: str):
    case = point_case(None, count, profile)
    for phase in ('before', 'after'):
        outputs = {}
        for variant in ('baseline', 'candidate'):
            outputs[variant] = {}
            for name in case['outputs']:
                path = Path(f'{output_prefix}.{phase}.{variant}.{name}.bin')
                data = path.read_bytes()
                if len(data) != case['arrays'][name].nbytes:
                    raise ValueError(f'{path}: wrong output byte count')
                outputs[variant][name] = np.frombuffer(data, dtype='<f4').copy()
            check_point_oracle(case, outputs[variant])
        for name in case['outputs']:
            if outputs['baseline'][name].tobytes() != outputs['candidate'][name].tobytes():
                raise AssertionError(f'{phase}: byte differential mismatch in {name}')
    return dict(independent_float64_oracle=True, full_byte_differential=True,
                poisoned_padding=True, phases=['before', 'after'], count=count, profile=profile)


def self_test():
    import tempfile
    for profile in ('aosoa', 'scalar'):
        for count in COUNTS:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case = export_fixture(directory, count, profile)
                assert struct.unpack('<8I', (directory / 'params.bin').read_bytes()) == (
                    count, max(1, count // 2), count + 7, count + 7, count + 7, count + 7, 0, 0)
                for name in NAMES:
                    assert (directory / f'{name}.bin').read_bytes() == case['arrays'][name].tobytes()
                # Feed exact float64 oracle rounded to float32 through the same checker.
                from family_support import point_oracle
                residual, jacobian = point_oracle(case)
                result = {name: case['arrays'][name].copy() for name in case['outputs']}
                result['out_0'][:count] = residual
                result['out_1'][:4*count] = jacobian[:, :4].reshape(-1)
                result['out_1'][4*case['allocation']:4*case['allocation']+2*count] = jacobian[:, 4:].reshape(-1)
                prefix = directory / 'result.json'
                for phase in ('before', 'after'):
                    for variant in ('baseline', 'candidate'):
                        for name, array in result.items():
                            Path(f'{prefix}.{phase}.{variant}.{name}.bin').write_bytes(array.tobytes())
                check_native_outputs(prefix, count, profile)
                path = Path(f'{prefix}.after.candidate.out_0.bin')
                corrupted = bytearray(path.read_bytes()); corrupted[-1] ^= 1; path.write_bytes(corrupted)
                try:
                    check_native_outputs(prefix, count, profile)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('corrupt poison padding was accepted')
    print('CPU fixture and oracle checks passed for both profiles and all seven representative counts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--profile', choices=('aosoa', 'scalar'), default='aosoa')
    parser.add_argument('--check-native-output', type=Path, help='native JSON output path, used as dump prefix')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.count or args.count <= 0:
        parser.error('--count must be positive')
    if args.fixture_dir:
        export_fixture(args.fixture_dir, args.count, args.profile)
    if args.check_native_output:
        print(json.dumps(check_native_outputs(args.check_native_output, args.count, args.profile), indent=2))
    if not args.fixture_dir and not args.check_native_output:
        parser.error('provide --fixture-dir and/or --check-native-output')


if __name__ == '__main__':
    main()
