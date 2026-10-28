#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise fixture/ABI rejection without creating a Vulkan instance or lock."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CANDIDATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE / 'benchmarks'))
from native_fixture import COUNTS, export_fixture, self_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('/tmp/sintergpu-point-plane-ab'))
    parser.add_argument('--spirv-root', type=Path, default=CANDIDATE / 'evidence/iteration3/compilers')
    args = parser.parse_args()
    self_test()
    passed = 0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        output = directory / 'never-written.json'

        def run(count, baseline, candidate, expected=0, extra=()):
            nonlocal passed
            command = [str(args.binary), '--baseline', str(baseline), '--candidate', str(candidate),
                       '--fixture-dir', str(directory), '--count', str(count),
                       '--vendor', '0x1002', '--device', '0x1586', '--output', str(output),
                       '--validate-only', *extra]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != expected:
                raise AssertionError((command, result.returncode, result.stdout, result.stderr))
            assert not output.exists(), 'validation wrote a benchmark report'
            assert not Path(str(output) + '.ownership.json').exists(), 'validation acquired a GPU lock'
            passed += 1
            return result

        for profile in ('aosoa', 'scalar'):
            for count in COUNTS:
                export_fixture(directory, count, profile)
                for compiler in ('2026.12', '2026.19'):
                    baseline = args.spirv_root / compiler / f'point-{profile}-baseline.spv'
                    candidate = args.spirv_root / compiler / f'point-{profile}-candidate.spv'
                    run(count, baseline, candidate)
        export_fixture(directory, 63, 'aosoa')
        baseline = args.spirv_root / '2026.12/point-aosoa-baseline.spv'
        candidate = args.spirv_root / '2026.12/point-aosoa-candidate.spv'
        # Six/seven/eight uint Params are all accepted, always uploaded as32 bytes.
        params = (directory / 'params.bin').read_bytes()
        for size in (24, 28, 32):
            (directory / 'params.bin').write_bytes(params[:size])
            run(63, baseline, candidate)
        (directory / 'params.bin').write_bytes(params[:-4] + struct.pack('<I', 1))
        assert 'reserved params' in run(63, baseline, candidate, expected=1).stderr
        (directory / 'params.bin').write_bytes(params)
        index = directory / 'points_indices.bin'
        original = index.read_bytes()
        index.write_bytes(struct.pack('<I', 31) + original[4:])
        assert 'out of bounds' in run(63, baseline, candidate, expected=1).stderr
        index.write_bytes(original[:-1])
        assert 'byte count' in run(63, baseline, candidate, expected=1).stderr
        index.write_bytes(original)
        assert 'explicit allowed vendor' in run(63, baseline, candidate, expected=1,
                                               extra=('--device', '0')).stderr
        altered = directory / 'invalid.spv'
        altered.write_bytes(b'bad')
        assert 'SPIR-V byte count' in run(63, baseline, altered, expected=1).stderr
        def with_local_size(source, local):
            words = list(struct.unpack(f'<{source.stat().st_size//4}I', source.read_bytes()))
            offset = 5
            while offset < len(words):
                length, opcode = words[offset] >> 16, words[offset] & 65535
                if opcode == 16 and words[offset+2] == 17:
                    words[offset+3:offset+6] = local
                    break
                offset += length
            else:
                raise AssertionError('test SPIR-V lacks LocalSize')
            altered.write_bytes(struct.pack(f'<{len(words)}I', *words))
            return altered

        for mode in ('wait', 'spin'):
            result = run(63, baseline, candidate, extra=('--completion-mode', mode))
            assert f'completion mode {mode}' in result.stdout
        assert 'completion-mode must be wait or spin' in run(
            63, baseline, candidate, expected=1, extra=('--completion-mode', 'invalid')).stderr
        # Candidate launch configuration may be256, while baseline remains1024.
        candidate256 = with_local_size(candidate, (256, 1, 1))
        result = run(63, baseline, candidate256)
        assert 'baseline local size 1024, candidate local size 256' in result.stdout
        assert 'baseline must retain' in run(63, candidate256, candidate, expected=1).stderr
        for invalid_local in ((64, 1, 1), (512, 1, 1), (256, 2, 1), (1024, 1, 2)):
            invalid = with_local_size(candidate, invalid_local)
            assert 'numthreads' in run(63, baseline, invalid, expected=1).stderr
        # Validate both pinned lanes and all count/tail fixtures under the256 configuration.
        for profile in ('aosoa', 'scalar'):
            for count in COUNTS:
                export_fixture(directory, count, profile)
                for compiler in ('2026.12', '2026.19'):
                    baseline = args.spirv_root / compiler / f'point-{profile}-baseline.spv'
                    candidate = args.spirv_root / compiler / f'point-{profile}-candidate.spv'
                    result = run(count, baseline, with_local_size(candidate, (256, 1, 1)))
                    assert 'baseline local size 1024, candidate local size 256' in result.stdout
    print(f'{passed} native CPU-only CLI/fixture/SPIR-V checks passed; no GPU work performed')


if __name__ == '__main__':
    main()
