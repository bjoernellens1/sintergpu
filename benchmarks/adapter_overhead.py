"""CPU recording overhead only: real adapter with a minimal non-GPU dispatch sink.

Run: PYTHONPATH=src python benchmarks/adapter_overhead.py --output /tmp/adapter-overhead.json
No device, driver, shader compiler, GPU buffer, submission, or GPU timing is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from sintergpu.adapters.slangpy import SlangPyAdapter
from sintergpu.catalog import list_operations


class RecordingSink:
    """Implements only setup and dispatch calls; cannot perform GPU work."""
    def __init__(self):
        self.calls = 0

    def load_module(self, path):
        return self

    def entry_point(self, name):
        return self

    def link_program(self, modules, entry_points):
        return self

    def create_compute_kernel(self, program):
        return self

    def dispatch(self, **kwargs):
        self.calls += 1


def summary(samples):
    ordered = sorted(samples)
    return {'p50_us': statistics.median(ordered), 'p95_us': ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
            'min_us': ordered[0], 'samples': len(ordered)}


def run(operation_id, iterations=1000, samples=10):
    setup, cached = [], []
    sink = RecordingSink()
    encoder, bindings, extent = object(), {}, (640, 480, 1)
    for _ in range(samples):
        start = time.perf_counter_ns()
        adapter = SlangPyAdapter(sink, allow_experimental=True)
        adapter.prepare(operation_id)
        setup.append((time.perf_counter_ns()-start)/1000)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            adapter.record(operation_id, thread_count=extent, bindings=bindings, command_encoder=encoder)
        cached.append((time.perf_counter_ns()-start)/1000/iterations)
    source = Path(__file__).resolve().parents[1] / 'src/sintergpu/adapters/slangpy.py'
    return {'scope': 'CPU adapter setup and warm record overhead with non-GPU recording sink; no GPU cost or correctness claim',
            'python': platform.python_version(), 'operation': operation_id, 'adapter_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'extent': list(extent), 'iterations_per_sample': iterations, 'dispatch_records': sink.calls,
            'setup_including_source_identity_no_compiler': summary(setup), 'cached_record_per_call': summary(cached),
            'setup_samples_us': setup, 'record_samples_us_per_call': cached}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.samples < 1:
        parser.error('iterations and samples must be positive')
    result = run(args.operation or list_operations()[0]['id'], args.iterations, args.samples)
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')
