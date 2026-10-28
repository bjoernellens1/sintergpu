# SPDX-License-Identifier: Apache-2.0
"""CPU-only fake-encoder recording overhead, not a GPU or throughput benchmark."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.camera_registration_support import BufferDescriptor, PreparedCameraGraph
from sintergpu.camera_registration import camera_rectify_grid_plan, camera_rectify_from_grid_plan, camera_rectify_plan, depth_register_plan


class FakeKernel:
    def dispatch(self, *, thread_count, vars, command_encoder):
        command_encoder.dispatch_count += 1


class FakeModule:
    def entry_point(self, name): return name


class FakeDevice:
    def load_module(self, path): return FakeModule()
    def link_program(self, modules, entries): return entries[0]
    def create_compute_kernel(self, program): return FakeKernel()


class FakeEncoder:
    def __init__(self): self.barrier_count = 0; self.dispatch_count = 0
    def global_barrier(self): self.barrier_count += 1


def cases():
    limit = 2**24
    return {
        'grid-720p': camera_rectify_grid_plan(1,720,1280,720,1280,max_workgroups_x=limit,distortion=True,debug=True),
        'rectify-cached-720p': camera_rectify_from_grid_plan(1,3,720,1280,720,1280,max_workgroups_x=limit),
        'rectify-direct-720p': camera_rectify_plan(1,3,720,1280,720,1280,max_workgroups_x=limit,distortion=True),
        'registration-direct-dense-720p': depth_register_plan(1,480,640,720,1280,max_workgroups_x=limit,depth_distortion=True,rgb_distortion=True),
        'registration-cached-dense-720p': depth_register_plan(1,480,640,720,1280,max_workgroups_x=limit,cached_rays=True),
        'registration-cached-collisions-720p': depth_register_plan(1,480,640,720,1280,max_workgroups_x=limit,cached_rays=True,debug=True),
    }


def collect(iterations=50, warmups=5, records_per_sample=1000):
    if any(type(v) is not int or v < 1 for v in (iterations, warmups, records_per_sample)):
        raise ValueError('positive integer sample, warmup, and repeat counts required')
    results = []
    for name, graph in cases().items():
        device = FakeDevice()
        buffers = {key: object() for key in graph['resources']}
        descriptors = {name: BufferDescriptor(buffer, device, object(), 0, graph['resources'][name]['bytes'], frozenset({'shader_resource', 'unordered_access'})) for name, buffer in buffers.items()}
        prepared = PreparedCameraGraph(device, graph, buffers, descriptors=descriptors, enabled_features=graph['features'])
        encoder = FakeEncoder()
        for _ in range(warmups):
            for _ in range(records_per_sample): prepared.record(encoder)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            for _ in range(records_per_sample): prepared.record(encoder)
            elapsed = time.perf_counter_ns() - start
            if elapsed <= 0: raise RuntimeError('non-increasing clock')
            samples.append(elapsed / records_per_sample)
        expected_records = (warmups + iterations) * records_per_sample
        if encoder.dispatch_count != expected_records * len(graph['stages']):
            raise RuntimeError('dispatch count mismatch')
        expected_barriers = sum(bool(s['barriers']) for s in graph['stages'])
        if encoder.barrier_count != expected_records * expected_barriers:
            raise RuntimeError('barrier count mismatch')
        ordered = sorted(samples)
        results.append(dict(case=name, stage_count=len(graph['stages']), barrier_count=expected_barriers,
                            entries=[s['entrypoint'] for s in graph['stages']], graph_profile=graph['profile'],
                            iterations=iterations, warmups=warmups, records_per_sample=records_per_sample,
                            samples_ns_per_record=samples, median_ns=statistics.median(samples),
                            p95_ns=ordered[(95*len(ordered)+99)//100-1]))
    paths = [Path(__file__), Path(__file__).with_name('camera_registration_support.py'),
             Path(__file__).resolve().parents[1] / 'src/sintergpu/camera_registration.py']
    return dict(scope='CPU Python fake-encoder record calls only; no GPU, allocation, submission, synchronization, transfer, numerical data or kernel execution. Dense/collision cases select graph profiles only; collision density does not alter recorder work. Not a no-regression comparison.',
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(hostname=platform.node(), platform=platform.platform(), python=platform.python_version(), processor=platform.processor(),
                                 logical_cpus=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                                 clock=time.get_clock_info('perf_counter')._asdict() if hasattr(time.get_clock_info('perf_counter'), '_asdict') else str(time.get_clock_info('perf_counter'))),
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--records-per-sample', type=int, default=1000)
    args=parser.parse_args()
    report=collect(args.iterations, args.warmups, args.records_per_sample)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    for row in report['results']:
        print(f"{row['case']}: stages={row['stage_count']} median={row['median_ns']:.1f} ns p95={row['p95_ns']:.1f} ns")


if __name__ == '__main__': main()
