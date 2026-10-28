# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture recording microbenchmark; no GPU or performance qualification."""
import json
from pathlib import Path
import time

import numpy as np
from benchmarks.depth_pyramid_gpu_fixture import Fixture, specification


class Encoder:
    def global_barrier(self): pass


class Kernel:
    def dispatch(self, *, thread_count, vars, command_encoder): pass


def measure():
    rows = []
    for profile in ('vkl', 'slam'):
        d = np.ones((2, 34, 38), np.float32)
        spec = (specification('vkl', d, valid=np.ones_like(d, np.uint32), confidence=d, threshold=np.finfo(np.float32).eps)
                if profile == 'vkl' else specification('slam', d[0], threshold=.1))
        fixture = object.__new__(Fixture)
        fixture.spec, fixture.bindings, fixture.kernel = spec, {'prebound': object()}, Kernel()
        encoder = Encoder()
        for _ in range(5): fixture.record(encoder)
        samples = []
        for _ in range(50):
            start = time.perf_counter_ns(); fixture.record(encoder); samples.append(time.perf_counter_ns() - start)
        rows.append(dict(profile=profile, warmups=5, samples_ns=samples))
    return dict(scope='CPU fake encoder record call only; no GPU, runtime, submission, driver or performance qualification', measurements=rows)


if __name__ == '__main__':
    path = Path('build/depth-pyramid-recording-cpu.json')
    path.write_text(json.dumps(measure(), indent=2) + '\n')
    print(path)
