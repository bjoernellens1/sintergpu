# Kernel timing methodology

[The timing table](kernel-timings.md) is generated from the machine-readable
[`timings.json`](../src/sintergpu/data/timings.json) packaged in the Python wheel
and installed by CMake under `share/sintergpu/data`. It covers every current
catalog operation/variant. Pending rows identify missing workloads; they are not
zero-duration results. The four optional native vrdx variants appear separately with explicit pending
CPU/GPU baselines; their successful C++ build is not a timing measurement.
Current baseline coverage remains incomplete.

Each measurement retains raw **milliseconds**, iteration/warmup counts, median
and linearly interpolated p95, exact workload/input and shader/reference hashes,
compiler identity, CPU model/affinity/governor/frequency snapshot, GPU/driver/API,
software versions and collection time. Dependency-inclusive shader and reference
tree hashes prevent a changed include/helper from leaving old timings current.
The whole-tree check is conservative: an unrelated source addition can require
refreshing a baseline. Stale measurements remain in the JSON as history.

CPU reference wall time includes the actual reference call, including its
validation and output construction on prepared inputs. It is not an optimized
CPU-backend claim. GPU kernel timestamps bracket dispatch after compilation,
allocation and resource binding. Required mutable-output/state initialization
is outside isolated kernel time. Host submission/completion time is separate.
Transfer-inclusive time includes every input upload, required output/state
initialization, dispatch/completion and full final output/state readback. Do not
compare a GPU kernel-only number with a complete CPU-export path as an end-to-end
speedup. Unmeasured resident pipelines remain pending; pipeline measurements are
never assigned to their individual kernels.

The initial paired baseline uses all 28 existing Locus correctness fixtures for
17 operations, with 5 warmups and 50 samples per metric on the shared local
Radeon 8060S machine. Every GPU case checks full outputs/state against the frozen
scalar reference before and after sampling. These tiny fixtures establish a
reproducible reference workload, not production throughput. Larger recipes and
matching CPU references for all remaining operations are required work. The
selected NVIDIA workstation remains pending while unreachable. No timing row
selects an optimization or satisfies the three-run fusion acceptance gate.

```sh
# Standalone collection intentionally owns benchmark resources.
PYTHONPATH=src python benchmarks/locus_timings.py --iterations 50 --warmups 5 \
  --output cpu-timings.json
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/locus_timings.py --gpu \
  --iterations 50 --warmups 5 --output paired-timings.json
python tools/update_timings.py --input paired-timings.json
python tools/update_timings.py --check
sinter timings depth.bilateral
sinter timings --json
```

The update tool retains distinct workloads/environments and deduplicates identical
records. It does not choose the fastest run. Repeat collections are separate
records; raw samples support independent analysis. CPU-only collection does not
import a GPU runtime. Hardware GPU collection requires explicit opt-in and rejects
a software Vulkan adapter. Ordinary library recording still borrows the caller's
device/context and introduces no measurement, allocation or synchronization work.
