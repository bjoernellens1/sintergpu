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

The initial raw dataset was collected with runner revision
`1890894c70ef87ed7b6c29a4e356c38c5231f20a`; its recorded runner hash identifies
that collector. The subsequent optional-driver-diagnostic fallback changes no
measured region or shader/reference inputs.

## Camera-color workloads

The color baseline adds 30 workloads across all six camera-color catalog entries.
Twenty-four use padded 6×4 inputs across BT.601/BT.709 and limited/full range,
with odd 7×5 remap/resize outputs. Six use padded 640×480 inputs in BT.709 limited
range: decode/encode outputs retain the full input extent, while remap, resize
and fused decode/remap produce 64×48 outputs. Synthetic maps include nonfinite
coordinates. These dimensions and pitches are part of each workload identity;
a 64×48 output is not a full VGA-output benchmark.

CPU samples include independent NumPy float64 reference computation, validation
and seeded-output copies; resize also includes reference map construction. GPU
parity checks every live byte against the existing 1-LSB float64-oracle contract,
and every padding byte exactly. Allocation/compilation and output resets stay
outside isolated kernel timing. Transfer timing includes every input/map/output
seed upload and both output-plane readbacks for NV12 encode. Its transfer path
has independent warmups. Driver diagnostics are optional; exact compiler and
hardware identities remain required.

```sh
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. python benchmarks/color_timings.py --gpu \
  --iterations 50 --warmups 5 --output color-paired-timings.json
python tools/update_timings.py --input color-paired-timings.json
```

The initial color run uses collector revision `eb661d9` and retains its full runner
hash plus the imported CPU sampling helper hash. It passes all numerical/padding
checks on Radeon 8060S. This increases paired coverage to 23/130 portable entries;
107 portable entries and four optional native variants still need baselines.
The new run does not replace existing Locus samples or promote any optimized path.
