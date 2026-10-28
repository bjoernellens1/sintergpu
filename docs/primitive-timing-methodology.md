# Primitive stage timing methodology

Run from a source checkout with optional test/GPU dependencies:

```sh
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. python -m benchmarks.primitive_timings \
  --gpu --iterations 50 --warmups 5 --output benchmarks/results/primitives-cpu-gpu-DEVICE-VERSION.json
python tools/update_timings.py --input benchmarks/results/primitives-cpu-gpu-DEVICE-VERSION.json
```

Omit `--gpu` for CPU-only samples. The benchmark explicitly owns device, buffers,
allocation, submission, synchronization and readback. Library recipes/adapters
continue to borrow caller resources. Hardware execution requires independent
source/measurement clearance and coordination with other agents using the device.

The 61 fixtures exercise eight existing dispatch entries independently: float32
and uint32 scanBlocks, scanCarry and scanExclusive, and compactPrepare/Scatter.
Scan sizes255/256/257/70001 cover partial/exact/multiple workgroups, nondecreasing
segment IDs with gaps and boundaries, and long single segments. SUM/PRODUCT
include uint32 modulo wrap and zero factors. Compaction covers all/none/sparse
predicates, duplicate payload values, two extents and empty initialization.

Prepared stage inputs come from independent serial NumPy oracles, outside both
CPU and GPU measured regions. scanBlocks restarts its prefix at every256 lanes
and writes complete block summaries. Carry receives block-local prefixes and
independently scanned summaries, applies only matching-segment carry and preserves
all other seed values exactly. Exclusive receives an inclusive prefix and shifts
within segments using0/1 identity. Scatter receives valid uint inclusive flag
prefixes and preserves selected payload order. Empty prepare still dispatches one
256-thread group to initialize selectedCount; the complete empty scan recipe has
no dispatch. These are stage fixtures, not complete hierarchical scan timings.
For counts255/256, carry is a timed identity/no-write dispatch; larger fixtures
perform real carry updates. Existing end-to-end hierarchy/compaction correctness
tests remain separate probes.

CPU wall time includes actual oracle work, output construction/copies and temporary
arrays. It is not an optimized CPU alternative. GPU timestamps bracket a single
entry after allocation/compile/warmup, with all outputs/inout state reset before
every sample. Host submit/completion is separate. Transfer-inclusive time uploads
every bound buffer, including output seeds, dispatches/waits and reads all complete
output allocations. Both regions are warmed and parity checked before/after.
Integer/count/block-ID results, padding, undefined output capacity and untouched
carry values are exact; float live values use existing2e-4 absolute/relative bounds.
No kernel, float tolerance, synchronization or ownership contract is changed.

Workload identity hashes every uniform, ordered binding, launch extent, dtype,
input, complete output seed and defined live extent. Raw records bind source and
reference trees, the stage/reference/helper files, committed collector, actual
compiler and CPU/GPU/software environment. The8MiB per-buffer fixture budget is a
conservative bound, not an assertion about the hardware descriptor limit. Shared
snapshots do not establish exclusive-device performance, optimized selection,
production throughput, full-domain/cross-vendor operation qualification or gradients.
