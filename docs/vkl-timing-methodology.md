# VKL timing baseline methodology

The standalone recipe measures each of the 13 non-feature VKL entries independently. The binary-feature recipe covers the other seven entries. The measurement recipes leave runtime ownership explicit. Release 0.2.0 also corrects camera-ray division after these new workloads exposed out-of-tolerance normals; that numerical change is versioned separately in the migration guidance.

Run from a source checkout with the optional GPU/test dependencies installed:

```sh
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. python -m benchmarks.vkl_timings \
  --gpu --iterations 50 --warmups 5 --output benchmarks/results/vkl-cpu-gpu-DEVICE-VERSION.json
python tools/update_timings.py --input benchmarks/results/vkl-cpu-gpu-DEVICE-VERSION.json
```

Omit `--gpu` for CPU-only measurements. GPU collection requires independent source/measurement clearance and explicit opt-in. The benchmark owns its device, buffers, allocations, submissions and readbacks; library adapters retain caller ownership.

The 38 seeded synthetic workloads cover two irregular multigroup sizes per entry, with optional photometric/world/confidence profiles, all three conversion representations, relative/metric fusion and temporal-selection optional fields. Initialization sanitizes raw depths; direct labeling consumes an established mask. Densification measures one step with an explicit direct mask. Pyramid, label and matching stage times are individual entries, not complete application pipelines. Inputs, output seeds/tails, types, launch extents and ordered bindings are hashed. The 8-MiB per-buffer fixture budget is a conservative test budget, not a reported hardware descriptor-range limit.

CPU wall time includes independent NumPy reference computation, scalar loops where present, temporary arrays and full output construction. These references prioritize correctness and independence; these timings cannot establish an optimized CPU speedup.

GPU timestamps bracket one entry after compilation, allocation, full output/state reset and warmup. Reset uploads and their completion precede the timestamp interval. Host submit/complete time is retained separately. Transfer-inclusive time uploads every bound buffer, including required output seeds, dispatches/waits and reads every complete output allocation. That standalone boundary deliberately exposes transfer overhead and is distinct from a resident multi-stage application path. Both isolated and transfer regions have warmups, and all complete outputs are checked before and after samples. Logical/integer outputs and untouched tails must match exactly; floating tolerances are the existing per-operation correctness-test tolerances.

Retain all raw samples, CPU/GPU/compiler/driver/library identities, independent reference/setup/helper/runner hashes, workload identity and uploaded/readback byte counts. Shared-device snapshots and small synthetic workloads do not qualify production throughput, trajectories, cross-vendor behavior, controlled fusion adoption or optimized-variant selection. The three existing strict expected failures for extreme depth-conversion denormal flushing remain pending qualifications; no tolerance is weakened by collecting these admitted fixtures.
