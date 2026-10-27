# Triton qualification

This document records benchmark findings required before a reference kernel is
replaced by a Triton implementation. Generated JSON artifacts remain ignored
under `results/triton/`; this file records the reproducible command, immutable
source/image identities, workload, and decision.

## 2026-09-02 inventory baseline

Source revision: `3686d5da95182651ea8010efcce0074a274a182f`.

Workload: every public reference export plus `select_best_inverse_depth`, with
deterministic `8x8` controls and batches B1, B2, B4, and B8. Each row used two
warmups and five timed GPU-event iterations. The report records complete
per-control tensor and algorithm dimensions. `compiled` means a
`torch.compile` request that returned; it is deliberately labelled
`compiled_unverified` rather than claimed as an inspected single compiled
graph.

| Backend | Device | Runtime image | Image revision | Artifact |
| --- | --- | --- | --- | --- |
| AMD ROCm 7.2 | Radeon 8060S Graphics | `localhost/vkl:dev` | `17bdae9302e75a88ccac9c0ab684563cb39ca607` | `results/triton/baseline_amd_gfx1151_final.json` |
| NVIDIA CUDA 12.8 | NVIDIA GeForce RTX 4090 | `localhost/vkl:cuda-triton-3686d5d` | `3686d5da95182651ea8010efcce0074a274a182f` | `results/triton/baseline_rtx4090_final.json` |

AMD command:

```text
VKL_SOURCE_REVISION=3686d5da95182651ea8010efcce0074a274a182f \
VKL_IMAGE_REVISION=17bdae9302e75a88ccac9c0ab684563cb39ca607 \
VKL_IMAGE_REFERENCE=localhost/vkl:dev \
podman compose run --rm \
  -e VKL_SOURCE_REVISION=3686d5da95182651ea8010efcce0074a274a182f \
  -e VKL_IMAGE_REVISION=17bdae9302e75a88ccac9c0ab684563cb39ca607 \
  -e VKL_IMAGE_REFERENCE=localhost/vkl:dev \
  shell python recipes/triton/benchmark_inventory.py --device cuda \
  --output results/triton/baseline_amd_gfx1151_final.json
```

NVIDIA command used the same recipe in the CUDA container with Docker GPU
access and the image identity shown above.

### Result

All 172 eager rows and all 172 compile-request rows completed on both GPUs.
The 168 reference-only rows per GPU correctly report Triton as unsupported.
The four inverse-depth argmin rows dispatched actual Triton and matched the
independent eager `torch.topk` reference exactly, but are slower on this tiny
control:

| Device | B1 | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: |
| Radeon 8060S, Triton/eager p50 | 1.43x | 1.42x | 1.52x | 1.58x |
| RTX 4090, Triton/eager p50 | 1.63x | 1.63x | 1.24x | 1.49x |

No row is accepted or used to replace a reference implementation. The argmin
kernel remains a correct, non-winning candidate; the next candidate is a
measured fusion boundary with enough work to amortize launch overhead, starting
with project/depth evidence.

### Complete public-family triage

The baseline artifacts above cover every public reference export at B1/B2/B4/B8
on both vendors. `recipes/triton/benchmark_inventory.py:classifications()` is
test-backed against `reference.__all__`; the following is the durable reviewed
disposition, not an inference from an unsupported Triton row.

| Disposition | Kernels | Reason |
| --- | --- | --- |
| candidate | `project_depth_evidence` | Measured fixed-capacity fusion; it is now qualified in the section below, pending only a Splatograph transaction gate. |
| deferred | `aggregate_depth_cost_volume`, `dense_pose_depth`, `direct_pose_depth_dso`, `direct_pose_depth_tensorized` | Dynamic depth-search/cost-volume work must not be materialized or custom-ported speculatively. |
| deferred | `rasterize_reprojection` | Visibility rasterization is backend-specialized scatter semantics. |
| deferred | `select_compact`, `spatial_bin_reduce` | Stable ordering and lexicographic `unique`/reduction are dynamic semantics. |
| retain_vendor | `camera_rectify`, `canonical_rescale`, `depth_pyramid`, `resolution_pyramid` | Vendor sampling/pooling owns the required interpolation semantics. |
| retain_compile | `align_depth`, `depth_convert`, `depth_geometry`, `fuse_depth`, `temporal_depth_fuse`, `triangulate_depth`, `world_transform` | Straight tensor reductions or small linear algebra; no measured fixed fusion boundary. |
| retain_compile | `depth_evidence`, `keypoint_evidence`, `rgbd_evidence`, `residual_evidence` | Evidence reductions need transaction profiling; the per-pixel-atomic residual probe was rejected below. |
| retain_compile | `depth_register`, `depth_reproject`, `reproject_rgbd`, `temporal_frame_compare`, `temporal_reproject_rgb`, `temporal_restore_rgb`, `temporal_restore_rgbd` | Composed vendor sampling/reprojection semantics. |
| retain_compile | `dso_select_points`, `fast_select`, `pose_observable_select`, `shi_tomasi_select`, `spatial_select_points`, `uniform_select` | NMS/top-k/spatial selection policy rather than a fixed compute boundary. |
| retain_compile | `keyframe_filter`, `landmark_update`, `temporal_best_observation`, `validate_frame_batch` | Pipeline policy, state/identity, or boundary validation is intentionally not a Triton target. |
| retain_compile | `lk_track`, `photometric_isp` | Respectively vendor sampling/iterative tracking and autograd-sensitive photometry; both require dedicated qualification before any fusion. |

`select_best_inverse_depth` remains a correct but non-winning existing Triton
candidate: it is slower than eager on the `8x8` inventory control and is not
accepted. This table does not make a performance claim for any retained path;
it records why no Triton implementation should be added before a new measured
transaction boundary appears.

## 2026-09-02 project/depth evidence fusion candidate

Source revision: `2bcd6f0c9f671ea6bc5090ca99d4cc78745e36a2` for the
corrected kernel; timing controls at
`07d9bd3a7fce29333fe1c6961b53523f791a57f3`.

This candidate fuses point eligibility, world-to-camera transform, projection,
nearest-pixel sensor-depth lookup, residual, and support/conflict classification
for 100,000 points per batch against a 640x480 metric depth image. Inputs use
the existing prevalidated transaction-boundary contract; the defensive public
path remains the PyTorch reference.

Correctness qualification is exact for B1/B8 on both GPU vendors, including an
unaligned tail, non-finite projection, fp16 epsilon behavior, and the public
structural/type/threshold contract. The focused GPU suite is:

```text
pytest -q tests/test_triton_project_depth_evidence.py
```

### Timing tiers

Do not use one-launch Python GPU-event samples as a device-kernel latency claim
below 20 microseconds. They are a framework diagnostic only: host submission,
event resolution, and scheduling gaps can dominate. Qualification records three
distinct quantities:

| Metric | Meaning | Acceptance use |
| --- | --- | --- |
| PyTorch event diagnostic | Python-submitted transaction interval | diagnose framework cost only |
| graph-amortized | exact captured Triton launches divided by repeat count | device-feeding lower bound, not single-launch latency |
| profiler device duration | GPU start-to-end timestamp for the actual kernel | required latency evidence |

The fixed-repeat diagnostic is available through
`--calls-per-sample`; each GPU-event sample surrounds N launches and is divided
by N. It is intentionally not the canonical number for a very short kernel.

### Compiler qualification is separate from timing

`torch.compile` is a compiler-path qualification, not an execution-timing
fallback. Run it alone through `recipes/triton/compile_gate.py`, which records
`compile_status` (`completed`, `timeout`, or `failed`),
`qualification_wall_s`, `cold_qualification_wall_s`, `compiler_cache_state`,
and the absence of a warmed
measurement when compilation did not finish. The gate has a 300-second default
and runs exactly one child process, so concurrent cold compilers cannot
contend for CPU, cache, or temporary storage.

```text
python recipes/triton/compile_gate.py --timeout-s 300 --cache-state cold_empty \
  --output results/triton/project_depth_compile_rocm_cold.json -- \
  python recipes/triton/bench_project_depth_evidence.py \
  --variants compiled --output results/triton/project_depth_compiled_amd.json
```

The gate records compiler state only; the inner benchmark records any warmed
execution values after it becomes executable. A timeout is a compiler-path
limitation, not a kernel failure and not a zero-latency result.

The completed Radeon 8060S project/depth B1 cold control at source revision
`db5da60` is distinct from the rejected residual probe. With
`compiler_cache_state=cold_empty`, the gate completed in **9.759 s**. Its
separate inner B1 warmed event diagnostic (five warmups, ten samples) was
**0.339 ms p50**; this remains a Python-submitted GPU-event transaction value,
not a kernel-body latency claim. The generated records are
`results/triton/project_depth_compile_rocm_cold.json` and
`results/triton/project_depth_compiled_amd.json`. The runtime report identifies
ROCm 7.2 / PyTorch `2.10.0+rocm7.2.2.git23d69b29` on Radeon 8060S.

#### B1/B8 shape and persistent-cache control

The B1 result alone did not classify B8 or real cache reuse. At source revision
`89559dd`, the bounded gate was therefore run sequentially for B1 and B8 with
separate, initially absent cache directories under `results/triton/`. The
workspace location is important: `/tmp` is destroyed with each disposable
container and cannot be labelled a persistent-cache experiment. Each command
used a 60-second gate, five benchmark warmups and ten timed calls after
compilation; the recorded gate wall includes that small post-compile benchmark
tail and is not a compiler-internal phase breakdown.

| Batch | cold-empty gate wall | warm-persistent gate wall | inner compiled event p50 |
| ---: | ---: | ---: | ---: |
| B1 | 9.890 s | 6.565 s | 0.343 / 0.367 ms |
| B8 | 11.044 s | 6.988 s | 0.320 / 0.343 ms |

All four commands completed, had less than 8 microseconds of gate-lock wait,
and produced an inner `compiled_unverified` execution record. The event values
remain Python-submitted transaction diagnostics, not device-kernel duration.
This classifies the full-resolution project/depth reference on this ROCm image
as a bounded reusable per-shape cold cost rather than a reproducible B8
compiler-path limitation. It does not establish that every larger or dynamic
shape has the same behavior; those require a new gated experiment.

Generated records:

```text
results/triton/project_depth_compile_rocm_b1_cold_persistent_cache.json
results/triton/project_depth_compile_rocm_b1_warm_persistent_cache.json
results/triton/project_depth_compile_rocm_b8_cold_persistent_cache.json
results/triton/project_depth_compile_rocm_b8_warm_persistent_cache.json
```

### Three-pass profiler evidence

NVIDIA used Nsight Systems and AMD used ROCprofv3. Each independent pass runs
100 untimed warmups and 100 timed Triton invocations; the trace therefore
contains 200 actual candidate-kernel dispatches. Profiler timing is the GPU
start-to-end duration of `_project_depth_evidence`, not a Python wall clock.

NVIDIA GeForce RTX 4090, CUDA 12.8, PyTorch `2.10.0+cu128`, image
`localhost/vkl:cuda-triton-3686d5d`
(`sha256:bbcb99071d81cb4491be10129edc245741c68307c237d5ea451f588a4bb45f52`).
Nsight Systems ran inside the container because profiling the outer Docker
client captures no CUDA activity. The trace executed 100 untimed warmups then
100 B1 or B8 Triton invocations.

| Device | B | pass 1 median / mean / max (us) | pass 2 median / mean / max (us) | pass 3 median / mean / max (us) |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 1 | 3.328 / 3.342 / 5.920 | 3.328 / 3.353 / 6.464 | 3.328 / 3.338 / 6.304 |
| RTX 4090 | 8 | 13.312 / 13.465 / 29.088 | 13.056 / 13.148 / 28.896 | 12.480 / 12.675 / 29.825 |
| Radeon 8060S | 1 | 7.770 / 8.896 / 46.347 | 7.817 / 310.841 / 58615.787 | 7.708 / 9.590 / 44.551 |
| Radeon 8060S | 8 | 62.074 / 121.429 / 6095.946 | 72.227 / 113.626 / 4971.880 | 70.297 / 70.611 / 239.244 |

The AMD central values are stable, but its profiler recorded rare scheduling or
timestamp outliers (including one 58.6 ms B1 instance). Report per-pass
medians and p95s for AMD rather than a mean-only result:

| Radeon 8060S B | pass 1 p95 (us) | pass 2 p95 (us) | pass 3 p95 (us) | instances over 100 us |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.007 | 17.459 | 31.510 | 0 / 2 / 0 |
| 8 | 71.534 | 92.075 | 84.069 | 4 / 4 / 2 |

The B8 ROCprofv3 run reported two corrected inverted dispatch timestamps. That
warning is retained as an environment caveat; it does not alter the stable
median range but prevents interpreting max/mean as intrinsic kernel cost.

The CUDA graph check captures 128 exact Triton launches and divides the event
duration by 128. It removes Python gaps but is **not** a single-launch latency:

| RTX 4090 B | graph-amortized p50 (us) | graph-amortized p95 (us) |
| ---: | ---: | ---: |
| 1 | 3.240 | 6.064 |
| 8 | 15.560 | 20.568 |

The profiler and graph results are consistent. A previous single-launch CUDA
event B1 median of 5.120 us with p95 about 147 us is therefore rejected as a
latency result: it contains framework/scheduling variance rather than showing
device-kernel variance.

### Rejected Radeon 2-D launch tuning

Issue #58 tested a vendor-neutral two-dimensional launch before considering
any AMD-specific implementation. The experiment used one program dimension
over points and a second dimension over batch, eliminating the flattened
kernel's per-lane `offset // points` and preventing a block from spanning two
batches. It was a short-lived benchmark-only variant and is deliberately not
retained in the public wrapper.

On Radeon 8060S / ROCm 7.2, the B1/B2/B4/B8 event screen used 100,000 points
per batch, 640x480 depth, 30 warmups, 80 samples, and 32 calls per sample. It
selected 2-D `BLOCK=256`, `num_warps=8` only for profiler confirmation; these
event values are a screening control, not latency evidence:

| Layout/configuration | B1 | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: |
| flattened, 256/4 | 39.135 us | 38.833 us | 34.308 us | 66.321 us |
| 2-D, 256/8 | 38.804 us | 39.000 us | 34.605 us | 62.921 us |

The same workload was traced with ROCprofv3 using 100 warmups and 100 timed
dispatches per batch. Profiler device timestamps reject the apparent B8 event
improvement:

| Layout/configuration | B1 p50 | B2 p50 | B4 p50 | B8 p50 |
| --- | ---: | ---: | ---: | ---: |
| flattened, 256/4 | 7.478 us | 13.062 us | 26.898 us | **64.042 us** |
| 2-D, 256/8 | 6.346 us | 10.825 us | 28.940 us | **73.968 us** |

The 2-D candidate improves B1/B2 but regresses B4 and B8; B8 is 15.5% slower
than the retained flattened launch. ROCprofv3 emitted corrected inverted
timestamp warnings in both runs, so means/maxima are not used for this
decision. Exact output parity held for B1/B2/B4/B8. The simple flattened
`BLOCK=256`, `num_warps=4` kernel remains the retained configuration.

Generated screening and trace artifacts are ignored under `results/triton/`:

```text
podman compose run --rm --entrypoint bash shell -lc \
  'rocprofv3 --kernel-trace --output-format json \
   -d results/triton/project_depth_amd_flat_rocprof_pass1 -- \
   python recipes/triton/bench_project_depth_evidence.py \
   --device cuda --batches 1,2,4,8 --points 100000 \
   --warmup 100 --iterations 100 --calls-per-sample 1 --variants triton \
   --triton-layout flat --triton-block 256 --triton-num-warps 4 \
   --output results/triton/project_depth_amd_flat_rocprof_pass1_inner.json'
```

Use the same command with `--triton-layout 2d --triton-num-warps 8` and the
`project_depth_amd_2d_b256_w8_rocprof_pass1` output path to reproduce the
rejected comparison. A future configuration must improve B8 under profiler
timestamps and preserve RTX control results before it replaces the current
launch.

Separate legal ROCprofv3 counter passes then isolated B1 from B8. Median
counter values were: occupancy 28.8% / 82.2%, L2 hit rate 54.6% / 52.1%, read
traffic 1,371.5 / 12,548.5 KB, and write traffic 111.1 / 3,249.4 KB. Thus B8
raises occupancy rather than exposing an underfilled launch, retains comparable
L2 behavior, and scales traffic with the workload (especially six output
writes). Together with the rejected 2-D configuration, this is a negative
resource-scaling result, not evidence for an AMD-specific layout.

## 2026-09-02 rejected residual/cell reduction probe

Source revision: uncommitted short-lived probe on the project/depth evidence
worktree; no wrapper was retained. This negative result is documented because
it rules out a tempting but incorrect fusion strategy.

The candidate wrote the existing RGB-D residual maps and used one global
atomic add per pixel for frame and fixed `12x16` tile sums. It preserved the
reference result at B1/B2/B4/B8, including invalid-depth masking, but is not a
valid optimization. On Radeon 8060S, ROCm 7.2, at 640x480 RGB-D inputs,
five warmups and ten single-call GPU-event samples produced:

| B | eager p50 (ms) | per-pixel-atomic Triton p50 (ms) | Result |
| ---: | ---: | ---: | --- |
| 1 | 0.384 | 1131.206 | reject |
| 8 | 3.389 | 3013.867 | reject |

Parity maximum absolute error was `2.78e-7` (B1) and `2.44e-7` (B8), with no
mismatches under the qualification tolerance. The slowdown is therefore not a
numerical fallback: all pixels contend on the same frame/tile atomic locations
on the shared-memory AMD GPU. The reference path remains authoritative.

Do not retry this design. A future candidate must give a program exclusive
ownership of a complete tile (local reduction, one tile write) and benchmark
the required second frame-reduction stage as part of the transaction. It must
still meet the dual-vendor >=10% transaction gate before any wrapper is kept.

Exact commands and generated artifacts:

```text
# RTX profiler trace: run nsys inside the CUDA container, not around docker.
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  -v "$PWD:/workspace" -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -w /workspace localhost/vkl:cuda-triton-3686d5d \
  /opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys profile \
  --trace=cuda --sample=none --force-overwrite=true \
  -o results/triton/project_depth_evidence_rtx4090_trace_b1 \
  python recipes/triton/bench_project_depth_evidence.py \
  --height 480 --width 640 --points 100000 --batches 1 \
  --warmup 100 --iterations 100 --variants triton \
  --output results/triton/project_depth_evidence_rtx4090_trace_b1.json

# Report actual device timestamps.
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  results/triton/project_depth_evidence_rtx4090_trace_b1.nsys-rep

# AMD ROCprofv3 trace inside the normal privileged ROCm container.
podman compose run --rm -T -e HSA_XNACK=1 -e HSA_ENABLE_SDMA=0 shell \
  bash -lc '/opt/rocm/bin/rocprofv3 --kernel-trace --stats --summary \
    --summary-units usec \
    --output-directory results/triton/project_depth_evidence_amd_trace_b1 -- \
    python recipes/triton/bench_project_depth_evidence.py \
    --height 480 --width 640 --points 100000 --batches 1 \
    --warmup 100 --iterations 100 --variants triton \
    --output results/triton/project_depth_evidence_amd_trace_b1.json'

results/triton/project_depth_evidence_rtx4090_trace_b1.nsys-rep
results/triton/project_depth_evidence_rtx4090_trace_b8.nsys-rep
results/triton/project_depth_evidence_rtx4090_trace_b1.json
results/triton/project_depth_evidence_rtx4090_trace_b8.json
results/triton/project_depth_evidence_amd_trace_b1/
results/triton/project_depth_evidence_amd_trace_b8/
results/triton/project_depth_evidence_amd_trace_b1.json
results/triton/project_depth_evidence_amd_trace_b8.json
```

For a future hand-written CUDA/HIP kernel, add a native ordinary-launch and
native graph-replay harness beside that source. A native harness cannot launch
the generated Triton kernel without maintaining a second implementation, so
the actual Triton candidate uses graph replay plus profiler timestamps instead.
The candidate is not accepted for Splatograph integration until its
transaction-level benchmark is measured.

### Project-depth wrapper transaction and output ownership

Revision `dac930b` adds an opt-in caller-owned `out` mapping and the
`bench_project_depth_transaction.py` recipe. It measures already-device-resident,
contiguous B1/B2/B4/B8 inputs at 640x480 with 100,000 points, 30 warmups, 100
GPU-event samples, and 32 calls per sample. It deliberately excludes H2D,
input construction, validation, and compiler startup. The metric is a
Python-submitted GPU-event transaction diagnostic, not device-kernel latency.

Each fresh call allocates six output tensors. Fresh allocation volume is 0.90,
1.80, 3.60, and 7.20 MB for B1, B2, B4, and B8 respectively. On Radeon 8060S
(ROCm 7.2 / PyTorch 2.10.0), reuse reduced p50 from 35.24 to 26.48 us at B1,
32.14 to 27.23 us at B2, and 35.26 to 29.12 us at B4, but regressed from 64.42
to 71.35 us at B8. On the idle Ablator-registered RTX 4090c (CUDA 12.8 /
PyTorch 2.9.1), reuse consistently reduced p50 by about 7.3 us: B1 33.47 to
26.11 us, B2 33.57 to 26.12 us, B4 33.31 to 26.05 us, B8 33.68 to 26.37 us.

Retain the simple fresh-allocation default and expose reuse only through the
explicit caller-owned mapping: it is beneficial on NVIDIA and some AMD batch
sizes, but not a universal fast path. The regression test proves a later
invalid transaction overwrites all supplied buffers rather than exposing stale
values. Generated evidence: `results/triton/project_depth_transaction_amd.json`
and `results/triton/project_depth_transaction_nvidia.json`.

### Environment repair discovered by the matrix

The first RTX run showed `torch.compile(depth_convert)` failing with
`InvalidCxxCompiler: No working C++ compiler found`. The CUDA runtime image
configured Inductor to use `g++` but did not contain it. Revision `3686d5d`
adds `build-essential`; the exact compiled `depth_convert` reproduction then
completed and the final 172-row NVIDIA matrix contained no compile errors.
