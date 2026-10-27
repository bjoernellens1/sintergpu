# Resident dispatch review and measured boundary

An independent worker reviewed `GaussNewton.dispatch` before GPU numerical
validation on 2026-10-05. The hot path passes the resident execution contract:

- Caspar generation and all four compiled compute kernels belong to construction.
- `prepare` performs input uploads and allocates residual, Jacobian, partials,
  update and solve-status buffers once. Repeated dispatches reuse them.
- A single encoder records residual/Jacobian, first-stage reduction, hierarchical
  reductions and Cholesky solve. Normal usage submits once; caller-owned encoders
  support batching multiple complete steps into one submission.
- No per-correspondence Python loop, buffer allocation, recompile, upload,
  numerical readback or host wait occurs in `dispatch`. Its sole explicit Python
  loop iterates over logarithmically many prepared reduction levels. Small host
  dictionaries and command recording remain; this is not an allocation-free API.
- Readback belongs exclusively to `readback`; `step` is a convenience API that
  prepares and reads back, and should not be used as the resident benchmark path.

`tests/test_hot_path.py` guards these properties using the dispatch AST. The
guard is intentionally local: it does not claim to inspect internal driver or
SlangPy allocations. A cached compile call is still excluded from resident dispatch.

## Measurement

`run_benchmark(size, iterations, reduction)` prepares one synthetic indexed
point-to-plane problem, warms five steps, waits once, and records the measured
steps in one command buffer. GPU timestamp pairs bracket every full step,
including dependencies between compute passes. Timestamp integers are converted
to milliseconds using `device.info.timestamp_frequency` (ticks per second).
`QueryPool.get_results` must not be interpreted as seconds.

Host wall time starts before recording and ends after submission and queue-idle
wait. It includes Python and submission costs and is separately reported,
amortized over the batch. Neither timing includes generation, compilation,
buffer allocation, upload, warmup, query retrieval or numerical readback.
Repeated steps use fixed inputs rather than iterating a changing pose. These are
kernel-pipeline measurements, not SLAM throughput or full optimizer latency.
If timestamp pools are unavailable, GPU timing is explicitly null with a reason;
device or shader failures still fail the benchmark.

The following measurements were recorded after this task's numerical GPU tests
had finished. They do not establish exclusive use of the workstation GPU by
unrelated processes. Raw samples and metadata are in
[`benchmark-amd-20261005.json`](benchmark-amd-20261005.json).

| Correspondences | Reduction | GPU median ms | Host wall ms/step |
| ---: | --- | ---: | ---: |
| 4099 | workgroup | 0.03585 | 0.10187 |
| 65537 | workgroup | 0.08925 | 0.15404 |
| 4099 | subgroup | 0.03691 | 0.10437 |
| 65537 | subgroup | 0.09815 | 0.16122 |

Each row uses 20 measured steps, one measured submission, five warmups, synthetic
seed 41, and default damping 0.001. Hardware: AMD Radeon 8060S Graphics
(RADV STRIX_HALO), Vulkan. SlangPy 0.43.1, SymForce 0.12.0, NumPy 2.5.3,
Python 3.13.12, Linux 7.2.6-200.fc44.x86_64, glibc 2.43. Device timestamp
frequency: 99,809,998 Hz. These samples favor the workgroup path on this device;
no universal subgroup advantage or CUDA comparison is claimed.

NVIDIA Vulkan versus Caspar CUDA, AMD discrete GPUs, and Intel qualification
remain unmeasured. Downstream RGB-D development proceeds while these comparisons remain pending.
