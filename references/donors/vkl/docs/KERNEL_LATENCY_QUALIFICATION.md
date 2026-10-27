# Deterministic kernel latency qualification

Tiny GPU kernels cannot be qualified from Python wall time or isolated single-event samples alone. When a kernel is in the single-digit or low-tens-of-microseconds range, framework dispatch, host launch gaps, event quantization, GPU scheduling, allocator activity, clock state, and profiler overhead can be comparable to or larger than device execution itself.

This document defines the latency evidence required before VKL publishes or uses a kernel-latency claim.

## Measurement questions

Different measurements answer different questions and must not be collapsed into one number.

| Metric | Meaning | Preferred mechanism |
| --- | --- | --- |
| `device_kernel_us` | GPU start-to-end execution duration of the actual kernel | CUPTI/Nsight Systems on CUDA; rocprofv3/ROCprofiler SDK on ROCm |
| `graph_amortized_us` | repeated captured launch cost without per-launch Python feeding gaps | CUDA/HIP graph replay of the exact implementation, divided by N |
| `native_dispatch_us` | ordinary runtime/driver launch spacing for a hand-written native implementation | native C++ CUDA/HIP repeated-launch harness |
| `framework_transaction_us` | complete framework-side wrapper/operator transaction | PyTorch benchmark; integration metric, never device-kernel latency |

Single-launch Python event timing is not sufficient qualification evidence for a microsecond kernel.

The required supporting timer depends on how the kernel is implemented:

- **generated/runtime-compiled kernels such as Triton:** use profiler device timestamps plus graph-amortized replay of that exact generated kernel where the backend supports it. Do not create a second hand-written CUDA/HIP implementation solely to obtain a native timer; that would measure different code.
- **hand-written CUDA/HIP kernels:** use profiler device timestamps plus native ordinary-launch timing, and add native graph replay when host feeding can starve the GPU.

## Why repeated GPU-event timing is still useful

A fixed-repeat interval is a useful diagnostic because it expands a very short event interval into a stable measurable duration:

```text
GPU event start
kernel x N
GPU event end

amortized time = elapsed / N
```

Choose N so the measured interval is long compared with event resolution and ordinary launch jitter. A useful initial target is 20--50 ms per interval with a bounded repeat count.

However, a Python loop around N launches is not the canonical kernel timer. If the GPU completes one tiny kernel before Python/PyTorch submits the next, GPU events include the resulting idle gaps. VKL reports this layer explicitly as `python_submitted_gpu_event_amortized` and marks `canonical_device_kernel_latency=false`.

Capturing the exact generated launch sequence into a GPU graph is different: it removes the per-launch Python feeding gap while retaining the same kernel implementation. Its result is `graph_amortized_us`, not single-launch wall latency.

## Native harness: only for native kernels

When VKL owns a hand-written CUDA/HIP kernel, provide a small native benchmark executable beside that implementation. The measured region must contain no Python, allocation, input generation, compilation, file I/O, or host-device transfer.

Ordinary-launch mode:

```text
allocate buffers once
initialize buffers once
warm up
synchronize

record start event
for N launches:
    launch exact native kernel
record end event
synchronize

native_dispatch_us = elapsed / N
```

Graph mode:

```text
record start event
launch graph containing native kernel x N
record end event
synchronize

graph_amortized_us = elapsed / N
```

A native harness is **not** required for a generated Triton kernel when constructing it would require a second implementation. In that case record `native_dispatch_us` as not applicable with an explicit reason and use profiler + exact graph replay instead.

## Profiler confirmation

A profiler trace must independently confirm the kernel-body duration and explain high-percentile tails.

### CUDA

Use Nsight Systems and/or CUPTI activity timestamps for:

- kernel start/end duration;
- host-to-device submission gaps;
- stream overlap;
- synchronization;
- context/scheduling gaps.

Use Nsight Compute only after latency has been established, for occupancy, memory traffic, instruction mix, stalls, cache behavior, and launch geometry. Detailed counter collection can perturb or replay kernels and should not be the headline latency source.

### ROCm

Use `rocprofv3` / ROCprofiler SDK for equivalent GPU-dispatch timestamps and timeline evidence. Retain timestamp-correction or scheduling warnings in the evidence record rather than hiding them in a mean. Detailed performance counters are diagnostic after basic device timing is established.

### PyTorch profiler

`torch.profiler` is useful to answer which PyTorch operation launched which GPU kernels and whether a supposedly single operator decomposes into many launches. It is not the authoritative timer for a 5--20 us kernel.

### Debuggers

`gdb`, `cuda-gdb`, and ROCm debuggers are correctness tools, not latency tools. Debugger-attached execution must not be used for performance claims.

## Standard qualification protocol

For reproducible kernel timing:

1. use a dedicated/quiet GPU where possible;
2. record GPU model, driver/runtime, PyTorch/Triton/compiler versions and exact source revision;
3. use fixed application clocks when permitted and record whether clocks were fixed;
4. warm the device and kernel before the measured phase;
5. allocate all buffers before canonical device timing;
6. use fixed input/output addresses, shapes, dtypes, launch geometry and stream where the implementation permits;
7. perform enough untimed warmup to exclude compilation/autotuning/lazy initialization;
8. for repeated controls, choose N so the interval is long relative to timer resolution;
9. record multiple independent passes, not only repeated samples in one process;
10. report median, p05, p95, minimum and MAD; retain mean/max only as secondary diagnostics;
11. collect an independent GPU timeline trace on the same workload shape;
12. keep cold-start, first-compiled-launch and steady-state results separate.

Allocator growth, compilation, autotuning, lazy initialization, host-device copies, decoder work, file I/O, and Python setup must be outside `device_kernel_us`. Wrapper allocation/dispatch belongs in `framework_transaction_us` instead of being silently ignored.

## Interpreting event tails

A result such as:

```text
median = 0.005 ms
p95    = 0.147 ms
```

must not be described as timer quantization without further evidence. A large high-percentile tail can originate from event resolution, submission gaps, GPU scheduling, context interference, clock state, allocator/lazy work, or true kernel variation.

The qualification wording should be:

> Single-launch event measurements are below the reliable scale for this benchmark and exhibit a large high-percentile tail. Repeated intervals and device timeline timestamps are required to separate device execution from launch/scheduling jitter.

If profiler device timestamps remain approximately 5 us while event p95 is approximately 147 us, the tail is external to kernel execution. If the profiler shows the same tail, investigate GPU-side scheduling, occupancy, memory behavior, or workload-dependent kernel execution.

## Required result schema

For each accepted optimized kernel, retain at minimum:

```text
device_kernel_us:
  median
  p05
  p95
  min
  mad

graph_amortized_us:
  status: measured | unsupported | not_applicable
  median/p05/p95/min/mad when measured

native_dispatch_us:
  status: measured | not_applicable
  reason when not applicable
  median/p05/p95/min/mad when measured

framework_transaction_us: optional integration metric
implementation_kind: triton | cuda | hip | other
repeat_count
warmup_count
sample_count
independent_pass_count
backend
GPU/device
clock policy
source revision
build/image revision
input shape/dtype
launch geometry
stream policy
profiler/tool version
```

Do not publish a generic `latency_us` field for tiny kernels without stating which timing layer it represents.

## Acceptance rule

For a microsecond **Triton/generated** kernel, VKL may call a result a kernel latency only when:

- profiler timestamps measure the actual generated kernel on the target GPU;
- repeated independent passes establish a stable central device scale;
- graph-amortized replay of the exact kernel is used where supported to separate Python feeding jitter;
- numerical parity and workload shape are unchanged;
- AMD and NVIDIA results use equivalent semantic workloads;
- framework transaction cost is measured separately before pipeline integration when wrapper overhead can be material.

For a microsecond **hand-written CUDA/HIP** kernel, additionally require native ordinary-launch timing; add native graph replay when host feeding can starve the GPU.

The Python fixed-repeat timing mode remains useful as a diagnostic/regression signal but is explicitly non-canonical for device-kernel latency.

Issue #56 was closed as not planned for the current Triton candidate because a duplicate native implementation would invalidate the measurement target. Reopen it or create a kernel-specific native-timing issue when VKL introduces a hand-written CUDA/HIP kernel.
