"""Pinned micro-benchmark for the per-frame GPU hot path (MANDATORY
performance-refinement pass, data-path rule): nv12_to_rgb's fused kernel
latency at 720p/1080p, GPU-resident (HIP tensor in, HIP tensor out, no
host round trip -- see test_no_host_sync_guard.py for the static check
that enforces this structurally).

This is the real per-frame cost once Vulkan Video's zero-copy decode is
unblocked (see vulkan_backend.py) -- decode hands the colour kernel a
device-resident NV12 buffer, and this is what happens to it before a
consumer (the mapper, or a render loop) ever sees an RGB tensor. The spike
(RESULTS.md Sec 3d) measured 0.4954 ms/frame @720p inside
eligsir-mapper:from-source (ROCm 7.2.2) for the unfused predecessor of
this kernel; this test pins a looser bound (3x that) since CI/dev
containers vary in GPU contention, while still catching a real regression
(e.g. an accidentally reintroduced host sync would blow this bound by
10-100x, not 20%).

Requires a HIP-capable torch build and an actual GPU (skips otherwise --
this is explicitly a GPU-container test, never meant to run on bare host
per this repo's own documented host/ROCm-userspace mismatch risk, see
RESULTS.md Sec 3d "Risk").
"""
from __future__ import annotations

import time

import pytest
import torch

from eligsir_sim.video.colour import ColourTags, nv12_to_rgb, rgb_to_nv12

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a HIP/CUDA-capable torch build + GPU"
)

# Spike's own measured baseline (RESULTS.md Sec 3d), used only as a
# documented reference point in the assertion message, not re-derived here.
SPIKE_BASELINE_MS_720P = 0.4954
REGRESSION_MULTIPLE = 3.0

# Phase 9 (IMPLEMENTATION.md): fixed-point matrix-apply, uint8-output
# path. Measured (CUDA-event-timed, p50 of 200 warm iters, in-container
# on localhost/eligsir-sim:local-rocm-gsplat-ros): 720p 0.0526ms
# (78.80 GB/s), 1080p 0.1101ms (84.72 GB/s) -- vs the pre-Phase-9 fp64
# matrix-apply (same kernel, ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1):
# 720p 0.1624ms (25.54 GB/s), 1080p 0.3565ms (26.17 GB/s), matching
# Phase 7's independently-measured 24.82/25.35 GB/s baseline. These
# bounds pin the NEW (faster) behaviour with a 2x regression margin,
# same convention as SPIKE_BASELINE_MS_720P above.
FP9_BASELINE_MS_720P_UINT8 = 0.0526
FP9_BASELINE_MS_1080P_UINT8 = 0.1101
FP9_REGRESSION_MULTIPLE = 2.0


def _bench(w, h, iters=200, warmup=20):
    device = "cuda"
    tags = ColourTags(matrix="bt601", range="limited")
    y = torch.randint(16, 235, (h, w), dtype=torch.uint8, device=device)
    uv = torch.randint(16, 240, (h // 2, w // 2, 2), dtype=torch.uint8, device=device)

    for _ in range(warmup):
        nv12_to_rgb(y, uv, tags, out_dtype="float32")
    torch.cuda.synchronize()  # test-harness-only sync, bounding warmup; not in the hot path itself

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nv12_to_rgb(y, uv, tags, out_dtype="float32")
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    return p50, p95


def test_nv12_to_rgb_latency_720p_pinned():
    p50, p95 = _bench(1280, 720)
    bound = SPIKE_BASELINE_MS_720P * REGRESSION_MULTIPLE
    assert p50 < bound, (
        f"720p nv12_to_rgb p50={p50:.4f}ms exceeds {REGRESSION_MULTIPLE}x the spike's "
        f"measured baseline ({SPIKE_BASELINE_MS_720P}ms, RESULTS.md Sec 3d) -- bound={bound:.4f}ms"
    )
    assert p95 < bound * 2


def test_nv12_to_rgb_latency_1080p_pinned():
    p50, p95 = _bench(1920, 1080)
    # 1080p is ~2.25x the pixel count of 720p; bound scales accordingly.
    bound = SPIKE_BASELINE_MS_720P * REGRESSION_MULTIPLE * 2.25
    assert p50 < bound, f"1080p nv12_to_rgb p50={p50:.4f}ms exceeds bound={bound:.4f}ms"
    assert p95 < bound * 2


def _bench_uint8(w, h, iters=200, warmup=20):
    """Same methodology as _bench above but uint8 output (the actual
    bandwidth target this repo's brief measures against -- float32
    output moves 4x the bytes and stays on the unmodified fp64 path,
    see _colour_hip_ext.py's Phase 9 docstring) -- CUDA-event timed
    (not wall-clock perf_counter) since that is what every GB/s number
    in IMPLEMENTATION.md Phase 9/Phase 7 was measured with."""
    device = "cuda"
    tags = ColourTags(matrix="bt601", range="limited")
    y = torch.randint(16, 235, (h, w), dtype=torch.uint8, device=device)
    uv = torch.randint(16, 240, (h // 2, w // 2, 2), dtype=torch.uint8, device=device)

    for _ in range(warmup):
        nv12_to_rgb(y, uv, tags, out_dtype="uint8")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        nv12_to_rgb(y, uv, tags, out_dtype="uint8")
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def test_nv12_to_rgb_uint8_latency_720p_pinned_fixed_point():
    """Phase 9: pins the fixed-point fast path's uint8-output latency.
    Also asserts it actually beats the fp64 fallback-forced path by a
    healthy margin -- a regression that silently fell back to fp64 for
    every pixel (e.g. a guard-band bug that always returns "needs
    fallback") would still pass a latency-only bound if the bound were
    too loose; this comparison catches that specifically."""
    p50_fixed = _bench_uint8(1280, 720)
    bound = FP9_BASELINE_MS_720P_UINT8 * FP9_REGRESSION_MULTIPLE
    assert p50_fixed < bound, (
        f"720p uint8 nv12_to_rgb (fixed-point) p50={p50_fixed:.4f}ms exceeds "
        f"{FP9_REGRESSION_MULTIPLE}x the measured baseline "
        f"({FP9_BASELINE_MS_720P_UINT8}ms, IMPLEMENTATION.md Phase 9) -- bound={bound:.4f}ms"
    )

    import os

    os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"] = "1"
    try:
        p50_fp64 = _bench_uint8(1280, 720)
    finally:
        del os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"]
    assert p50_fixed < p50_fp64 / 1.5, (
        f"fixed-point path ({p50_fixed:.4f}ms) is not meaningfully faster than "
        f"the fp64-forced path ({p50_fp64:.4f}ms) -- the Phase 9 fast path may "
        f"not actually be taken (e.g. guard band always falls back)"
    )


def test_nv12_to_rgb_uint8_latency_1080p_pinned_fixed_point():
    p50 = _bench_uint8(1920, 1080)
    bound = FP9_BASELINE_MS_1080P_UINT8 * FP9_REGRESSION_MULTIPLE
    assert p50 < bound, (
        f"1080p uint8 nv12_to_rgb (fixed-point) p50={p50:.4f}ms exceeds bound={bound:.4f}ms"
    )


# Phase 10 (IMPLEMENTATION.md): fixed-point matrix-apply, ENCODE
# direction (rgb_to_nv12). Measured (CUDA-event-timed, p50 of 200 warm
# iters, in-container on localhost/eligsir-sim:local-rocm-gsplat-ros):
# 720p 0.1892ms (65.77 GB/s), 1080p 0.4077ms (68.67 GB/s) -- vs the
# fp64-forced path (ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1): 720p 0.2963ms
# (41.99 GB/s), 1080p 0.6529ms (42.88 GB/s). These bounds pin the NEW
# (faster) behaviour with a 2x regression margin, same convention as
# FP9_BASELINE_MS_720P_UINT8 above.
FP10_BASELINE_MS_720P = 0.1892
FP10_BASELINE_MS_1080P = 0.4077
FP10_REGRESSION_MULTIPLE = 2.0


def _bench_rgb_to_nv12(w, h, iters=200, warmup=20):
    """Same CUDA-event methodology as _bench_uint8 above, encode
    direction. RGB input is float32 [0,1], the actual rgb_to_nv12
    calling convention."""
    device = "cuda"
    tags = ColourTags(matrix="bt601", range="limited")
    rgb = torch.rand((h, w, 3), dtype=torch.float32, device=device)

    for _ in range(warmup):
        rgb_to_nv12(rgb, tags)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        rgb_to_nv12(rgb, tags)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def test_rgb_to_nv12_latency_720p_pinned_fixed_point():
    """Phase 10: pins the fixed-point fast path's rgb_to_nv12 (encode)
    latency, and asserts it beats the fp64-forced path by a healthy
    margin (catches a guard-band regression that silently always falls
    back -- see test_nv12_to_rgb_uint8_latency_720p_pinned_fixed_point's
    docstring for the same rationale, decode side)."""
    p50_fixed = _bench_rgb_to_nv12(1280, 720)
    bound = FP10_BASELINE_MS_720P * FP10_REGRESSION_MULTIPLE
    assert p50_fixed < bound, (
        f"720p rgb_to_nv12 (fixed-point) p50={p50_fixed:.4f}ms exceeds "
        f"{FP10_REGRESSION_MULTIPLE}x the measured baseline "
        f"({FP10_BASELINE_MS_720P}ms, IMPLEMENTATION.md Phase 10) -- bound={bound:.4f}ms"
    )

    import os

    os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"] = "1"
    try:
        p50_fp64 = _bench_rgb_to_nv12(1280, 720)
    finally:
        del os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"]
    assert p50_fixed < p50_fp64 / 1.2, (
        f"fixed-point path ({p50_fixed:.4f}ms) is not meaningfully faster than "
        f"the fp64-forced path ({p50_fp64:.4f}ms) -- the Phase 10 fast path may "
        f"not actually be taken (e.g. guard band always falls back)"
    )


def test_rgb_to_nv12_latency_1080p_pinned_fixed_point():
    """Same mutation check as the 720p test above -- the absolute bound
    alone is NOT sufficient at 1080p to catch a guard-band regression
    that always falls back to fp64 (the fp64-forced 1080p latency,
    ~0.653ms, is itself comfortably inside the 2x absolute bound,
    ~0.815ms -- an absolute-bound-only test would pass even with the
    fast path never taken), so the relative (>=1.2x speedup vs
    fp64-forced) check is required here too, not just at 720p."""
    p50 = _bench_rgb_to_nv12(1920, 1080)
    bound = FP10_BASELINE_MS_1080P * FP10_REGRESSION_MULTIPLE
    assert p50 < bound, (
        f"1080p rgb_to_nv12 (fixed-point) p50={p50:.4f}ms exceeds bound={bound:.4f}ms"
    )

    import os

    os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"] = "1"
    try:
        p50_fp64 = _bench_rgb_to_nv12(1920, 1080)
    finally:
        del os.environ["ELIGSIR_COLOUR_FORCE_FP64_MATRIX"]
    assert p50 < p50_fp64 / 1.2, (
        f"fixed-point path ({p50:.4f}ms) is not meaningfully faster than "
        f"the fp64-forced path ({p50_fp64:.4f}ms) at 1080p -- the Phase 10 fast "
        f"path may not actually be taken (e.g. guard band always falls back)"
    )


def test_nv12_to_rgb_allocation_free_steady_state():
    """Steady-state repeated calls must not grow HIP's allocator cache
    (pooled-buffer requirement) -- a per-call growth means a new
    allocation is happening every frame instead of reusing cached blocks."""
    device = "cuda"
    tags = ColourTags()
    w, h = 1280, 720
    y = torch.randint(16, 235, (h, w), dtype=torch.uint8, device=device)
    uv = torch.randint(16, 240, (h // 2, w // 2, 2), dtype=torch.uint8, device=device)

    for _ in range(10):
        nv12_to_rgb(y, uv, tags, out_dtype="float32")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(50):
        nv12_to_rgb(y, uv, tags, out_dtype="float32")
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after == before, (
        f"memory_allocated grew from {before} to {after} bytes over 50 steady-state "
        f"calls -- expected torch's caching allocator to reuse the same block sizes"
    )


def test_rgb_to_nv12_allocation_free_steady_state():
    """Phase 10 review addition: the mandatory per-repo performance-
    refinement pass requires an allocation-growth check for every
    data-path change; Phase 10 (encode direction) shipped
    test_nv12_to_rgb_allocation_free_steady_state (decode) but no
    rgb_to_nv12 (encode) equivalent -- added here. Mirrors the decode
    test above: steady-state repeated calls must not grow HIP's
    allocator cache."""
    device = "cuda"
    tags = ColourTags(matrix="bt601", range="limited")
    w, h = 1280, 720
    rgb = torch.rand((h, w, 3), dtype=torch.float32, device=device)

    for _ in range(10):
        rgb_to_nv12(rgb, tags)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(50):
        rgb_to_nv12(rgb, tags)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after == before, (
        f"memory_allocated grew from {before} to {after} bytes over 50 steady-state "
        f"rgb_to_nv12 calls -- expected torch's caching allocator to reuse the same "
        f"block sizes"
    )
