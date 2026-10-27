"""Fused HIP kernels for NV12/P010 <-> RGB colour conversion, JIT-compiled
once per process via ``torch.utils.cpp_extension.load_inline`` and cached on
disk across processes (``EXT_BUILD_DIR`` below) so a later container run
reuses the compiled ``.so`` instead of paying hipcc's build cost again --
same amortization strategy ``isp.py`` uses for ``torch.compile``/Inductor
(see ``TORCHINDUCTOR_CACHE_DIR`` there), applied to a hand-written extension
instead of a traced graph.

Technology choice (brief: "pick with evidence"): a hand-written HIP kernel
via ``load_inline``, not Triton or ``torch.compile``.

  - Triton 3.6.0+rocm7.2.2 *is* present in the target image
    (localhost/eligsir-sim:local-rocm-gsplat-ros), so it was not ruled out
    for availability. It was ruled out for *this* kernel because the
    decode-path requirement (read Y/UV through an explicit, caller-supplied
    row pitch so the D2D de-pitch copy can be dropped later, see
    vulkan_backend.py's "PHASE" notes) needs precise control over the
    per-thread address computation and the exact rounding mode
    (round-half-to-even, to stay bit-identical with
    ``colour.nv12_to_rgb_numpy``'s ``np.round``) that is far more direct to
    express and audit in a ~40-line HIP kernel than to coax out of a Triton
    block-pointer / rounding-mode combination -- and RDNA3 (gfx1151) Triton
    codegen is newer/less exercised in upstream CI than CDNA, so a
    hand-written kernel is also the lower-risk choice for a first fused
    pass.
  - ``torch.compile``/Inductor was benchmarked for comparison (see
    tests/video/test_colour_kernel_benchmark.py's ``compile`` variant
    results in IMPLEMENTATION.md): it fuses the existing elementwise chain
    into fewer kernels but still materializes the ``repeat_interleave``
    upsample and a separate stack/clamp, so it cannot reach the
    one-thread-per-2x2-block memory-traffic reduction a hand-written kernel
    gets "for free" by never materializing the upsampled chroma plane at
    all. Measured numbers are in IMPLEMENTATION.md's "Fused colour kernels"
    section.

Every exported function here is called only from colour.py's dispatch
(nv12_to_rgb/p010_to_rgb/rgb_to_nv12), which falls back to the pure-torch
reference implementation (_*_torch_ref) if this extension fails to build
(e.g. CPU-only dev container, or no hipcc) -- the torch version is kept
indefinitely as the documented reference, never deleted.
"""
from __future__ import annotations

import os
import threading

_EXT_BUILD_DIR = os.path.expanduser("~/.cache/eligsir-sim/colour_hip_ext")

_lock = threading.Lock()
_ext = None
_ext_load_error: Exception | None = None

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------

// Round-half-to-even in double precision, matching numpy's np.round /
// torch's .round() (both IEEE-754 round-to-nearest-even), then clamp to
// the [0,255] uint8 range. `rint` respects the default FP rounding mode
// (round-to-nearest-even) on both CUDA and HIP.
__device__ __forceinline__ unsigned char to_u8(double v) {
    v = fmin(fmax(v, 0.0), 255.0);
    return static_cast<unsigned char>(rint(v));
}

// ---------------------------------------------------------------------
// Phase 9: fixed-point (int64) matrix-apply, uint8-output path only.
//
// Root cause (Phase 7's elimination experiments, IMPLEMENTATION.md): the
// uint8-output kernel's cost is dominated by fp64 ALU throughput in the
// matrix-apply (`r=yf+a*vf`, `g=yf-c*uf-d*vf`, `b=yf+b*uf`) and to_u8's
// fmin/fmax/rint chain -- gfx1151 (RDNA3.5) has crippled native fp64
// throughput vs int32/int64, which run at full rate. This replaces that
// double arithmetic with int64 fixed-point (Q=2**24 for the a/b/c/d and
// lscale/cscale coefficients, Q4=16 i.e. 4 extra bits for the raw-sample
// domain -- matching the chroma ladder's <=4-fraction-bit blended output
// exactly, see fetch_uv_raw's docstring above) for the OVERWHELMING
// majority of pixels, falling back to the proven-bit-exact double
// formula (same op order as the torch/numpy reference, same
// -ffp-contract=off build flag) for the rare pixel whose fixed-point
// value lands within a certified guard band of an exact half-integer
// rounding boundary.
//
// Why a guard band, not a bigger integer (derive_guardband.py /
// IMPLEMENTATION.md Phase 9 "Why not just use more bits"): the fp64
// reference's OWN double-rounding chain (each of its ~4 operations
// rounds to the nearest representable double) occasionally produces a
// result that is itself off by ~1e-14 from the true mathematical value,
// and -- at the small set of inputs where the true value is an EXACT
// half-integer tie -- that tiny reference-side bias, not any finite
// fixed-point precision, decides which way the final round() goes.
// No power-of-two (or exact-rational) fixed-point scheme can reproduce
// that per-operation IEEE-754 rounding chain without re-implementing
// double arithmetic bit-for-bit in integer code (defeating the entire
// point of moving off the slow fp64 units). Verified empirically
// (derive_guardband.py, run at the host-side derivation step, BEFORE
// writing this kernel code, exactly as the brief required): across the
// full exhaustive 256^3 (Y,U,V) grid x 4 (matrix,range) combos, dense
// 4M-sample sweeps of fractional ("interpolated" ladder, denominator-16)
// chroma, and 4M-sample P010 10-bit sweeps, a guard band of width
// W = 2*(8422*2**-(S+1) + 1e-11) at S=24 produces ZERO pixels outside
// the band whose fast int64 rounding disagrees with the fp64 reference,
// at a fallback rate of ~0.02%-0.3% of pixels depending on channel/
// domain (g is worst; see IMPLEMENTATION.md Phase 9 for the full table).
//
// `w_scaled` (passed from the host, already scaled by `den` =
// Q4*SCALE*SCALE = 2**52) and the int64 coefficients are precomputed
// once per `nv12_to_rgb`/`p010_to_rgb` call (per-stream ColourTags), not
// per-pixel -- this is a Python-side (colour.py) derivation, not
// per-thread work.
#define FP9_SCALE_SHIFT 24
#define FP9_Q4_SHIFT 4
#define FP9_DEN_SHIFT (FP9_Q4_SHIFT + 2 * FP9_SCALE_SHIFT)  // 52

// Returns true if the fast fixed-point value is OUTSIDE the guard band
// (trustworthy): `*out_u8` is then the final uint8 value. Returns false
// if this pixel/channel needs the double fallback (caller must then
// compute it via the existing double formula).
__device__ __forceinline__ bool to_u8_fixed_or_fallback(
    long long channel_fixed, long long w_scaled, unsigned char *out_u8
) {
    const long long den = (long long)1 << FP9_DEN_SHIFT;
    long long clamped = channel_fixed < 0 ? 0 : (channel_fixed > 255 * den ? 255 * den : channel_fixed);
    long long rem = clamped & (den - 1);            // den is a power of two
    long long signed_dist = rem - (den >> 1);
    long long dist = signed_dist < 0 ? -signed_dist : signed_dist;  // distance to the nearest half-integer boundary
    if (dist < w_scaled) {
        return false;  // ambiguous -- caller must fall back to the double path
    }
    long long rounded = (clamped + (den >> 1)) >> FP9_DEN_SHIFT;  // round-half-up; safe, we're outside the tie band
    if (rounded < 0) rounded = 0;
    if (rounded > 255) rounded = 255;
    *out_u8 = static_cast<unsigned char>(rounded);
    return true;
}

// ---------------------------------------------------------------------
// NV12/P010 -> RGB: one thread per 2x2 luma block + its shared chroma
// pair. `bits` selects 8-bit (NV12, uint8 planes) or 10-bit-in-16
// (P010, uint16 planes, data in the top 10 bits) sample width.
// y_pitch/uv_pitch are *row strides in samples* (NV12: bytes; P010:
// 16-bit words), independent of W, so a pitched/imported Vulkan plane
// can be read directly without a de-pitch copy -- defaults to W when the
// caller passes a contiguous plane.
// ---------------------------------------------------------------------

// Fetch one chroma (U,V) sample at (row,col), clamped to the valid
// [0,ch)x[0,cw) range (edge replication, matching _chroma_upsample_numpy/
// _chroma_upsample_torch's np.concatenate([v[:1],...])-style clamp).
//
// Returns the RAW sample value (post >>6 unpack for P010, pre range-
// rescale) widened to `float`, NOT the rescaled double the previous
// version of this kernel produced. Rationale (perf-refinement pass,
// Task 3): the interpolation ladder below blends these raw values with
// weights that are finite dyadic fractions (0.25/0.75/0.5 and their
// nested products, e.g. 0.0625/0.1875/0.5625), and the raw operands are
// bounded 8-bit (NV12, <=255) or 10-bit (P010, <=1023) integers -- every
// such blend is EXACT in float32 (result needs <=14 mantissa bits, float32
// has 24) because (range rescale) is an affine map and affine maps commute
// with any weighted average whose weights sum to 1:
//     sum_i w_i * ((raw_i - coff) * cscale) == (sum_i w_i*raw_i - coff) * cscale
// So blending raw samples in float32 THEN rescaling the blended result
// once in double (done at the matrix-apply call site below) is
// mathematically identical to the old order (rescale every fetch in
// double, then blend in double), verified exhaustively/densely in
// fp32_ladder_sweep.py (see IMPLEMENTATION.md Phase 5): max residual
// 2.8e-14 across both ranges, both 8/10-bit domains, and every weight
// pattern this kernel uses -- that residual is double-precision
// summation-reordering noise (double epsilon * ~1e3 magnitude), not fp32
// error, and is 20 orders of magnitude below the 0.5 threshold that would
// flip np.rint's rounding decision. This cuts the ladder's double FMAs
// (up to ~20/thread in "center" siting) down to float32 ops, leaving only
// ONE double rescale per corner plus the (still fp64, NOT fp32-safe per
// fp32_sweep.py's exhaustive 256x256x256 sweep) a/b/c/d matrix-apply.
template <typename T>
__device__ __forceinline__ void fetch_uv_raw(
    const T* __restrict__ uv_plane, long uv_pitch, int ch, int cw,
    int row, int col, int shift6,
    float &u_raw_f, float &v_raw_f
) {
    row = row < 0 ? 0 : (row >= ch ? ch - 1 : row);
    col = col < 0 ? 0 : (col >= cw ? cw - 1 : col);
    long idx = (long)row * uv_pitch + (long)col * 2;
    long u_raw = uv_plane[idx + 0];
    long v_raw = uv_plane[idx + 1];
    if (shift6) { u_raw >>= 6; v_raw >>= 6; }
    u_raw_f = (float)u_raw;
    v_raw_f = (float)v_raw;
}

// Rescale one blended-raw-domain (still in raw sample units, e.g. up to
// 1023 with up to 4 fractional bits) float value to the final full-range
// double used by the matrix-apply step: single (x-coff)*cscale affine
// map in double, plus the P010 10-bit normalization. This is the ONE
// double rescale per corner that fetch_uv_raw's docstring above refers
// to -- replacing up to 6 double rescales/thread in the old code.
__device__ __forceinline__ double rescale_chroma(float raw_blended, int shift6, double coff, double cscale) {
    double v = ((double)raw_blended - coff) * cscale;
    if (shift6) v *= 0.25;
    return v;
}

template <typename T>
__global__ void yuv420_to_rgb_kernel(
    const T* __restrict__ y_plane, long y_pitch,
    const T* __restrict__ uv_plane, long uv_pitch,
    unsigned char* __restrict__ out_u8,
    float* __restrict__ out_f32,
    int out_is_float,
    int H, int W,
    double a, double b, double c, double d,
    double loff, double lscale, double coff, double cscale,
    int shift6,        // 1 for P010 (>>6 unpack), 0 for NV12
    int chroma_center,  // 0: "left" siting, 1: "center" siting
    int chroma_nearest,   // 1: legacy nearest-neighbour replication
    // Phase 9 fixed-point matrix-apply (uint8 output path only -- see
    // to_u8_fixed_or_fallback's docstring above). a_fp/b_fp/c_fp/d_fp,
    // lscale_fp, cscale_fp are round(coeff * 2**FP9_SCALE_SHIFT), with
    // the P010 0.25 unpack factor already folded into lscale_fp/
    // cscale_fp host-side (colour.py) so no runtime shift6 branch is
    // needed on the fixed-point path. loff16/coff16 are loff/coff
    // (already bit-depth-scaled by colour.py) times Q4=16, exact
    // integers. w_scaled is the certified guard-band threshold (see
    // derive_guardband.py / IMPLEMENTATION.md Phase 9), already scaled
    // by den=2**FP9_DEN_SHIFT. use_fixed_point selects this path at all
    // (host sets it False only when ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1,
    // a debug/benchmark escape hatch).
    long long a_fp, long long b_fp, long long c_fp, long long d_fp,
    long long lscale_fp, long long cscale_fp,
    long long loff16, long long coff16, long long w_scaled,
    int use_fixed_point
) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = (W + 1) / 2;
    int ch = (H + 1) / 2;
    if (cx >= cw || cy >= ch) return;

    // uf00/vf00: top-left (even row, even/co-sited col) reconstructed
    // chroma value for this block; uf01/uf10/uf11 the other 3 corners.
    // See this module's and colour.py's top-of-file docstrings for the
    // position algebra these weights come from. Everything in this block
    // is raw-domain float32 (fetch_uv_raw / blend below) -- the final
    // per-corner double rescale happens once, at first use, in the
    // dy/dx loop (rescale_chroma()), not here. See fetch_uv_raw's
    // docstring above for why this reorder is bit-exact.
    float uf00, vf00, uf01, vf01, uf10, vf10, uf11, vf11;

    if (chroma_nearest) {
        fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy, cx, shift6, uf00, vf00);
        uf01 = uf10 = uf11 = uf00;
        vf01 = vf10 = vf11 = vf00;
    } else {
        float u_c, v_c, u_up, v_up, u_dn, v_dn;
        fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy, cx, shift6, u_c, v_c);
        fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy - 1, cx, shift6, u_up, v_up);
        fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy + 1, cx, shift6, u_dn, v_dn);
        float u_top = 0.25f * u_up + 0.75f * u_c, v_top = 0.25f * v_up + 0.75f * v_c;
        float u_bot = 0.75f * u_c + 0.25f * u_dn, v_bot = 0.75f * v_c + 0.25f * v_dn;

        if (!chroma_center) {
            // "left": horizontally co-sited (direct at even col); odd col
            // is the 0.5/0.5 average with the chroma sample to the right.
            float u_c_r, v_c_r, u_up_r, v_up_r, u_dn_r, v_dn_r;
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy, cx + 1, shift6, u_c_r, v_c_r);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy - 1, cx + 1, shift6, u_up_r, v_up_r);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy + 1, cx + 1, shift6, u_dn_r, v_dn_r);
            float u_top_r = 0.25f * u_up_r + 0.75f * u_c_r, v_top_r = 0.25f * v_up_r + 0.75f * v_c_r;
            float u_bot_r = 0.75f * u_c_r + 0.25f * u_dn_r, v_bot_r = 0.75f * v_c_r + 0.25f * v_dn_r;
            uf00 = u_top; vf00 = v_top;
            uf01 = 0.5f * u_top + 0.5f * u_top_r; vf01 = 0.5f * v_top + 0.5f * v_top_r;
            uf10 = u_bot; vf10 = v_bot;
            uf11 = 0.5f * u_bot + 0.5f * u_bot_r; vf11 = 0.5f * v_bot + 0.5f * v_bot_r;
        } else {
            // "center": symmetric 0.25/0.75 centred blend on both axes.
            float u_c_l, v_c_l, u_up_l, v_up_l, u_dn_l, v_dn_l;
            float u_c_r, v_c_r, u_up_r, v_up_r, u_dn_r, v_dn_r;
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy, cx - 1, shift6, u_c_l, v_c_l);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy - 1, cx - 1, shift6, u_up_l, v_up_l);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy + 1, cx - 1, shift6, u_dn_l, v_dn_l);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy, cx + 1, shift6, u_c_r, v_c_r);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy - 1, cx + 1, shift6, u_up_r, v_up_r);
            fetch_uv_raw(uv_plane, uv_pitch, ch, cw, cy + 1, cx + 1, shift6, u_dn_r, v_dn_r);
            float u_top_l = 0.25f * u_up_l + 0.75f * u_c_l, v_top_l = 0.25f * v_up_l + 0.75f * v_c_l;
            float u_bot_l = 0.75f * u_c_l + 0.25f * u_dn_l, v_bot_l = 0.75f * v_c_l + 0.25f * v_dn_l;
            float u_top_r = 0.25f * u_up_r + 0.75f * u_c_r, v_top_r = 0.25f * v_up_r + 0.75f * v_c_r;
            float u_bot_r = 0.75f * u_c_r + 0.25f * u_dn_r, v_bot_r = 0.75f * v_c_r + 0.25f * v_dn_r;
            uf00 = 0.25f * u_top_l + 0.75f * u_top; vf00 = 0.25f * v_top_l + 0.75f * v_top;
            uf01 = 0.75f * u_top + 0.25f * u_top_r; vf01 = 0.75f * v_top + 0.25f * v_top_r;
            uf10 = 0.25f * u_bot_l + 0.75f * u_bot; vf10 = 0.25f * v_bot_l + 0.75f * v_bot;
            uf11 = 0.75f * u_bot + 0.25f * u_bot_r; vf11 = 0.75f * v_bot + 0.25f * v_bot_r;
        }
    }

#pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        int py = cy * 2 + dy;
        if (py >= H) continue;
        long y_row = (long)py * y_pitch;
#pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            int px = cx * 2 + dx;
            if (px >= W) continue;
            long y_raw = y_plane[y_row + px];
            if (shift6) y_raw >>= 6;

            float uf_raw = dy == 0 ? (dx == 0 ? uf00 : uf01) : (dx == 0 ? uf10 : uf11);
            float vf_raw = dy == 0 ? (dx == 0 ? vf00 : vf01) : (dx == 0 ? vf10 : vf11);

            long out_idx = ((long)py * W + px) * 3;

            if (out_is_float) {
                // Float32-output path stays fp64 (Phase 9 scope: the
                // bandwidth target is uint8 NV12->RGB; reproducing
                // `(float)(r/255.0)` bit-exactly in fixed point is a
                // separate, unscoped problem -- see IMPLEMENTATION.md
                // Phase 9).
                double yf = ((double)y_raw - loff) * lscale;
                if (shift6) yf *= 0.25;
                double uf = rescale_chroma(uf_raw, shift6, coff, cscale);
                double vf = rescale_chroma(vf_raw, shift6, coff, cscale);
                double r = fmin(fmax(yf + a * vf, 0.0), 255.0);
                double g = fmin(fmax(yf - c * uf - d * vf, 0.0), 255.0);
                double bch = fmin(fmax(yf + b * uf, 0.0), 255.0);
                out_f32[out_idx + 0] = (float)(r / 255.0);
                out_f32[out_idx + 1] = (float)(g / 255.0);
                out_f32[out_idx + 2] = (float)(bch / 255.0);
                continue;
            }

            unsigned char r_u8 = 0, g_u8 = 0, b_u8 = 0;
            bool r_ok = false, g_ok = false, b_ok = false;

            if (use_fixed_point) {
                // y_raw is an exact integer; uf_raw/vf_raw are exact
                // multiples of 1/16 (fetch_uv_raw's docstring) -- both
                // convert to an EXACT Q4=16 fixed-point integer, no
                // rounding error introduced here.
                long long y_raw16 = ((long long)y_raw) << FP9_Q4_SHIFT;
                long long u_raw16 = (long long)lroundf(uf_raw * 16.0f);
                long long v_raw16 = (long long)lroundf(vf_raw * 16.0f);

                long long y_diff16 = y_raw16 - loff16;
                long long u_diff16 = u_raw16 - coff16;
                long long v_diff16 = v_raw16 - coff16;

                long long yf_fixed = y_diff16 * lscale_fp;   // scale: Q4 * 2**S
                long long uf_fixed = u_diff16 * cscale_fp;
                long long vf_fixed = v_diff16 * cscale_fp;

                long long yf_scaled = yf_fixed << FP9_SCALE_SHIFT;  // scale: Q4 * 2**2S == den
                long long r_fixed = yf_scaled + a_fp * vf_fixed;
                long long g_fixed = yf_scaled - c_fp * uf_fixed - d_fp * vf_fixed;
                long long b_fixed = yf_scaled + b_fp * uf_fixed;

                r_ok = to_u8_fixed_or_fallback(r_fixed, w_scaled, &r_u8);
                g_ok = to_u8_fixed_or_fallback(g_fixed, w_scaled, &g_u8);
                b_ok = to_u8_fixed_or_fallback(b_fixed, w_scaled, &b_u8);
            }

            if (!r_ok || !g_ok || !b_ok) {
                // Rare path (certified by derive_guardband.py to be the
                // ONLY case where the fast path's rounding could possibly
                // disagree with the reference): same double formula, same
                // op order, as the always-fp64 code this replaces --
                // proven bit-exact by the existing exhaustive sweep
                // (test_colour_fp32_exactness.py), -ffp-contract=off
                // still applies (whole translation unit).
                double yf = ((double)y_raw - loff) * lscale;
                if (shift6) yf *= 0.25;
                double uf = rescale_chroma(uf_raw, shift6, coff, cscale);
                double vf = rescale_chroma(vf_raw, shift6, coff, cscale);
                if (!r_ok) r_u8 = to_u8(yf + a * vf);
                if (!g_ok) g_u8 = to_u8(yf - c * uf - d * vf);
                if (!b_ok) b_u8 = to_u8(yf + b * uf);
            }

            out_u8[out_idx + 0] = r_u8;
            out_u8[out_idx + 1] = g_u8;
            out_u8[out_idx + 2] = b_u8;
        }
    }
}

// ---------------------------------------------------------------------
// Phase 10: fixed-point (int64) matrix-apply, RGB->NV12 (encode)
// direction. Same guard-band technique as Phase 9 (decode), different
// derivation -- see colour.py's Phase 10 docstring and
// tools/colour_fixed_point/derive_encode_guardband.py for the full
// algebra/error-bound derivation. Summary: the reference chain's
// division-by-(a|b|lscale|cscale) steps are all by STREAM-CONSTANT
// values, so substituting cb=(bch-yf)/b, cr=(r-yf)/a into the chain
// collapses each of y/cb/cr into a single linear combination of
// (r,g,bch) plus an offset -- a 3x3 affine matrix, precomputed host-side
// (colour.py's _fp10_coeffs) into int64 coefficients at scale 2**S
// (FP10_SCALE_SHIFT). Unlike decode, the OPERANDS here (r,g,bch) are a
// continuous double (from an arbitrary float32 RGB pixel, not an
// already-exact small-denominator grid), so they are also quantized
// in-kernel to QIN fractional bits (FP10_QIN_SHIFT) -- a second
// quantization error source decode did not have, included in the
// derivation's error bound.
#define FP10_SCALE_SHIFT 24
#define FP10_QIN_SHIFT 24
#define FP10_DEN_SHIFT (FP10_SCALE_SHIFT + FP10_QIN_SHIFT)  // 48

// Generalized guard-band check/round (see to_u8_fixed_or_fallback above
// for the Phase 9 analogue): `den`/`shift` let the SAME function serve
// both the per-pixel Y channel (den = 2**FP10_DEN_SHIFT) and the
// 2x2-block-averaged Cb/Cr channels (den = 4 * 2**FP10_DEN_SHIFT, still
// a power of two since 4 is -- `channel_fixed` there is the SUM of 4
// per-pixel fixed values, representing 4x the block average, not yet
// divided by 4; dividing-by-shifting at the end folds the /4 in exactly,
// no separate rounding step).
__device__ __forceinline__ bool to_u8_fp10_or_fallback(
    long long channel_fixed, long long den, int shift, long long w_scaled, unsigned char *out_u8
) {
    long long clamped = channel_fixed < 0 ? 0 : (channel_fixed > 255 * den ? 255 * den : channel_fixed);
    long long rem = clamped & (den - 1);
    long long signed_dist = rem - (den >> 1);
    long long dist = signed_dist < 0 ? -signed_dist : signed_dist;
    if (dist < w_scaled) {
        return false;  // ambiguous -- caller must fall back to the double path
    }
    long long rounded = (clamped + (den >> 1)) >> shift;
    if (rounded < 0) rounded = 0;
    if (rounded > 255) rounded = 255;
    *out_u8 = static_cast<unsigned char>(rounded);
    return true;
}

__device__ __forceinline__ long long fp10_quantize(double v) {
    return (long long)llrint(v * (double)(1LL << FP10_QIN_SHIFT));
}

// ---------------------------------------------------------------------
// RGB -> NV12: one thread per 2x2 RGB block. Chroma is averaged over the
// 2x2 block *before* rounding (matches the torch reference's "average
// then round once" order, documented siting: co-sited with the
// *top-left* luma sample of each 2x2 block, i.e. standard MPEG-2/H.26x
// 4:2:0 chroma siting used by NV12 -- not the JPEG/4:2:0 centred siting).
// Requires even H, W (same constraint as the torch reference's
// `.unfold(...,2,2)`): odd trailing row/column is dropped, matching
// unfold's truncation exactly.
// ---------------------------------------------------------------------

__global__ void rgb_to_nv12_kernel(
    const float* __restrict__ rgb, // (H,W,3) contiguous, [0,1]
    unsigned char* __restrict__ y_out,  // (H,W)
    unsigned char* __restrict__ uv_out, // (H/2,W/2,2)
    int H, int W,
    double a, double b, double c, double d,
    double kr, double kg, double kb,
    double loff, double lscale, double coff, double cscale,
    // Phase 10 fixed-point 3x3-matrix args (encode direction, see above).
    long long my_r, long long my_g, long long my_b,
    long long mcb_r, long long mcb_g, long long mcb_b,
    long long mcr_r, long long mcr_g, long long mcr_b,
    long long loff_fixed, long long coff_fixed,
    long long w_scaled_px, long long w_scaled_avg,
    int use_fixed_point
) {
    int cw = W / 2;
    int ch = H / 2;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cw || cy >= ch) return;

    const long long den_px = (long long)1 << FP10_DEN_SHIFT;
    const long long den_avg = den_px * 4;

    long long cb_fixed_sum = 0, cr_fixed_sum = 0;
    double r4[4], g4[4], b4[4];

#pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        int py = cy * 2 + dy;
#pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            int px = cx * 2 + dx;
            long idx = ((long)py * W + px) * 3;
            double r = (double)rgb[idx + 0] * 255.0;
            double g = (double)rgb[idx + 1] * 255.0;
            double bch = (double)rgb[idx + 2] * 255.0;
            r = fmin(fmax(r, 0.0), 255.0);
            g = fmin(fmax(g, 0.0), 255.0);
            bch = fmin(fmax(bch, 0.0), 255.0);
            int slot = dy * 2 + dx;
            r4[slot] = r; g4[slot] = g; b4[slot] = bch;

            unsigned char y_u8 = 0;
            bool y_ok = false;
            if (use_fixed_point) {
                long long rf = fp10_quantize(r), gf = fp10_quantize(g), bf = fp10_quantize(bch);
                long long y_fixed = my_r * rf + my_g * gf + my_b * bf + loff_fixed;
                y_ok = to_u8_fp10_or_fallback(y_fixed, den_px, FP10_DEN_SHIFT, w_scaled_px, &y_u8);

                long long cb_fixed = mcb_r * rf + mcb_g * gf + mcb_b * bf + coff_fixed;
                long long cr_fixed = mcr_r * rf + mcr_g * gf + mcr_b * bf + coff_fixed;
                cb_fixed_sum += cb_fixed;
                cr_fixed_sum += cr_fixed;
            }
            // Fallback (rare: !y_ok, or use_fixed_point disabled entirely
            // via ELIGSIR_COLOUR_FORCE_FP64_MATRIX) -- same double formula,
            // same op order, as the pre-Phase-10 kernel this replaces.
            if (!y_ok) {
                double yf = kr * r + kg * g + kb * bch;
                double yfs = yf / lscale + loff;
                y_u8 = to_u8(yfs);
            }
            y_out[(long)py * W + px] = y_u8;
        }
    }

    unsigned char cb_u8 = 0, cr_u8 = 0;
    bool cb_ok = false, cr_ok = false;
    if (use_fixed_point) {
        cb_ok = to_u8_fp10_or_fallback(cb_fixed_sum, den_avg, FP10_DEN_SHIFT + 2, w_scaled_avg, &cb_u8);
        cr_ok = to_u8_fp10_or_fallback(cr_fixed_sum, den_avg, FP10_DEN_SHIFT + 2, w_scaled_avg, &cr_u8);
    }
    if (!cb_ok || !cr_ok) {
        // Rare path (also taken wholesale when use_fixed_point is
        // disabled): recompute the double cb/cr formula from the 4
        // stored RGB corners and average -- same op order as the
        // pre-Phase-10 kernel, computed only when actually needed (not
        // unconditionally in the hot loop), so the common case never
        // pays this fp64 cost.
        double cb_sum = 0.0, cr_sum = 0.0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            double yf = kr * r4[i] + kg * g4[i] + kb * b4[i];
            cb_sum += (b4[i] - yf) / b;
            cr_sum += (r4[i] - yf) / a;
        }
        if (!cb_ok) cb_u8 = to_u8(cb_sum * 0.25 / cscale + coff);
        if (!cr_ok) cr_u8 = to_u8(cr_sum * 0.25 / cscale + coff);
    }
    long uv_idx = ((long)cy * cw + cx) * 2;
    uv_out[uv_idx + 0] = cb_u8;
    uv_out[uv_idx + 1] = cr_u8;
}

// ---------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------

static inline dim3 block2d() { return dim3(16, 16); }
static inline dim3 grid2d(int cw, int ch, dim3 blk) {
    return dim3((cw + blk.x - 1) / blk.x, (ch + blk.y - 1) / blk.y);
}

torch::Tensor nv12_to_rgb_hip(
    torch::Tensor y, torch::Tensor uv,
    double a, double b, double c, double d,
    double loff, double lscale, double coff, double cscale,
    bool out_is_float, int64_t y_pitch, int64_t uv_pitch,
    bool chroma_center, bool chroma_nearest,
    int64_t a_fp, int64_t b_fp, int64_t c_fp, int64_t d_fp,
    int64_t lscale_fp, int64_t cscale_fp,
    int64_t loff16, int64_t coff16, int64_t w_scaled, bool use_fixed_point
) {
    TORCH_CHECK(y.is_cuda() && uv.is_cuda(), "nv12_to_rgb_hip requires device tensors");
    TORCH_CHECK(y.scalar_type() == torch::kUInt8, "y must be uint8");
    int H = y.size(0);
    int W = y.size(1);
    if (y_pitch <= 0) y_pitch = W;
    if (uv_pitch <= 0) uv_pitch = (W / 2 + (W % 2)) * 2; // elements per chroma row

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk = block2d();
    dim3 grd = grid2d((W + 1) / 2, (H + 1) / 2, blk);

    if (out_is_float) {
        auto out = torch::empty({H, W, 3}, y.options().dtype(torch::kFloat32));
        yuv420_to_rgb_kernel<unsigned char><<<grd, blk, 0, stream.stream()>>>(
            y.data_ptr<unsigned char>(), y_pitch,
            uv.data_ptr<unsigned char>(), uv_pitch,
            nullptr, out.data_ptr<float>(), 1,
            H, W, a, b, c, d, loff, lscale, coff, cscale, 0, chroma_center, chroma_nearest,
            a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16, w_scaled, use_fixed_point);
        return out;
    } else {
        auto out = torch::empty({H, W, 3}, y.options().dtype(torch::kUInt8));
        yuv420_to_rgb_kernel<unsigned char><<<grd, blk, 0, stream.stream()>>>(
            y.data_ptr<unsigned char>(), y_pitch,
            uv.data_ptr<unsigned char>(), uv_pitch,
            out.data_ptr<unsigned char>(), nullptr, 0,
            H, W, a, b, c, d, loff, lscale, coff, cscale, 0, chroma_center, chroma_nearest,
            a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16, w_scaled, use_fixed_point);
        return out;
    }
}

torch::Tensor p010_to_rgb_hip(
    torch::Tensor y, torch::Tensor uv,
    double a, double b, double c, double d,
    double loff, double lscale, double coff, double cscale,
    bool out_is_float, int64_t y_pitch, int64_t uv_pitch,
    bool chroma_center, bool chroma_nearest,
    int64_t a_fp, int64_t b_fp, int64_t c_fp, int64_t d_fp,
    int64_t lscale_fp, int64_t cscale_fp,
    int64_t loff16, int64_t coff16, int64_t w_scaled, bool use_fixed_point
) {
    TORCH_CHECK(y.is_cuda() && uv.is_cuda(), "p010_to_rgb_hip requires device tensors");
    TORCH_CHECK(y.scalar_type() == torch::kInt16 || y.scalar_type() == torch::kUInt16,
                "y must be 16-bit (P010 packed)");
    int H = y.size(0);
    int W = y.size(1);
    if (y_pitch <= 0) y_pitch = W;
    if (uv_pitch <= 0) uv_pitch = (W / 2 + (W % 2)) * 2;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk = block2d();
    dim3 grd = grid2d((W + 1) / 2, (H + 1) / 2, blk);

    // P010 tensors arrive as int16/uint16; reinterpret bit pattern as
    // unsigned short in-kernel (values are never negative: top 10 bits
    // of a 16-bit sample is always >= 0 in the P010 packing).
    const unsigned short* yp = reinterpret_cast<const unsigned short*>(y.data_ptr());
    const unsigned short* uvp = reinterpret_cast<const unsigned short*>(uv.data_ptr());

    if (out_is_float) {
        auto out = torch::empty({H, W, 3}, y.options().dtype(torch::kFloat32));
        yuv420_to_rgb_kernel<unsigned short><<<grd, blk, 0, stream.stream()>>>(
            yp, y_pitch, uvp, uv_pitch,
            nullptr, out.data_ptr<float>(), 1,
            H, W, a, b, c, d, loff, lscale, coff, cscale, 1, chroma_center, chroma_nearest,
            a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16, w_scaled, use_fixed_point);
        return out;
    } else {
        auto out = torch::empty({H, W, 3}, y.options().dtype(torch::kUInt8));
        yuv420_to_rgb_kernel<unsigned short><<<grd, blk, 0, stream.stream()>>>(
            yp, y_pitch, uvp, uv_pitch,
            out.data_ptr<unsigned char>(), nullptr, 0,
            H, W, a, b, c, d, loff, lscale, coff, cscale, 1, chroma_center, chroma_nearest,
            a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16, w_scaled, use_fixed_point);
        return out;
    }
}

std::vector<torch::Tensor> rgb_to_nv12_hip(
    torch::Tensor rgb,
    double a, double b, double c, double d,
    double kr, double kg, double kb,
    double loff, double lscale, double coff, double cscale,
    int64_t my_r, int64_t my_g, int64_t my_b,
    int64_t mcb_r, int64_t mcb_g, int64_t mcb_b,
    int64_t mcr_r, int64_t mcr_g, int64_t mcr_b,
    int64_t loff_fixed, int64_t coff_fixed,
    int64_t w_scaled_px, int64_t w_scaled_avg, bool use_fixed_point
) {
    TORCH_CHECK(rgb.is_cuda(), "rgb_to_nv12_hip requires a device tensor");
    TORCH_CHECK(rgb.scalar_type() == torch::kFloat32, "rgb must be float32");
    auto rgbc = rgb.contiguous();
    int H = rgbc.size(0);
    int W = rgbc.size(1);
    int ch = H / 2, cw = W / 2;

    auto y_out = torch::empty({H, W}, rgbc.options().dtype(torch::kUInt8));
    auto uv_out = torch::empty({ch, cw, 2}, rgbc.options().dtype(torch::kUInt8));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk = block2d();
    dim3 grd = grid2d(cw, ch, blk);
    rgb_to_nv12_kernel<<<grd, blk, 0, stream.stream()>>>(
        rgbc.data_ptr<float>(),
        y_out.data_ptr<unsigned char>(), uv_out.data_ptr<unsigned char>(),
        H, W, a, b, c, d, kr, kg, kb, loff, lscale, coff, cscale,
        my_r, my_g, my_b, mcb_r, mcb_g, mcb_b, mcr_r, mcr_g, mcr_b,
        loff_fixed, coff_fixed, w_scaled_px, w_scaled_avg, use_fixed_point);
    return {y_out, uv_out};
}
"""

# load_inline auto-generates the pybind11 registration (and its own
# main.cpp) from `functions=` below; it needs forward declarations of the
# host launcher functions visible to that generated file, which is what
# this cpp_sources entry provides. The actual kernels/launcher bodies
# live in _CUDA_SOURCE above and get hipified/compiled separately.
_CPP_DECLARATIONS = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor nv12_to_rgb_hip(
    torch::Tensor y, torch::Tensor uv,
    double a, double b, double c, double d,
    double loff, double lscale, double coff, double cscale,
    bool out_is_float, int64_t y_pitch, int64_t uv_pitch,
    bool chroma_center, bool chroma_nearest,
    int64_t a_fp, int64_t b_fp, int64_t c_fp, int64_t d_fp,
    int64_t lscale_fp, int64_t cscale_fp,
    int64_t loff16, int64_t coff16, int64_t w_scaled, bool use_fixed_point);

torch::Tensor p010_to_rgb_hip(
    torch::Tensor y, torch::Tensor uv,
    double a, double b, double c, double d,
    double loff, double lscale, double coff, double cscale,
    bool out_is_float, int64_t y_pitch, int64_t uv_pitch,
    bool chroma_center, bool chroma_nearest,
    int64_t a_fp, int64_t b_fp, int64_t c_fp, int64_t d_fp,
    int64_t lscale_fp, int64_t cscale_fp,
    int64_t loff16, int64_t coff16, int64_t w_scaled, bool use_fixed_point);

std::vector<torch::Tensor> rgb_to_nv12_hip(
    torch::Tensor rgb,
    double a, double b, double c, double d,
    double kr, double kg, double kb,
    double loff, double lscale, double coff, double cscale,
    int64_t my_r, int64_t my_g, int64_t my_b,
    int64_t mcb_r, int64_t mcb_g, int64_t mcb_b,
    int64_t mcr_r, int64_t mcr_g, int64_t mcr_b,
    int64_t loff_fixed, int64_t coff_fixed,
    int64_t w_scaled_px, int64_t w_scaled_avg, bool use_fixed_point);
"""


def get_ext():
    """Build (once per process, cached on disk across processes) and
    return the compiled extension module, or re-raise the build error on
    every call once it has failed once (so callers fail fast/consistently
    rather than retrying a doomed compile every frame)."""
    global _ext, _ext_load_error
    if _ext is not None:
        return _ext
    if _ext_load_error is not None:
        raise _ext_load_error
    with _lock:
        if _ext is not None:
            return _ext
        if _ext_load_error is not None:
            raise _ext_load_error
        try:
            os.makedirs(_EXT_BUILD_DIR, exist_ok=True)
            from torch.utils.cpp_extension import load_inline

            _ext = load_inline(
                name="eligsir_colour_hip_ext",
                cpp_sources=[_CPP_DECLARATIONS],
                cuda_sources=[_CUDA_SOURCE],
                functions=["nv12_to_rgb_hip", "p010_to_rgb_hip", "rgb_to_nv12_hip"],
                build_directory=_EXT_BUILD_DIR,
                verbose=False,
                # -ffp-contract=off: root-cause fix (Phase 5/Task 3's exhaustive
                # HIP-kernel-vs-numpy sweep, the first one ever run against this
                # kernel directly rather than the torch reference) for a
                # pre-existing (not introduced by the float32 ladder change --
                # confirmed by re-running the sweep against the unmodified
                # kernel, see IMPLEMENTATION.md Phase 5) 1-LSB mismatch at exact
                # round-half-to-even tie boundaries: without this flag, hipcc
                # contracts the matrix-apply's `a*vf + yf`-shaped expressions
                # into fused multiply-add instructions whose single rounding
                # step can disagree with numpy's separately-rounded double ops
                # at an exact .5 tie (classic FMA-contraction non-associativity,
                # not a precision bug). Verified empirically: with this flag,
                # the full 256x256x256 nearest-siting sweep (this module's
                # tests/video/test_colour_fp32_exactness.py) goes from 1
                # mismatch to 0 across every (matrix, range) combination tried.
                extra_cuda_cflags=["-ffp-contract=off"],
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any build
            # failure (missing hipcc, CPU-only dev container, etc.) must fall
            # back to the torch reference implementation, never crash import.
            _ext_load_error = exc
            raise
    return _ext
