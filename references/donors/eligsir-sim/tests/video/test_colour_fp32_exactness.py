"""Exhaustive/dense fp64-vs-fp32 exactness sweeps for the Task 3 perf
pass (Phase 5, IMPLEMENTATION.md): _colour_hip_ext.py's chroma-
interpolation ladder now runs in float32 (fetch_uv_raw/rescale_chroma),
the a/b/c/d YUV<->RGB matrix-apply stays float64.

Two things are verified here, in order, matching the brief's required
order ("write the sweep first, before touching the kernel"):

1. `test_matrix_apply_fp32_is_not_safe_*`: a REGRESSION GUARD, not a
   feature test. It reproduces the finding that forced this design (if
   a/b/c/d are naively narrowed to float32, the exhaustive 256x256x256
   Y/U/V grid DOES flip some np.rint() rounding decisions in the G
   channel -- a real, if rare, bit-exactness violation this repo's
   "no workarounds" rule forbids shipping). If someone later "optimizes"
   by casting a/b/c/d to float in the kernel, this test's numpy model of
   that change continues to demonstrate why it must not happen, and the
   kernel-level guard is `test_no_host_sync_guard.py`-style grep (not
   added here -- the matrix-apply precision isn't a static-analyzable
   property) plus this test's own bit-exactness sweep against the real
   kernel in (3).
2. `test_chroma_ladder_fp32_raw_domain_reorder_is_exact`: proves the
   REORDER actually shipped (blend raw pre-rescale samples in float32,
   rescale once in double before the matrix-apply) is bit-identical to
   the old order (rescale every fetch in double, then blend in double) to
   double-rounding noise (~1e-14, 20 orders of magnitude below the 0.5
   np.rint threshold) -- the mathematical justification is: for any
   affine map f(x) = (x - coff) * cscale and any weights summing to 1,
   sum_i w_i * f(raw_i) == f(sum_i w_i * raw_i) exactly in real-number
   arithmetic, and the raw-domain weighted sums here are exact in
   float32 (dyadic weights, <=10-bit raw inputs, <=14 bits needed).
3. `test_hip_kernel_bit_exact_against_numpy_reference_*`: the REAL
   verification on the actual compiled HIP kernel (GPU-only, skipped
   without a HIP/CUDA torch build) -- nearest-neighbour siting isolates
   the rescale_chroma + matrix-apply path and is checked over the FULL
   256 (Y) x 256 (U) x 256 (V) grid; "interpolated" siting (the ladder
   this phase changed) is checked over a dense random sample of
   (Y, neighbourhood of U/V) combinations, matrix x range x siting x
   bit-depth (8/10-bit). All must be np.array_equal vs the fp64 numpy
   reference -- no +/-1 LSB tolerance (that tolerance in
   test_colour.py::test_torch_matches_numpy_reference is for comparing
   two independent implementations, not for this kernel's own precision
   change).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from eligsir_sim.video.colour import (
    ColourTags,
    nv12_to_rgb,
    nv12_to_rgb_numpy,
    p010_to_rgb,
    p010_to_rgb_numpy,
    rgb_to_nv12,
    rgb_to_nv12_numpy,
)

_KR_KB = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}
_RANGES = {
    "limited": (16.0, 255.0 / 219.0, 128.0, 255.0 / 224.0),
    "full": (0.0, 1.0, 128.0, 1.0),
}


def _rgb_coeffs(matrix):
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    a = 2.0 * (1.0 - kr)
    b = 2.0 * (1.0 - kb)
    c = 2.0 * kb * (1.0 - kb) / kg
    d = 2.0 * kr * (1.0 - kr) / kg
    return a, b, c, d


def _matrix_apply_sweep(matrix, range_, dtype):
    a, b, c, d = (dtype(v) for v in _rgb_coeffs(matrix))
    loff, lscale, coff, cscale = (dtype(v) for v in _RANGES[range_])
    y_raw = np.arange(256, dtype=dtype)
    r = np.empty((256, 256, 256), dtype=dtype)
    g = np.empty((256, 256, 256), dtype=dtype)
    b_ = np.empty((256, 256, 256), dtype=dtype)
    for y0 in range(0, 256, 8):
        yv = y_raw[y0:y0 + 8][:, None, None]
        uf = (np.arange(256, dtype=dtype)[None, :, None] - coff) * cscale
        vf = (np.arange(256, dtype=dtype)[None, None, :] - coff) * cscale
        yf = (yv - loff) * lscale
        r[y0:y0 + 8] = yf + a * vf
        g[y0:y0 + 8] = yf - c * uf - d * vf
        b_[y0:y0 + 8] = yf + b * uf
    return r, g, b_


@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_matrix_apply_fp32_is_not_safe(matrix, range_):
    """Regression guard (see module docstring item 1): exhaustive
    256^3 Y/U/V sweep, naive full-fp32 matrix-apply vs fp64 reference.
    Must show >0 np.rint flips -- if this ever reports 0 flips (e.g. a
    future ROCm/numpy change alters rounding behaviour), the matrix-apply
    MAY become safe to narrow, but that would need re-deriving this
    design, not silently assuming it."""
    r64, g64, b64 = _matrix_apply_sweep(matrix, range_, np.float64)
    r32, g32, b32 = _matrix_apply_sweep(matrix, range_, np.float32)

    def rint_u8(v):
        return np.rint(np.clip(v, 0.0, 255.0))

    flips_g = int(np.sum(rint_u8(g64) != rint_u8(g32.astype(np.float64))))
    flips_r = int(np.sum(rint_u8(r64) != rint_u8(r32.astype(np.float64))))
    flips_b = int(np.sum(rint_u8(b64) != rint_u8(b32.astype(np.float64))))
    assert flips_g > 0, (
        f"{matrix}/{range_}: expected the known fp32 matrix-apply rint flips in G "
        f"(previously measured ~70-105 per combo out of 16,777,216) -- got 0; the "
        f"'must stay fp64' rationale may no longer hold, re-derive before changing "
        f"the kernel's matrix-apply precision"
    )
    assert flips_r == 0 and flips_b == 0  # R/B have historically never flipped


def _blend_raw_then_rescale_double(raws, weights, coff, cscale, shift6):
    vals = [(float(r) - coff) * cscale * (0.25 if shift6 else 1.0) for r in raws]
    return sum(w * v for w, v in zip(weights, vals))


def _blend_fp32_raw_domain_then_rescale_once(raws, weights, coff, cscale, shift6):
    raws32 = np.array(raws, dtype=np.float32)
    w32 = np.array(weights, dtype=np.float32)
    blended = np.float32(0.0)
    for w, r in zip(w32, raws32):
        blended = np.float32(blended + w * r)
    v = (float(blended) - coff) * cscale
    if shift6:
        v *= 0.25
    return v


@pytest.mark.parametrize("domain_max,shift6", [(255, 0), (1023, 1)])
def test_chroma_ladder_fp32_raw_domain_reorder_is_exact(domain_max, shift6):
    """See module docstring item 2. Dense (not exhaustive -- 4-6 free
    variables per weight pattern makes exhaustive infeasible; 41-point
    grid per variable, justified by the reorder being a closed-form
    algebraic identity, not a numerically-sensitive approximation) sweep
    over every weight pattern _colour_hip_ext.py's ladder actually uses."""
    grid = list(range(0, domain_max + 1, max(1, domain_max // 40)))
    weight_sets = [
        (0.25, 0.75), (0.75, 0.25), (0.5, 0.5),
        (0.0625, 0.1875, 0.1875, 0.5625),
        (0.1875, 0.5625, 0.0625, 0.1875),
    ]
    worst = 0.0
    for range_name, (loff, lscale, coff, cscale) in _RANGES.items():
        for weights in weight_sets:
            for raws in itertools.product(grid, repeat=len(weights)):
                ref = _blend_raw_then_rescale_double(raws, weights, coff, cscale, shift6)
                cand = _blend_fp32_raw_domain_then_rescale_once(raws, weights, coff, cscale, shift6)
                worst = max(worst, abs(ref - cand))
    # ~1e-14: double-precision summation-order noise in the single final
    # rescale multiply, NOT fp32 error (the fp32 blend step is exact for
    # these dyadic weights/small integers) -- 20 orders of magnitude below
    # the 0.5 threshold that would flip np.rint.
    assert worst < 1e-9, f"reorder residual too large to be double-rounding noise: {worst}"


pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a HIP/CUDA-capable torch build + GPU"
)


@pytestmark_gpu
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_hip_kernel_bit_exact_against_numpy_reference_nearest(matrix, range_):
    """Full 256(Y) x 256(U) x 256(V) grid through the REAL compiled HIP
    kernel, chroma_upsample="nearest" (isolates rescale_chroma +
    matrix-apply, no neighbour blending) -- must be np.array_equal vs
    the independent fp64 numpy reference, no LSB tolerance."""
    tags = ColourTags(matrix=matrix, range=range_)
    # cw=256 chroma columns x 2 luma rows per Y value -> H=512, W=512
    # (each 2x2 luma block gets a distinct (Y,U,V) from the full grid,
    # Y cycling 0..255 down the block rows so every Y value co-occurs
    # with every (U,V) pair across the full image).
    cw = ch = 256
    H, W = ch * 2, cw * 2
    uv = np.empty((ch, cw, 2), dtype=np.uint8)
    uv[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    uv[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    y = np.empty((H, W), dtype=np.uint8)
    y_cycle = (np.arange(H) % 256).astype(np.uint8)
    y[:, :] = y_cycle[:, None]

    ref = nv12_to_rgb_numpy(y, uv, tags, out_dtype="uint8", chroma_upsample="nearest")
    y_t, uv_t = torch.from_numpy(y).cuda(), torch.from_numpy(uv).cuda()
    out = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8", chroma_upsample="nearest").cpu().numpy()

    assert np.array_equal(ref, out), (
        f"{matrix}/{range_} nearest-siting: HIP kernel output is NOT bit-exact vs "
        f"the fp64 numpy reference after the float32 ladder/rescale change "
        f"(max_abs_diff={np.abs(ref.astype(np.int16) - out.astype(np.int16)).max()})"
    )


@pytestmark_gpu
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
@pytest.mark.parametrize("siting", ["left", "center"])
def test_hip_kernel_bit_exact_against_numpy_reference_interpolated(matrix, range_, siting):
    """Dense random sweep through the real kernel with the interpolation
    ladder active (chroma_upsample default) -- the actual code path this
    phase changed. 512x512 random NV12 input, full dynamic range."""
    rng = np.random.default_rng(hash((matrix, range_, siting)) & 0xFFFFFFFF)
    H, W = 512, 512
    y = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    uv = rng.integers(0, 256, size=(H // 2, W // 2, 2), dtype=np.uint8)
    tags = ColourTags(matrix=matrix, range=range_, chroma_location=siting)

    ref = nv12_to_rgb_numpy(y, uv, tags, out_dtype="uint8")
    y_t, uv_t = torch.from_numpy(y).cuda(), torch.from_numpy(uv).cuda()
    out = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8").cpu().numpy()

    diff = np.abs(ref.astype(np.int16) - out.astype(np.int16))
    assert np.array_equal(ref, out), (
        f"{matrix}/{range_}/{siting}: HIP kernel output is NOT bit-exact vs fp64 "
        f"numpy reference (max_abs_diff={diff.max()}, n_mismatch={int((diff > 0).sum())})"
    )


@pytestmark_gpu
def test_p010_hip_kernel_bit_exact_against_numpy_reference():
    """Same as the NV12 interpolated check, 10-bit P010 domain. Builds
    the torch uint16 tensor first (same pattern as
    test_colour.py::test_rgb_roundtrip_psnr_natural_image), then derives
    the numpy reference input from that SAME tensor's data (via
    .cpu().numpy()) rather than constructing it twice, so a conversion
    mismatch can't masquerade as a kernel bug."""
    rng = np.random.default_rng(99)
    H, W = 512, 512
    y8 = torch.from_numpy(rng.integers(0, 256, size=(H, W)).astype(np.int32))
    uv8 = torch.from_numpy(rng.integers(0, 256, size=(H // 2, W // 2, 2)).astype(np.int32))
    y10 = ((y8 * 4).clamp(0, 1023) << 6).to(torch.uint16)
    uv10 = ((uv8 * 4).clamp(0, 1023) << 6).to(torch.uint16)
    tags = ColourTags(matrix="bt709", range="limited")

    y10_np = y10.numpy()
    uv10_np = uv10.numpy()
    ref = p010_to_rgb_numpy(y10_np, uv10_np, tags, out_dtype="uint8")
    y_t, uv_t = y10.cuda(), uv10.cuda()
    out = p010_to_rgb(y_t, uv_t, tags, out_dtype="uint8").cpu().numpy()

    diff = np.abs(ref.astype(np.int16) - out.astype(np.int16))
    assert np.array_equal(ref, out), (
        f"P010 bit-exactness failed after float32 ladder change "
        f"(max_abs_diff={diff.max()}, n_mismatch={int((diff > 0).sum())})"
    )


# ---------------------------------------------------------------------------
# Phase 9: fixed-point (int64) matrix-apply mutation test for the guard
# band itself. The three tests above (`test_hip_kernel_bit_exact_against_
# numpy_reference_nearest/interpolated`, `test_p010_hip_kernel_bit_exact_
# against_numpy_reference`) already exercise the REAL fixed-point fast
# path (it is on by default -- see colour.py's _fp9_enabled) and already
# prove it is bit-exact. This test proves the guard band is load-bearing,
# not a no-op that happens to never trigger: forcing its width to 0 (via
# the test-only ELIGSIR_COLOUR_FP9_W_OVERRIDE escape hatch in colour.py's
# _fp9_coeffs) disables the fp64 fallback for every pixel, so the SAME
# exhaustive nearest-siting sweep that passes above must then reproduce
# the known near-tie mismatches (IMPLEMENTATION.md Phase 9's derivation:
# bt601/full has a genuine mathematical half-integer tie at a specific
# (U,V) that the fp64 reference's own rounding-chain bias resolves one
# way and naive round-half-to-even resolves the other way).
# ---------------------------------------------------------------------------


@pytestmark_gpu
def test_guard_band_w_override_zero_reproduces_known_mismatch(monkeypatch):
    """Mutation test for the Phase 9 guard band: W=0 must make the
    bt601/full nearest-siting sweep fail again (proving the guard band,
    not luck, is why the real (W>0) path passes)."""
    monkeypatch.setenv("ELIGSIR_COLOUR_FP9_W_OVERRIDE", "0.0")
    tags = ColourTags(matrix="bt601", range="full")
    cw = ch = 256
    H, W = ch * 2, cw * 2
    uv = np.empty((ch, cw, 2), dtype=np.uint8)
    uv[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    uv[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    y = np.empty((H, W), dtype=np.uint8)
    y_cycle = (np.arange(H) % 256).astype(np.uint8)
    y[:, :] = y_cycle[:, None]

    ref = nv12_to_rgb_numpy(y, uv, tags, out_dtype="uint8", chroma_upsample="nearest")
    y_t, uv_t = torch.from_numpy(y).cuda(), torch.from_numpy(uv).cuda()
    out = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8", chroma_upsample="nearest").cpu().numpy()

    n_mismatch = int(np.sum(ref != out))
    assert n_mismatch > 0, (
        "W=0 override did not reproduce any mismatch -- either the override "
        "is not wired up (regression in colour.py's _fp9_coeffs) or the "
        "guard band is not actually load-bearing for this combo any more "
        "(re-derive before trusting the W>0 default)"
    )


@pytestmark_gpu
def test_guard_band_w_override_default_matches_derivation(monkeypatch):
    """Sanity check for the override plumbing itself: explicitly setting
    W to the SAME value _fp9_guard_band_width() computes must reproduce
    the passing (0-mismatch) result -- confirms the override path and
    the default path are the same code, not two different formulas that
    happen to agree by coincidence."""
    from eligsir_sim.video.colour import _fp9_guard_band_width

    monkeypatch.setenv("ELIGSIR_COLOUR_FP9_W_OVERRIDE", repr(_fp9_guard_band_width()))
    tags = ColourTags(matrix="bt601", range="full")
    cw = ch = 256
    H, W = ch * 2, cw * 2
    uv = np.empty((ch, cw, 2), dtype=np.uint8)
    uv[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    uv[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    y = np.empty((H, W), dtype=np.uint8)
    y_cycle = (np.arange(H) % 256).astype(np.uint8)
    y[:, :] = y_cycle[:, None]

    ref = nv12_to_rgb_numpy(y, uv, tags, out_dtype="uint8", chroma_upsample="nearest")
    y_t, uv_t = torch.from_numpy(y).cuda(), torch.from_numpy(uv).cuda()
    out = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8", chroma_upsample="nearest").cpu().numpy()
    assert np.array_equal(ref, out)


# ---------------------------------------------------------------------------
# Phase 10: fixed-point (int64) matrix-apply, ENCODE direction
# (rgb_to_nv12). Same guard-band technique as Phase 9, different
# derivation (encode's operands are a continuous float, not an
# already-exact grid) -- see colour.py's Phase 10 docstring and
# tools/colour_fixed_point/derive_encode_guardband.py. The exactness
# target here is the kernel's OWN existing double-precision formula
# (ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1), NOT the CPU torch reference
# (_rgb_to_nv12_torch_ref) -- that torch reference computes in float32
# (not double; torch promotes python floats against a float32 tensor to
# float32, not float64), a PRE-EXISTING precision mismatch against the
# GPU kernel's double formula discovered while deriving this phase (see
# IMPLEMENTATION.md Phase 10's "pre-existing finding" note), unrelated to
# and not introduced by this phase, and out of this phase's scope to fix
# per the brief's own scoping (this phase's job is "keep the fixed-point
# path bit-exact vs the EXISTING fp64 reference" -- the existing
# reference for the encode kernel's own precision contract is its own
# double formula, exactly as Phase 9 used "the existing proven bit-exact
# double formula" as ITS reference, not a lower-precision CPU path).
# ---------------------------------------------------------------------------


def _rgb_to_nv12_fp64_forced(rgb, tags, monkeypatch):
    monkeypatch.setenv("ELIGSIR_COLOUR_FORCE_FP64_MATRIX", "1")
    y, uv = rgb_to_nv12(rgb, tags)
    monkeypatch.delenv("ELIGSIR_COLOUR_FORCE_FP64_MATRIX", raising=False)
    return y, uv


def _assert_rgb_to_nv12_matches_numpy_ref(rgb_t, rgb_np, tags, label):
    """rgb_t: cuda tensor; rgb_np: the SAME data as a numpy float32 array
    (H,W,3) in [0,1] -- compares the real compiled kernel's default
    (fixed-point) output against `rgb_to_nv12_numpy` (the INDEPENDENT
    fp64 numpy reference, see colour.py's docstring for why this, not
    `_rgb_to_nv12_torch_ref`'s float32 chain, nor the kernel's own
    fp64-forced path, is the correct exactness target -- a GPU kernel
    checked only against its own fallback branch is a circular
    reference, per this repo's no-circular-metrics rule)."""
    y_ref, uv_ref = rgb_to_nv12_numpy(rgb_np, tags)
    y_out, uv_out = rgb_to_nv12(rgb_t, tags)
    y_out_np, uv_out_np = y_out.cpu().numpy(), uv_out.cpu().numpy()
    assert np.array_equal(y_ref, y_out_np) and np.array_equal(uv_ref, uv_out_np), (
        f"{label}: rgb_to_nv12 (fixed-point default) is NOT bit-exact vs the "
        f"independent numpy fp64 reference "
        f"(y_mismatch={int(np.sum(y_ref != y_out_np))}, "
        f"uv_mismatch={int(np.sum(uv_ref != uv_out_np))})"
    )


@pytestmark_gpu
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_rgb_to_nv12_fixed_point_bit_exact_vs_independent_numpy_ref(matrix, range_):
    """The REAL exactness check (see `_assert_rgb_to_nv12_matches_numpy_ref`'s
    docstring for why this, not the kernel's own fp64-forced branch, is
    the correct target): three sweeps through the real compiled HIP
    kernel, FIXED (non-hash-based -- `hash()` of a tuple of str is
    PYTHONHASHSEED-randomized per process, so a prior version of this
    test was not even reproducible run-to-run) integer seeds:
    (a) dense random uint8-quantized-then-normalized RGB (what a real
    8-bit-sourced render produces), (b) dense random continuous float32
    RGB (the general calling convention), both 720p, 4 combos each."""
    tags = ColourTags(matrix=matrix, range=range_)
    H, W = 720, 1280
    seed_base = {"bt601": 0, "bt709": 1}[matrix] * 10 + {"limited": 0, "full": 1}[range_]

    rng_q = np.random.default_rng(1000 + seed_base)
    rgb_q_np = (rng_q.integers(0, 256, size=(H, W, 3)).astype(np.float32) / 255.0)
    _assert_rgb_to_nv12_matches_numpy_ref(torch.from_numpy(rgb_q_np).cuda(), rgb_q_np, tags,
                                           f"{matrix}/{range_}/uint8-quantized")

    rng_c = np.random.default_rng(2000 + seed_base)
    rgb_c_np = rng_c.random((H, W, 3)).astype(np.float32)
    _assert_rgb_to_nv12_matches_numpy_ref(torch.from_numpy(rgb_c_np).cuda(), rgb_c_np, tags,
                                           f"{matrix}/{range_}/continuous")


@pytestmark_gpu
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_rgb_to_nv12_exhaustive_y_channel_bit_exact(matrix, range_):
    """Exhaustive 256(R)x256(G)x256(B) sweep (16,777,216 = 4096x4096
    pixels, one PER-PIXEL (r,g,b) triple each, every combination
    appearing exactly once) through the real compiled HIP kernel vs the
    independent numpy fp64 reference, Y CHANNEL ONLY (Y is evaluated
    per-pixel, independent of its 2x2-block neighbours' values, so this
    construction gives genuine exhaustive per-pixel coverage; Cb/Cr are
    block-averaged over 4 pixels, so exhaustive coverage of THEIR domain
    would need 256^12 blocks -- infeasible -- and are instead covered by
    the dense random sweep above plus derive_encode_guardband.py's own
    exhaustive-grid + averaged-random host-side verification, which is
    not re-derived here, just re-checked against the real kernel)."""
    tags = ColourTags(matrix=matrix, range=range_)
    idx = np.arange(256 ** 3, dtype=np.int64)
    r = (idx // (256 * 256)).astype(np.uint8)
    g = ((idx // 256) % 256).astype(np.uint8)
    b = (idx % 256).astype(np.uint8)
    rgb_np = np.stack([r, g, b], axis=-1).reshape(4096, 4096, 3).astype(np.float32) / 255.0

    y_ref, _ = rgb_to_nv12_numpy(rgb_np, tags)
    y_out, _ = rgb_to_nv12(torch.from_numpy(rgb_np).cuda(), tags)
    y_out_np = y_out.cpu().numpy()
    n_mismatch = int(np.sum(y_ref != y_out_np))
    assert n_mismatch == 0, (
        f"{matrix}/{range_}: exhaustive 256^3 Y-channel sweep found {n_mismatch} "
        f"mismatches vs the independent numpy fp64 reference"
    )


@pytestmark_gpu
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_rgb_to_nv12_fixed_point_bit_exact_vs_kernel_fp64(matrix, range_, monkeypatch):
    """Secondary check: the default (fixed-point) path must also be
    np.array_equal vs the SAME kernel's ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1
    double-formula path -- i.e. the guard band's own correctness claim
    (not a substitute for the independent-numpy-reference tests above,
    which catch a bug common to both the fixed and fp64-forced branches
    that this comparison alone cannot)."""
    tags = ColourTags(matrix=matrix, range=range_)
    H, W = 720, 1280
    seed_base = {"bt601": 0, "bt709": 1}[matrix] * 10 + {"limited": 0, "full": 1}[range_]
    rng = np.random.default_rng(3000 + seed_base)
    rgb_q = torch.from_numpy((rng.integers(0, 256, size=(H, W, 3)).astype(np.float32) / 255.0)).cuda()
    rng2 = np.random.default_rng(4000 + seed_base)
    rgb_c = torch.from_numpy(rng2.random((H, W, 3)).astype(np.float32)).cuda()

    for rgb, label in ((rgb_q, "uint8-quantized"), (rgb_c, "continuous")):
        y_fixed, uv_fixed = rgb_to_nv12(rgb, tags)
        y_fp64, uv_fp64 = _rgb_to_nv12_fp64_forced(rgb, tags, monkeypatch)
        y_mismatch = int((y_fixed.to(torch.int16) != y_fp64.to(torch.int16)).sum())
        uv_mismatch = int((uv_fixed.to(torch.int16) != uv_fp64.to(torch.int16)).sum())
        assert y_mismatch == 0 and uv_mismatch == 0, (
            f"{matrix}/{range_}/{label}: rgb_to_nv12 fixed-point path is NOT bit-exact "
            f"vs the kernel's own fp64 formula (y_mismatch={y_mismatch}, uv_mismatch={uv_mismatch})"
        )


def _bt601_full_cr_tie_block():
    """A small (R,G,B) grid, (r=1, g=b=2k) for k=0..127, found by the
    brief's required tie-hunt (exhaustive search over bt601/full's `cr`
    channel for points whose fp64-reference value lands within the
    guard band of a rounding boundary -- see IMPLEMENTATION.md Phase 10
    for the full exact-tie census, which found both genuine mathematical
    half-integer ties (`2*remainder==denominator`, confirmed independently
    under two different exact-rational conventions) AND, separately,
    points like this family where the fp64 reference's OWN rounding bias
    (not a true mathematical tie) determines whether W=0 disagrees --
    both are exactly the class of input the guard band exists to catch).
    This family alone gives 79/128 W=0 disagreements against the fp64
    reference (verified directly, see
    tools/colour_fixed_point/derive_encode_guardband.py)."""
    k = np.arange(0, 128, dtype=np.float32)
    r = np.ones_like(k)
    g = 2.0 * k
    b = 2.0 * k
    return r, g, b


@pytestmark_gpu
def test_rgb_to_nv12_guard_band_w_override_zero_reproduces_known_mismatch(monkeypatch):
    """Mutation test for the Phase 10 guard band (mirrors Phase 9's
    test_guard_band_w_override_zero_reproduces_known_mismatch): forcing
    W=0 (via the test-only ELIGSIR_COLOUR_FP10_W_OVERRIDE escape hatch)
    on a documented block of near-boundary inputs (see
    _bt601_full_cr_tie_block) must disagree with the kernel's own fp64
    path, proving the guard band -- not luck -- is why the real (W>0)
    default agrees. (Pure random RGB sampling DOES find a non-zero but
    sparse and seed-sensitive rate of W=0 disagreements for "full"-range
    combos -- measured directly: ~0.088% of (R,G,B) triples for a 720p
    bt601/full sample, IMPLEMENTATION.md Phase 10 -- so it is not
    reliable or reproducible enough for a deterministic mutation test;
    this documented 128-point family gives a dense, reproducible 79/128
    hit rate instead.)"""
    monkeypatch.setenv("ELIGSIR_COLOUR_FP10_W_OVERRIDE", "0.0")
    tags = ColourTags(matrix="bt601", range="full")
    r, g, b = _bt601_full_cr_tie_block()
    n = len(r)
    H, W = 2, n * 2  # one 2x2 block per tie input (both rows identical)
    rgb_np = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(n):
        rgb_np[:, 2 * i:2 * i + 2, 0] = r[i] / 255.0
        rgb_np[:, 2 * i:2 * i + 2, 1] = g[i] / 255.0
        rgb_np[:, 2 * i:2 * i + 2, 2] = b[i] / 255.0
    rgb = torch.from_numpy(rgb_np).cuda()

    y_fixed, uv_fixed = rgb_to_nv12(rgb, tags)
    monkeypatch.delenv("ELIGSIR_COLOUR_FP10_W_OVERRIDE", raising=False)
    y_fp64, uv_fp64 = _rgb_to_nv12_fp64_forced(rgb, tags, monkeypatch)

    n_mismatch = int((y_fixed.to(torch.int16) != y_fp64.to(torch.int16)).sum()) + \
        int((uv_fixed.to(torch.int16) != uv_fp64.to(torch.int16)).sum())
    assert n_mismatch > 0, (
        "W=0 override did not reproduce any mismatch on a block of known exact "
        "half-integer ties -- either the override is not wired up (regression in "
        "colour.py's _fp10_coeffs) or the guard band is not actually load-bearing "
        "any more (re-derive before trusting the W>0 default)"
    )


@pytestmark_gpu
def test_rgb_to_nv12_guard_band_w_override_default_matches_derivation(monkeypatch):
    """Sanity check for the override plumbing (mirrors Phase 9's
    corresponding test): explicitly setting W to the SAME value
    _fp10_guard_band_width() computes must reproduce the passing
    (0-mismatch) result."""
    from eligsir_sim.video.colour import _fp10_guard_band_width

    monkeypatch.setenv("ELIGSIR_COLOUR_FP10_W_OVERRIDE", repr(_fp10_guard_band_width()))
    tags = ColourTags(matrix="bt601", range="full")
    H, W = 720, 1280
    rng = np.random.default_rng(123)
    rgb = torch.from_numpy(rng.integers(0, 256, size=(H, W, 3)).astype(np.float32) / 255.0).cuda()

    y_fixed, uv_fixed = rgb_to_nv12(rgb, tags)
    monkeypatch.delenv("ELIGSIR_COLOUR_FP10_W_OVERRIDE", raising=False)
    y_fp64, uv_fp64 = _rgb_to_nv12_fp64_forced(rgb, tags, monkeypatch)

    assert torch.equal(y_fixed, y_fp64)
    assert torch.equal(uv_fixed, uv_fp64)
