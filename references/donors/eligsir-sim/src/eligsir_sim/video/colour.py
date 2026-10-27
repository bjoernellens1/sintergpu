"""Fused NV12/P010 <-> RGB colour conversion, GPU (torch) and CPU (numpy) reference.

The matrix (BT.601 vs BT.709) and the range (limited/"tv" vs full/"pc") are
**not** hardcoded: they are read once per stream from its colour tags (the
container/VUI ``color_space``/``color_range`` fields -- see
``capability.py``'s caller and ``vulkan_backend.py``'s stream probe) and
passed in as a ``ColourTags`` value. ``_rgb_coeffs``/`_yuv_coeffs`` derive the
4 conversion coefficients from the matrix's (Kr, Kb) luma weights by the
standard ITU-R formula, rather than embedding literal per-matrix constants,
so that picking the wrong tag actually changes the numeric output -- this is
what ``tests/video/test_colour.py::test_matrix_is_not_hardcoded`` (a
mutation test) checks: it asserts BT.601 and BT.709 produce *different*
results for the same input, which would silently stop failing if someone
"simplified" this back down to one hardcoded matrix.

Reference for the derivation: ITU-R BT.601-7 / BT.709-6, Kr/Kb per matrix:
  BT.601: Kr=0.299,  Kb=0.114
  BT.709: Kr=0.2126, Kb=0.0722
Full-range YCbCr -> RGB (Kg = 1 - Kr - Kb):
  R = Y + 2*(1-Kr)*Cr
  G = Y - 2*Kb*(1-Kb)/Kg*Cb - 2*Kr*(1-Kr)/Kg*Cr
  B = Y + 2*(1-Kb)*Cb
Limited ("tv") range rescales luma from [16,235] and chroma from [16,240]
to full range before applying the above (same rescale ffmpeg's swscale
applies for ``color_range=tv``, which is also what our H.265 test streams'
VUI reports when unset -- see RESULTS.md Sec 3d).

Chroma siting / upsampling (lossless-pipeline requirement, decoded frames
feed Gaussian-splatting frame preparation -- any avoidable loss here is a
loss splatting never gets back): 4:2:0 only samples chroma at half
resolution, so recovering a per-luma-pixel chroma value is an
interpolation problem, not a lookup. ``ColourTags.chroma_location`` records
*where* each chroma sample actually sits relative to its 2x2 luma block
(from the stream's VUI ``chroma_sample_loc_type`` -- see backend.py's
probe), and ``nv12_to_rgb``/``p010_to_rgb``'s ``chroma_upsample`` argument
picks the algorithm:

  - "left" (default -- MPEG-2/H.264/H.265's own default when the VUI
    doesn't override it, type 0): chroma is horizontally CO-SITED with the
    even/left luma column of its 2x2 block, and vertically CENTRED between
    the block's two luma rows. Correct reconstruction is a separable
    linear filter: horizontally, the even luma column takes the chroma
    sample directly and the odd column is the 0.5/0.5 average of that
    sample and its right neighbour (clamped at the right edge); vertically,
    the top luma row is 0.25*(row above) + 0.75*(this row) and the bottom
    luma row is 0.75*(this row) + 0.25*(row below) (clamped at top/bottom
    edges) -- this is the same filter ffmpeg's swscale/libyuv use for
    "left" 4:2:0 upsampling, derived here from the sample geometry (see
    ``_chroma_upsample_numpy``'s docstring for the position algebra), not
    copied as a magic coefficient table.
  - "center" (MPEG-1/JPEG, type 2): chroma sits at the centre of the 2x2
    block, equidistant from all four luma samples -- the same 0.25/0.75
    centred blend applies on BOTH axes (not co-sited on either).
  - ``chroma_upsample="nearest"`` keeps the previous nearest-neighbour
    replication (the brief's original fused-kernel design) as an explicit
    opt-in for callers that need bit-parity with that older behaviour or
    don't care about the extra half-pixel chroma blur nearest-neighbour
    adds; it is no longer the default because it is not what a real
    decoder/the splatting pipeline's own reference chain (ffmpeg) produces.

Colour primaries/transfer (``ColourTags.primaries``/``transfer``) are
carried as metadata only -- this module's R'G'B'<->Y'CbCr matrix math
depends only on ``matrix``/``range`` (primaries/transfer describe the
gamut and EOTF, a separate concern from the YCbCr<->RGB' matrix). They are
read from the stream's VUI the same way matrix/range/chroma_location are
(backend.py's probe), carried on ``ColourTags`` so a caller doing gamut-
aware work downstream has them, and logged (not silently defaulted) when
the VUI doesn't specify them -- see backend.py's ``probe()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Matrix = Literal["bt601", "bt709"]
Range = Literal["limited", "full"]
ChromaLocation = Literal["left", "center"]
ChromaUpsample = Literal["interpolated", "nearest"]

# (Kr, Kb) luma weights per matrix -- the only place the matrix choice is
# encoded as a literal; every coefficient downstream is derived from these.
_KR_KB = {
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
}


@dataclass(frozen=True)
class ColourTags:
    """Colour tags read from a stream's container/VUI metadata at open time
    (see vulkan_backend.py's probe of AVStream.codecpar.color_space/
    color_range/chroma_location/color_primaries/color_trc). Never guess
    these in the decode hot path -- read once at stream open and carry them
    on every DeviceFrame.

    chroma_location: where chroma is actually sited in the stream's 4:2:0
      sampling grid (see this module's docstring); "left" is the MPEG-2/
      H.264/H.265 default used when the VUI doesn't override it.
    primaries/transfer: gamut/EOTF tags, metadata only (not consumed by
      this module's matrix math) -- see this module's docstring.
    """

    matrix: Matrix = "bt601"
    range: Range = "limited"
    chroma_location: ChromaLocation = "left"
    primaries: str = "bt709"
    transfer: str = "bt709"

    def __post_init__(self):
        if self.matrix not in _KR_KB:
            raise ValueError(f"unknown colour matrix tag: {self.matrix!r}")
        if self.range not in ("limited", "full"):
            raise ValueError(f"unknown colour range tag: {self.range!r}")
        if self.chroma_location not in ("left", "center"):
            raise ValueError(f"unknown chroma_location tag: {self.chroma_location!r}")


def _rgb_coeffs(tags: ColourTags) -> tuple[float, float, float, float]:
    """Derive (a, b, c, d) such that, in full-range YCbCr:
    R = Y + a*Cr; G = Y - c*Cb - d*Cr; B = Y + b*Cb
    from the matrix's (Kr, Kb). Not a lookup table of hardcoded (a,b,c,d) --
    computed from first principles so a new matrix only needs its (Kr, Kb)."""
    kr, kb = _KR_KB[tags.matrix]
    kg = 1.0 - kr - kb
    a = 2.0 * (1.0 - kr)
    b = 2.0 * (1.0 - kb)
    c = 2.0 * kb * (1.0 - kb) / kg
    d = 2.0 * kr * (1.0 - kr) / kg
    return a, b, c, d


def _range_scale(tags: ColourTags) -> tuple[float, float, float, float]:
    """(luma_offset, luma_scale, chroma_offset, chroma_scale) to rescale
    limited-range samples to full range before the matrix above; full range
    is the identity (offset still removes chroma's 128 bias)."""
    if tags.range == "limited":
        return 16.0, 255.0 / 219.0, 128.0, 255.0 / 224.0
    return 0.0, 1.0, 128.0, 1.0


# --------------------------------------------------------------------------
# Phase 9: fixed-point (int64) matrix-apply coefficients, uint8-output HIP
# path only. Root cause + full derivation: IMPLEMENTATION.md Phase 9 and
# the derivation scripts kept at tools/colour_fixed_point/ (see
# derive_guardband.py). Summary: Phase 7's elimination experiments
# root-caused the uint8-output kernel's bandwidth ceiling to fp64 ALU
# throughput on gfx1151 (RDNA3.5 has crippled native double-precision
# units); int64 fixed-point runs on the full-rate integer ALUs instead.
# Bit-exactness to the existing fp64 reference (this repo's non-
# negotiable rule) is NOT achievable for 100% of inputs via ANY finite
# fixed-point precision alone -- proven, not assumed: the fp64
# reference's own per-operation IEEE-754 rounding occasionally produces
# a result that differs from the true mathematical value by ~1e-14, and
# at the rare inputs where the true value is an EXACT half-integer tie,
# that reference-side bias (not fixed-point precision) decides the
# rounding. The fix is a certified guard band (Ziv's-test style, as used
# by correctly-rounded libm implementations): fixed-point is used when
# it is FAR from an exact tie (verified below to always then agree with
# the reference); the existing proven-bit-exact double formula is used
# for the pixels within the guard band.
_FP9_SCALE_SHIFT = 24          # a/b/c/d, lscale, cscale quantized to 2**24
_FP9_Q4_SHIFT = 4              # raw-sample domain scale (2**4 == 16, matches
                               # the chroma ladder's <=4-fraction-bit blend)
_FP9_DEN_SHIFT = _FP9_Q4_SHIFT + 2 * _FP9_SCALE_SHIFT  # 52


def _fp9_guard_band_width() -> float:
    """Certified guard-band half-width W (real-valued, channel-value units)
    for S=_FP9_SCALE_SHIFT: W = 2*(E_fix_bound + E_ref_bound).
    E_fix_bound = 8422 * 2**-(S+1): a provable (not sampled) bound on the
    fixed-point scheme's own worst-case deviation from the TRUE
    mathematical value, derived from per-coefficient round-to-nearest
    quantization error (<=2**-(S+1) each for lscale/cscale/a/b/c/d)
    propagated through the matrix-apply with operand magnitude bounds
    |y_diff|,|u_diff|,|v_diff| <= ~1100 (10-bit domain) and
    |uf|,|vf| <= ~1280 (scaled chroma amplitude); g (2 cross terms) is the
    worst channel. E_ref_bound = 1e-11: a deliberately generous bound on
    the EXISTING fp64 reference's own deviation from the true value
    (measured residuals in this codebase, Phase 5/7/9, were 1e-14..1e-15
    -- this is a >1000x safety margin, not a tight estimate). The 2x
    outer factor is additional margin. Verified empirically (not just
    trusted): derive_guardband.py's exhaustive 256^3 sweep (all 4
    matrix/range combos) plus dense 4,000,000-sample sweeps of fractional
    ("interpolated"-ladder) chroma and of the full 10-bit P010 domain all
    show ZERO pixels outside this band where the fast fixed-point
    rounding disagrees with the fp64 reference -- see IMPLEMENTATION.md
    Phase 9 for the full table (fallback rate per channel/domain)."""
    e_fix_bound = 8422.0 * 2.0 ** (-(_FP9_SCALE_SHIFT + 1))
    e_ref_bound = 1e-11
    return 2.0 * (e_fix_bound + e_ref_bound)


def _fp9_coeffs(tags: ColourTags, is_p010: bool) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Host-side (once per nv12_to_rgb/p010_to_rgb call, NOT per-pixel)
    derivation of the int64 fixed-point matrix-apply args for the HIP
    kernel's Phase 9 fast path. Returns
    (a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16,
    w_scaled) -- see _colour_hip_ext.py's to_u8_fixed_or_fallback and the
    yuv420_to_rgb_kernel docstring for how these are consumed."""
    import os

    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)
    if is_p010:
        # P010 10-bit: loff/coff are already *4'd by the caller (see
        # p010_to_rgb below, matching the existing double-path
        # convention); the kernel's old shift6-triggered "*0.25" on
        # yf/uf/vf is folded directly into lscale_fp/cscale_fp here so
        # the fixed-point path needs no runtime shift6 branch.
        loff *= 4.0
        coff *= 4.0
        lscale *= 0.25
        cscale *= 0.25
    scale = 1 << _FP9_SCALE_SHIFT
    a_fp = round(a * scale)
    b_fp = round(b * scale)
    c_fp = round(c * scale)
    d_fp = round(d * scale)
    lscale_fp = round(lscale * scale)
    cscale_fp = round(cscale * scale)
    loff16 = int(round(loff)) << _FP9_Q4_SHIFT
    coff16 = int(round(coff)) << _FP9_Q4_SHIFT
    den = 1 << _FP9_DEN_SHIFT
    w = _fp9_guard_band_width()
    override = os.environ.get("ELIGSIR_COLOUR_FP9_W_OVERRIDE")
    if override is not None:
        # Test-only escape hatch (mutation test for the guard band itself,
        # test_colour_fp32_exactness.py::test_guard_band_w_override_*):
        # forcing W=0 disables the fallback entirely, so the exhaustive
        # bit-exactness sweep must then start failing on the known
        # near-tie inputs -- proving the guard band is load-bearing, not
        # a no-op. Never read in production (nv12_to_rgb/p010_to_rgb's
        # callers never set this env var).
        w = float(override)
    w_scaled = int(round(w * den))
    return a_fp, b_fp, c_fp, d_fp, lscale_fp, cscale_fp, loff16, coff16, w_scaled


# --------------------------------------------------------------------------
# Phase 10: fixed-point (int64) matrix-apply coefficients for the ENCODE
# (rgb_to_nv12) direction. Full derivation: IMPLEMENTATION.md Phase 10 and
# tools/colour_fixed_point/derive_encode_guardband.py. Summary: this is
# the same guard-band technique as Phase 9 (decode), but the encode
# formula shape is different and needed its own derivation (the brief's
# own instruction, since the inverse transform is numerically distinct):
#
# The reference chain (_rgb_to_nv12_torch_ref / rgb_to_nv12_kernel) is
#   yf = kr*r + kg*g + kb*bch
#   cb = (bch - yf) / b             (b = 2*(1-kb))
#   cr = (r  - yf) / a               (a = 2*(1-kr))
#   y_final  = yf/lscale + loff
#   cb_final = cb/cscale + coff      (averaged over the 2x2 block, THEN
#   cr_final = cr/cscale + coff       rounded -- see rgb_to_nv12_kernel)
# Substituting cb/cr's definitions shows each of y_final/cb_final/
# cr_final is, algebraically, a SINGLE linear combination of (r, g, bch)
# plus a constant offset -- i.e. nothing but the standard RGB->YCbCr 3x3
# affine matrix, with every division replaced by a stream-constant
# reciprocal coefficient (derive_encode_guardband.py's `float_matrix`).
# This means the fixed-point kernel needs only ONE multiply-accumulate
# per output channel (not a 2-level dependent-division chain like a
# literal port of the reference's op order would), and a derivation
# self-check confirms the algebra: M_cb_b == M_cr_r == 0.5/cscale exactly
# (both reduce to (1-kb)/b and (1-kr)/a, and b=2*(1-kb), a=2*(1-kr) by
# construction).
#
# Unlike decode, encode's operands (r,g,bch) are a CONTINUOUS double
# (derived from an arbitrary float32 RGB pixel, clamped to [0,255]), not
# an already-exact small-denominator grid -- so this scheme quantizes the
# OPERANDS too (to QIN fractional bits), not just the coefficients (a
# second error source decode did not have). Bit-exactness to the fp64
# reference is, as with decode, provably unreachable for 100% of inputs
# via any finite fixed-point precision alone (the reference's own
# per-operation IEEE-754 rounding occasionally biases a genuine
# mathematical half-integer tie) -- the same certified guard band is used:
# fixed-point outside the band, the existing proven double formula inside.
_FP10_SCALE_SHIFT = 24   # S: coefficient (M_y/M_cb/M_cr, loff, coff) quant bits
_FP10_QIN_SHIFT = 24     # QIN: operand (r,g,bch) quantization bits
_FP10_DEN_SHIFT = _FP10_SCALE_SHIFT + _FP10_QIN_SHIFT  # 48


def _fp10_guard_band_width() -> float:
    """Certified guard-band half-width W for the encode scheme (S=QIN=24):
    W = 2*(E_fix_bound + E_ref_bound).

    E_fix_bound = 3*255*2**-(S+1) + SUM_ABS_M*2**-(QIN+1): the two
    independent first-order error sources for one output term
    M_i_fp*x_fixed (x in {r,g,bch}), summed over the <=3 operand terms of
    one output channel -- coefficient quantization (|error|<=2**-(S+1)
    each, times the operand bound 255, times 3 terms) plus operand
    quantization (|error|<=2**-(QIN+1) each, times the coefficient
    magnitude). SUM_ABS_M=1.0 is the worst-case sum(|M_i|) over every
    (matrix,range,channel) combo, computed numerically (not hand-
    estimated) in derive_encode_guardband.py's `worst_case_sum_abs_M` --
    it is EXACTLY 1.0 because M_y_r+M_y_g+M_y_b == 1/lscale <= 1 and
    (for "full" range, where lscale=cscale=1) each of M_y/M_cb/M_cr's
    |coefficients| sum to exactly 1 (a derivation self-check, confirmed
    by that script, not a coincidence).

    E_ref_bound = 2e-11: a deliberately generous bound on the EXISTING
    fp64 reference's own deviation from the true mathematical value.
    Measured directly (derive_encode_guardband.py's `measure_e_ref`,
    dense random RGB sampled against an exact-Fraction ground truth):
    ~5.68e-14 for every (matrix,range) combo -- this bound is a >350x
    safety margin over that measurement (2x decode's 1e-11 bound, since
    encode's reference chain has one more op level: an extra division
    plus, for cb/cr, a 2x2-block average of 4 already-rounded-chain
    values -- averaging does not amplify the bound, since the mean's
    deviation from the true mean is <= the single worst sample's
    deviation, but the extra division makes a larger single-sample bound
    the conservative choice). The outer 2x factor is additional margin.

    Verified empirically (not just trusted),
    derive_encode_guardband.py's `guard_band_check`: the full exhaustive
    256(R)x256(G)x256(B) grid AND a dense 4,000,000-sample sweep of
    CONTINUOUS (non-integer) float RGB, for all 4 (matrix,range) combos,
    for y/cb/cr AND the actual 2x2-block-averaged cb_avg/cr_avg path the
    kernel uses for chroma -- zero pixels/blocks outside the guard band
    disagree with the fp64 reference, across every domain tested. See
    IMPLEMENTATION.md Phase 10 for the full fallback-rate table."""
    sum_abs_m = 1.0
    e_fix_bound = 3 * 255.0 * 2.0 ** (-(_FP10_SCALE_SHIFT + 1)) + sum_abs_m * 2.0 ** (-(_FP10_QIN_SHIFT + 1))
    e_ref_bound = 2e-11
    return 2.0 * (e_fix_bound + e_ref_bound)


def _fp10_coeffs(tags: ColourTags) -> tuple:
    """Host-side (once per rgb_to_nv12 call, NOT per-pixel) derivation of
    the int64 fixed-point 3x3-matrix args for the HIP kernel's Phase 10
    fast path. Returns (my_r, my_g, my_b, mcb_r, mcb_g, mcb_b, mcr_r,
    mcr_g, mcr_b, loff_fixed, coff_fixed, w_scaled_px, w_scaled_avg) --
    see _colour_hip_ext.py's rgb_to_nv12_kernel for how these are
    consumed. Derivation: substitute cb=(bch-yf)/b, cr=(r-yf)/a into the
    reference chain to get each output channel as a single linear
    combination of (r,g,bch) -- see this module's Phase 10 docstring
    above for the full algebra."""
    import os

    kr, kb = _KR_KB[tags.matrix]
    kg = 1.0 - kr - kb
    a = 2.0 * (1.0 - kr)
    b = 2.0 * (1.0 - kb)
    loff, lscale, coff, cscale = _range_scale(tags)

    m_y = (kr / lscale, kg / lscale, kb / lscale)
    m_cb = (-kr / (b * cscale), -kg / (b * cscale), (1.0 - kb) / (b * cscale))
    m_cr = ((1.0 - kr) / (a * cscale), -kg / (a * cscale), -kb / (a * cscale))

    scale = 1 << _FP10_SCALE_SHIFT
    den = 1 << _FP10_DEN_SHIFT
    my_fp = tuple(round(x * scale) for x in m_y)
    mcb_fp = tuple(round(x * scale) for x in m_cb)
    mcr_fp = tuple(round(x * scale) for x in m_cr)
    # loff/coff are exact small integers (16 or 0, 128 or 0) -> exact at
    # scale `den`, no quantization error contributed by the offset term.
    loff_fixed = int(round(loff)) * den
    coff_fixed = int(round(coff)) * den

    w = _fp10_guard_band_width()
    override = os.environ.get("ELIGSIR_COLOUR_FP10_W_OVERRIDE")
    if override is not None:
        # Test-only escape hatch, mirrors ELIGSIR_COLOUR_FP9_W_OVERRIDE
        # (Phase 9): forcing W=0 disables the fallback entirely, proving
        # the guard band is load-bearing, not a no-op. Never read in
        # production (rgb_to_nv12's caller never sets this env var).
        w = float(override)
    w_scaled_px = int(round(w * den))
    w_scaled_avg = int(round(w * den * 4))  # the 2x2-block-average path's own denominator (den*4)

    return (*my_fp, *mcb_fp, *mcr_fp, loff_fixed, coff_fixed, w_scaled_px, w_scaled_avg)


def _fp10_enabled() -> bool:
    """Phase 10 fixed-point matrix-apply fast path (rgb_to_nv12, encode
    direction), on by default. Set ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1 to
    force every pixel/block through the (slower, always-fp64) fallback
    path -- same escape hatch and env var as Phase 9's decode-direction
    flag (ELIGSIR_COLOUR_FORCE_FP64_MATRIX forces BOTH directions off,
    since they are the same kind of debug/benchmark A/B toggle)."""
    return _fp9_enabled()


# --------------------------------------------------------------------------
# GPU/CPU production kernel (torch) -- dtype-polymorphic over the *input*
# tensor's device, so the identical code path runs on a CUDA/HIP tensor
# (production, no host round trip) or a CPU tensor (tests, CPU fallback
# backend).
# --------------------------------------------------------------------------


def _fp9_enabled() -> bool:
    """Phase 9 fixed-point matrix-apply fast path, on by default. Set
    ELIGSIR_COLOUR_FORCE_FP64_MATRIX=1 to force every pixel through the
    (slower, always-fp64) fallback path -- a debug/benchmark escape
    hatch to A/B the Phase 9 change against the pre-Phase-9 kernel
    behaviour without rebuilding, same convention as
    ELIGSIR_COLOUR_FORCE_TORCH_REF."""
    import os

    return os.environ.get("ELIGSIR_COLOUR_FORCE_FP64_MATRIX") != "1"


def _hip_ext_or_none():
    """Return the compiled HIP extension module, or None if it is
    unavailable (build failure/CPU-only container) -- cached after the
    first attempt so a doomed build is never retried per-frame. Set
    ELIGSIR_COLOUR_FORCE_TORCH_REF=1 to force the torch reference path
    even when the extension is available (used by tests/benchmarks that
    need to exercise both implementations deliberately)."""
    import os

    if os.environ.get("ELIGSIR_COLOUR_FORCE_TORCH_REF") == "1":
        return None
    from . import _colour_hip_ext

    try:
        return _colour_hip_ext.get_ext()
    except Exception:  # noqa: BLE001 - fall back to the torch reference
        return None


def _chroma_upsample_torch(v, H: int, W: int, siting: str, mode: str):
    """Reconstruct a (H,W) full-res chroma plane from a (h2,w2) 4:2:0
    half-res plane. ``mode="nearest"`` is plain replication (legacy,
    opt-in). ``mode="interpolated"`` applies the geometry-correct
    separable linear filter for ``siting`` ("left" or "center" -- see
    this module's top-of-file docstring for the position algebra). All
    maths in the input's own dtype (float32 for the HIP-kernel-adjacent
    torch reference paths, float64 for the numpy reference) -- never
    narrowed to uint8 before this point, so no intermediate rounding."""
    import torch

    h2, w2 = v.shape
    if mode == "nearest":
        out = v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        return out[:H, :W]
    if mode != "interpolated":
        raise ValueError(f"unsupported chroma_upsample: {mode!r}")

    # Vertical: "left" and "center" sitings both centre chroma vertically
    # between the block's two luma rows -- shared 0.25/0.75 blend, clamped
    # at the top/bottom edge (replicate, matching swscale's edge handling).
    prev_row = torch.cat([v[:1], v[:-1]], dim=0)
    next_row = torch.cat([v[1:], v[-1:]], dim=0)
    vtop = 0.25 * prev_row + 0.75 * v   # value at the block's top luma row
    vbot = 0.75 * v + 0.25 * next_row   # value at the block's bottom luma row

    if siting == "left":
        # Horizontal: co-sited with the even/left column (direct sample);
        # odd/right column is the 0.5/0.5 average with its right neighbour.
        top_right_nb = torch.cat([vtop[:, 1:], vtop[:, -1:]], dim=1)
        bot_right_nb = torch.cat([vbot[:, 1:], vbot[:, -1:]], dim=1)
        top_even, top_odd = vtop, 0.5 * vtop + 0.5 * top_right_nb
        bot_even, bot_odd = vbot, 0.5 * vbot + 0.5 * bot_right_nb
    elif siting == "center":
        # Horizontal: centred the same way as vertical, on both columns.
        top_prev = torch.cat([vtop[:, :1], vtop[:, :-1]], dim=1)
        top_next = torch.cat([vtop[:, 1:], vtop[:, -1:]], dim=1)
        bot_prev = torch.cat([vbot[:, :1], vbot[:, :-1]], dim=1)
        bot_next = torch.cat([vbot[:, 1:], vbot[:, -1:]], dim=1)
        top_even, top_odd = 0.25 * top_prev + 0.75 * vtop, 0.75 * vtop + 0.25 * top_next
        bot_even, bot_odd = 0.25 * bot_prev + 0.75 * vbot, 0.75 * vbot + 0.25 * bot_next
    else:
        raise ValueError(f"unknown chroma_location: {siting!r}")

    out = torch.empty((h2 * 2, w2 * 2), dtype=v.dtype, device=v.device)
    out[0::2, 0::2] = top_even
    out[0::2, 1::2] = top_odd
    out[1::2, 0::2] = bot_even
    out[1::2, 1::2] = bot_odd
    return out[:H, :W]


def nv12_to_rgb(y, uv, tags: ColourTags, out_dtype: str = "float32", *,
                 y_pitch: int | None = None, uv_pitch: int | None = None,
                 chroma_upsample: str = "interpolated"):
    """y: (H,W) uint8 torch tensor. uv: (H/2,W/2,2) uint8 torch tensor
    (U,V interleaved, NV12 order). Returns (H,W,3) tensor on the same
    device: float32 in [0,1] if out_dtype=="float32", else uint8 in
    [0,255]. Fused: one pass, no intermediate host copy. The float32
    path computes in float (double, in-kernel) throughout and only
    narrows to float32 on write -- no intermediate uint8 rounding.

    On a HIP/CUDA device tensor this dispatches to the fused HIP kernel
    (_colour_hip_ext), one thread per 2x2 luma block + its chroma pair,
    falling back to the pure-torch reference (_nv12_to_rgb_torch_ref) if
    the extension failed to build. CPU tensors always use the torch
    reference (no device kernel to dispatch to).

    y_pitch/uv_pitch (HIP path only, elements per row, default: tight
    W / chroma-row-width) let a caller hand in a *pitched* plane -- e.g.
    the row stride of an imported Vulkan Video surface -- without first
    copying it down to a contiguous buffer; a future vulkan_backend.py
    change can pass the decoder's native row pitch here directly to drop
    its current D2D de-pitch copy.

    chroma_upsample: "interpolated" (default) reconstructs chroma with
    the geometry-correct filter for tags.chroma_location (see this
    module's docstring); "nearest" keeps the old nearest-neighbour
    replication as an explicit opt-in (e.g. bit-parity with pre-existing
    captured data) -- see the docstring for why that is no longer the
    default.
    """
    import torch

    if y.dtype != torch.uint8 or uv.dtype != torch.uint8:
        raise ValueError("nv12_to_rgb expects uint8 NV12 planes (use p010_to_rgb for P010)")
    if out_dtype not in ("float32", "uint8"):
        raise ValueError(f"unsupported out_dtype: {out_dtype!r}")
    if chroma_upsample not in ("interpolated", "nearest"):
        raise ValueError(f"unsupported chroma_upsample: {chroma_upsample!r}")

    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)

    if y.is_cuda:
        ext = _hip_ext_or_none()
        if ext is not None:
            fp9 = _fp9_coeffs(tags, is_p010=False)
            return ext.nv12_to_rgb_hip(
                y, uv, a, b, c, d, loff, lscale, coff, cscale,
                out_dtype == "float32", y_pitch or 0, uv_pitch or 0,
                tags.chroma_location == "center", chroma_upsample == "nearest",
                *fp9, _fp9_enabled(),
            )
    return _nv12_to_rgb_torch_ref(y, uv, a, b, c, d, loff, lscale, coff, cscale, out_dtype,
                                   tags.chroma_location, chroma_upsample)


def _nv12_to_rgb_torch_ref(y, uv, a, b, c, d, loff, lscale, coff, cscale, out_dtype,
                            chroma_location, chroma_upsample):
    """Pure-torch reference implementation, kept indefinitely: the CPU
    fallback path, and the documented baseline the fused HIP kernel is
    checked against (tests/video/test_colour.py,
    test_colour_kernel_benchmark.py). Chain of separate ops with
    full-size intermediates -- this is the 2.08ms/frame@1080p baseline
    the fused kernel replaces on-device (RESULTS.md Sec "Semaphore
    bridge"; IMPLEMENTATION.md "Fused colour kernels")."""
    import torch

    H, W = y.shape
    yf = y.to(torch.float32)
    u = _chroma_upsample_torch(uv[..., 0].to(torch.float32), H, W, chroma_location, chroma_upsample)
    v = _chroma_upsample_torch(uv[..., 1].to(torch.float32), H, W, chroma_location, chroma_upsample)

    yf = (yf - loff) * lscale
    u = (u - coff) * cscale
    v = (v - coff) * cscale

    r = yf + a * v
    g = yf - c * u - d * v
    bch = yf + b * u
    rgb = torch.stack([r, g, bch], dim=-1).clamp(0, 255)

    if out_dtype == "float32":
        return (rgb / 255.0).to(torch.float32)
    return rgb.round().to(torch.uint8)


def p010_to_rgb(y, uv, tags: ColourTags, out_dtype: str = "float32", *,
                 y_pitch: int | None = None, uv_pitch: int | None = None,
                 chroma_upsample: str = "interpolated"):
    """Same as nv12_to_rgb but for P010: y/uv are uint16 torch tensors with
    data in the top 10 bits of each 16-bit sample (the P010 convention) --
    full 10-bit range is [0,1023] packed as [0,65472] step 64; we unpack by
    `>> 6` before applying the same matrix/range math on a 10-bit scale,
    keeping the full 10-bit precision through double-precision (HIP kernel)
    or float32 (torch reference) math -- the `>> 6` unpack is lossless
    (10 significant bits fit exactly in either). See nv12_to_rgb's
    docstring for the HIP dispatch/pitch/chroma_upsample/fallback contract
    (identical here)."""
    import torch

    if y.dtype != torch.int16 and y.dtype != torch.uint16 and y.dtype != torch.int32:
        raise ValueError("p010_to_rgb expects 16-bit-packed P010 planes")
    if out_dtype not in ("float32", "uint8"):
        raise ValueError(f"unsupported out_dtype: {out_dtype!r}")
    if chroma_upsample not in ("interpolated", "nearest"):
        raise ValueError(f"unsupported chroma_upsample: {chroma_upsample!r}")

    a, b, c, d = _rgb_coeffs(tags)
    # 10-bit limited/full range endpoints scale by 4x vs 8-bit (16->64, 235->940, etc).
    loff, lscale, coff, cscale = _range_scale(tags)
    loff *= 4.0
    coff *= 4.0

    if y.is_cuda and y.dtype != torch.int32:
        ext = _hip_ext_or_none()
        if ext is not None:
            fp9 = _fp9_coeffs(tags, is_p010=True)
            return ext.p010_to_rgb_hip(
                y, uv, a, b, c, d, loff, lscale, coff, cscale,
                out_dtype == "float32", y_pitch or 0, uv_pitch or 0,
                tags.chroma_location == "center", chroma_upsample == "nearest",
                *fp9, _fp9_enabled(),
            )
    return _p010_to_rgb_torch_ref(y, uv, a, b, c, d, loff, lscale, coff, cscale, out_dtype,
                                   tags.chroma_location, chroma_upsample)


def _p010_to_rgb_torch_ref(y, uv, a, b, c, d, loff, lscale, coff, cscale, out_dtype,
                            chroma_location, chroma_upsample):
    """Pure-torch reference implementation, kept indefinitely (see
    _nv12_to_rgb_torch_ref)."""
    import torch

    H, W = y.shape
    yf = (y.to(torch.int32) >> 6).to(torch.float32)
    uf = _chroma_upsample_torch((uv[..., 0].to(torch.int32) >> 6).to(torch.float32),
                                 H, W, chroma_location, chroma_upsample)
    vf = _chroma_upsample_torch((uv[..., 1].to(torch.int32) >> 6).to(torch.float32),
                                 H, W, chroma_location, chroma_upsample)

    # Normalize 10-bit sample range to the same [0,255]-equivalent scale as
    # nv12_to_rgb by dividing the (already-offset) value by 4 before scaling.
    yf = (yf - loff) / 4.0 * lscale
    uf = (uf - coff) / 4.0 * cscale
    vf = (vf - coff) / 4.0 * cscale

    r = yf + a * vf
    g = yf - c * uf - d * vf
    bch = yf + b * uf
    rgb = torch.stack([r, g, bch], dim=-1).clamp(0, 255)

    if out_dtype == "float32":
        return (rgb / 255.0).to(torch.float32)
    return rgb.round().to(torch.uint8)


def rgb_to_nv12(rgb, tags: ColourTags):
    """Inverse of nv12_to_rgb: rgb is (H,W,3) float32 torch tensor in [0,1].
    Returns (y, uv) uint8 torch tensors (NV12 layout, 4:2:0 chroma via
    simple 2x2 average -- matches the sim's encode-for-parity use case,
    which is resampling a rendered frame, not re-encoding a captured one).

    Chroma siting (KNOWN LIMITATION, documented rather than silently
    assumed per this repo's no-workarounds rule): a plain 2x2 box average
    is equidistant from all four luma samples, i.e. it sites the chroma
    sample at the CENTRE of the block -- the MPEG-1/JPEG ("center", type
    2) siting, NOT the MPEG-2/H.264/H.265 "left" siting that
    ``ColourTags.chroma_location`` defaults to and that ``nv12_to_rgb``'s
    "interpolated" upsample reconstructs by default. This function is not
    yet siting-aware (``tags.chroma_location`` is not consulted here) --
    round-tripping through this encoder and then decoding with the
    default "left" upsample therefore adds a small additional siting-
    mismatch error on top of ordinary 4:2:0 subsampling loss. Tests that
    want an apples-to-apples round trip must decode with
    ``ColourTags(chroma_location="center")`` to match what this function
    actually produces (see
    test_colour.py::test_rgb_roundtrip_psnr_natural_image). Fixing this
    properly means teaching this function siting-aware decimation (a
    3-tap lowpass centred on the left column for "left"), left for a
    follow-up pass rather than bundled into this one.

    On a HIP/CUDA device tensor this dispatches to the fused HIP kernel,
    one thread per 2x2 RGB block, falling back to the pure-torch
    reference if the extension failed to build.
    """
    import torch

    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)
    kr, kb = _KR_KB[tags.matrix]
    kg = 1.0 - kr - kb

    if rgb.is_cuda:
        ext = _hip_ext_or_none()
        if ext is not None:
            fp10 = _fp10_coeffs(tags)
            y, uv = ext.rgb_to_nv12_hip(
                rgb.to(torch.float32), a, b, c, d, kr, kg, kb, loff, lscale, coff, cscale,
                *fp10, _fp10_enabled(),
            )
            return y, uv
    return _rgb_to_nv12_torch_ref(rgb, a, b, c, d, kr, kg, kb, loff, lscale, coff, cscale)


def _rgb_to_nv12_torch_ref(rgb, a, b, c, d, kr, kg, kb, loff, lscale, coff, cscale):
    """Pure-torch reference implementation, kept indefinitely (see
    _nv12_to_rgb_torch_ref)."""
    import torch

    rgb255 = (rgb.to(torch.float32) * 255.0).clamp(0, 255)
    r, g, bch = rgb255[..., 0], rgb255[..., 1], rgb255[..., 2]

    # Invert the full-range YCbCr matrix: solve for Y, Cb, Cr given R,G,B and
    # (a,b,c,d) -- standard ITU-R inversion, Kg recovered implicitly via the
    # same (Kr,Kb) this matrix's a/b/c/d were derived from.
    yf = kr * r + kg * g + kb * bch
    cb = (bch - yf) / b
    cr = (r - yf) / a

    yf = yf / lscale + loff
    cb = cb / cscale + coff
    cr = cr / cscale + coff

    y = yf.round().clamp(0, 255).to(torch.uint8)
    # 4:2:0 subsample chroma via 2x2 average (pre-rounding, on the full-res
    # planes) -- avoids aliasing vs. naive top-left-pixel decimation.
    cb2 = cb.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-1, -2))
    cr2 = cr.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-1, -2))
    uv = torch.stack([cb2, cr2], dim=-1).round().clamp(0, 255).to(torch.uint8)
    return y, uv


# --------------------------------------------------------------------------
# CPU (numpy) reference -- independent implementation (no torch) used by
# tests to catch torch-specific broadcasting/indexing bugs, and usable
# directly as the CPU fallback backend's conversion path (backend.py).
# --------------------------------------------------------------------------


def _chroma_upsample_numpy(v: np.ndarray, H: int, W: int, siting: str, mode: str) -> np.ndarray:
    """numpy twin of _chroma_upsample_torch (see its docstring and this
    module's top-of-file docstring for the position algebra) -- an
    independent implementation (no torch broadcasting shared with the
    torch path) so test_colour.py's torch-vs-numpy comparison actually
    catches a torch-specific indexing bug, not just a shared formula
    bug."""
    h2, w2 = v.shape
    if mode == "nearest":
        out = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
        return out[:H, :W]
    if mode != "interpolated":
        raise ValueError(f"unsupported chroma_upsample: {mode!r}")

    prev_row = np.concatenate([v[:1], v[:-1]], axis=0)
    next_row = np.concatenate([v[1:], v[-1:]], axis=0)
    vtop = 0.25 * prev_row + 0.75 * v
    vbot = 0.75 * v + 0.25 * next_row

    if siting == "left":
        top_right_nb = np.concatenate([vtop[:, 1:], vtop[:, -1:]], axis=1)
        bot_right_nb = np.concatenate([vbot[:, 1:], vbot[:, -1:]], axis=1)
        top_even, top_odd = vtop, 0.5 * vtop + 0.5 * top_right_nb
        bot_even, bot_odd = vbot, 0.5 * vbot + 0.5 * bot_right_nb
    elif siting == "center":
        top_prev = np.concatenate([vtop[:, :1], vtop[:, :-1]], axis=1)
        top_next = np.concatenate([vtop[:, 1:], vtop[:, -1:]], axis=1)
        bot_prev = np.concatenate([vbot[:, :1], vbot[:, :-1]], axis=1)
        bot_next = np.concatenate([vbot[:, 1:], vbot[:, -1:]], axis=1)
        top_even, top_odd = 0.25 * top_prev + 0.75 * vtop, 0.75 * vtop + 0.25 * top_next
        bot_even, bot_odd = 0.25 * bot_prev + 0.75 * vbot, 0.75 * vbot + 0.25 * bot_next
    else:
        raise ValueError(f"unknown chroma_location: {siting!r}")

    out = np.empty((h2 * 2, w2 * 2), dtype=v.dtype)
    out[0::2, 0::2] = top_even
    out[0::2, 1::2] = top_odd
    out[1::2, 0::2] = bot_even
    out[1::2, 1::2] = bot_odd
    return out[:H, :W]


def nv12_to_rgb_numpy(y: np.ndarray, uv: np.ndarray, tags: ColourTags,
                       out_dtype: str = "float32",
                       chroma_upsample: str = "interpolated") -> np.ndarray:
    """Independent (no torch) reference, and the exactness target for
    nv12_to_rgb's HIP kernel/torch-ref float path (see test_colour.py).
    chroma_upsample="interpolated" (default) uses tags.chroma_location's
    geometry-correct filter; "nearest" keeps the old nearest-neighbour
    replication as an explicit opt-in -- see this module's docstring."""
    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)
    H, W = y.shape

    yf = y.astype(np.float64)
    u = _chroma_upsample_numpy(uv[..., 0].astype(np.float64), H, W, tags.chroma_location, chroma_upsample)
    v = _chroma_upsample_numpy(uv[..., 1].astype(np.float64), H, W, tags.chroma_location, chroma_upsample)

    yf = (yf - loff) * lscale
    u = (u - coff) * cscale
    v = (v - coff) * cscale

    r = yf + a * v
    g = yf - c * u - d * v
    bch = yf + b * u
    rgb = np.stack([r, g, bch], axis=-1)
    np.clip(rgb, 0, 255, out=rgb)

    if out_dtype == "float32":
        return (rgb / 255.0).astype(np.float32)
    if out_dtype == "uint8":
        return np.round(rgb).astype(np.uint8)
    raise ValueError(f"unsupported out_dtype: {out_dtype!r}")


def p010_to_rgb_numpy(y: np.ndarray, uv: np.ndarray, tags: ColourTags,
                       out_dtype: str = "float32",
                       chroma_upsample: str = "interpolated") -> np.ndarray:
    """P010 (10-bit-in-16, top-bit-justified) twin of nv12_to_rgb_numpy --
    the exactness target for p010_to_rgb's HIP kernel/torch-ref float
    path. The `>> 6` unpack is lossless (float64 easily holds a 10-bit
    integer exactly), so full 10-bit precision survives into the matrix
    math untouched."""
    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)
    loff *= 4.0
    coff *= 4.0
    H, W = y.shape

    yf = (y.astype(np.int64) >> 6).astype(np.float64)
    uf = _chroma_upsample_numpy((uv[..., 0].astype(np.int64) >> 6).astype(np.float64),
                                 H, W, tags.chroma_location, chroma_upsample)
    vf = _chroma_upsample_numpy((uv[..., 1].astype(np.int64) >> 6).astype(np.float64),
                                 H, W, tags.chroma_location, chroma_upsample)

    yf = (yf - loff) / 4.0 * lscale
    uf = (uf - coff) / 4.0 * cscale
    vf = (vf - coff) / 4.0 * cscale

    r = yf + a * vf
    g = yf - c * uf - d * vf
    bch = yf + b * uf
    rgb = np.stack([r, g, bch], axis=-1)
    np.clip(rgb, 0, 255, out=rgb)

    if out_dtype == "float32":
        return (rgb / 255.0).astype(np.float32)
    if out_dtype == "uint8":
        return np.round(rgb).astype(np.uint8)
    raise ValueError(f"unsupported out_dtype: {out_dtype!r}")


def rgb_to_nv12_numpy(rgb: np.ndarray, tags: ColourTags) -> tuple[np.ndarray, np.ndarray]:
    """Independent (no torch) numpy fp64 reference for rgb_to_nv12 -- the
    exactness target for the HIP kernel's fp64 formula (and, by the
    fixed-vs-fp64-forced bit-exactness tests, transitively for the Phase
    10 fixed-point fast path). `_rgb_to_nv12_torch_ref` (the pre-existing
    torch CPU reference) is NOT a safe exactness target: multiplying a
    float32 torch tensor by a python float keeps the result in float32,
    not float64 -- that reference's ENTIRE chain runs in float32, a
    pre-existing precision gap from the GPU kernel's always-double
    formula, discovered while deriving this phase (see IMPLEMENTATION.md
    Phase 10's "pre-existing finding"). `nv12_to_rgb_numpy` had this
    float64 reference already (decode direction); `rgb_to_nv12` (encode)
    never did until now.

    rgb: (H,W,3) float32 array in [0,1] (H,W even). Returns (y, uv)
    uint8 arrays, NV12 layout -- same contract as rgb_to_nv12/
    _rgb_to_nv12_torch_ref/rgb_to_nv12_kernel.

    Op order matches _colour_hip_ext.py's rgb_to_nv12_kernel EXACTLY
    (not just "the same formula" -- the same sequence of double
    operations, since this repo's bit-exactness bar is 0 LSB, and
    floating-point is not associative): per-channel double widen +
    *255.0 + clamp; yf = (kr*r + kg*g) + kb*bch (left-to-right, matching
    C++'s `kr*r + kg*g + kb*bch`); per-pixel y = round(clip(yf/lscale +
    loff)); cb/cr accumulated via SEQUENTIAL addition in the kernel's
    exact corner visiting order ((dy=0,dx=0),(dy=0,dx=1),(dy=1,dx=0),
    (dy=1,dx=1)), each corner's cb/cr computed from THAT corner's own yf
    (not the block-averaged one); then cb_avg = round(clip(cb_sum*0.25/
    cscale + coff)) (and cr symmetric) -- the kernel's own "average
    pre-offset double values, then rescale/round once" order, not
    `.mean()` of post-offset per-pixel values (a DIFFERENT, non-bit-
    -identical order for floating point)."""
    a, b, c, d = _rgb_coeffs(tags)
    loff, lscale, coff, cscale = _range_scale(tags)
    kr, kb = _KR_KB[tags.matrix]
    kg = 1.0 - kr - kb

    H, W = rgb.shape[0], rgb.shape[1]
    Hc, Wc = H // 2, W // 2
    rgb64 = rgb.astype(np.float64) * 255.0
    np.clip(rgb64, 0.0, 255.0, out=rgb64)
    r_full, g_full, b_full = rgb64[..., 0], rgb64[..., 1], rgb64[..., 2]

    yf_full = (kr * r_full + kg * g_full) + kb * b_full
    yfs_full = yf_full / lscale + loff
    np.clip(yfs_full, 0.0, 255.0, out=yfs_full)
    y_out = np.rint(yfs_full).astype(np.uint8)

    corners = []  # (r,g,b,yf) per corner, in the kernel's visiting order
    for dy in (0, 1):
        for dx in (0, 1):
            rc = r_full[dy:dy + Hc * 2:2, dx:dx + Wc * 2:2]
            gc = g_full[dy:dy + Hc * 2:2, dx:dx + Wc * 2:2]
            bc = b_full[dy:dy + Hc * 2:2, dx:dx + Wc * 2:2]
            yfc = (kr * rc + kg * gc) + kb * bc
            corners.append((rc, gc, bc, yfc))

    cb_sum = np.zeros((Hc, Wc), dtype=np.float64)
    cr_sum = np.zeros((Hc, Wc), dtype=np.float64)
    for rc, gc, bc, yfc in corners:
        cb_sum = cb_sum + (bc - yfc) / b
        cr_sum = cr_sum + (rc - yfc) / a

    cb_avg = cb_sum * 0.25 / cscale + coff
    cr_avg = cr_sum * 0.25 / cscale + coff
    np.clip(cb_avg, 0.0, 255.0, out=cb_avg)
    np.clip(cr_avg, 0.0, 255.0, out=cr_avg)
    uv_out = np.stack([np.rint(cb_avg), np.rint(cr_avg)], axis=-1).astype(np.uint8)
    return y_out, uv_out
