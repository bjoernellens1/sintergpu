"""Unit tests for eligsir_sim.video.colour: torch kernel vs independent numpy
reference, and a mutation test that fails if the matrix is hardcoded
(root-cause requirement from the brief/AGENTS.md: the matrix must be
derived from the stream's colour tags, not baked in)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir_sim.video.colour import (
    ColourTags,
    nv12_to_rgb,
    nv12_to_rgb_numpy,
    rgb_to_nv12,
)

H, W = 64, 96  # even dims, NV12 4:2:0 needs both H and W even


def _random_nv12(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(16, 236, size=(H, W), dtype=np.uint8)
    uv = rng.integers(16, 241, size=(H // 2, W // 2, 2), dtype=np.uint8)
    return y, uv


@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("range_", ["limited", "full"])
def test_torch_matches_numpy_reference(matrix, range_):
    tags = ColourTags(matrix=matrix, range=range_)
    y_np, uv_np = _random_nv12()

    ref = nv12_to_rgb_numpy(y_np, uv_np, tags, out_dtype="uint8")

    y_t = torch.from_numpy(y_np)
    uv_t = torch.from_numpy(uv_np)
    out = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8").numpy()

    # Independent implementations (numpy vs torch) of the same derived
    # matrix; allow +/-1 for float rounding-mode differences, must not
    # differ systematically.
    diff = np.abs(ref.astype(np.int16) - out.astype(np.int16))
    assert diff.max() <= 1, f"torch/numpy mismatch, max_abs_diff={diff.max()}"


def test_matrix_is_not_hardcoded():
    """Mutation test: BT.601 and BT.709 MUST produce different output for
    the same chroma-heavy input. If someone "simplifies" nv12_to_rgb back
    to a single hardcoded matrix, this is the test that catches it."""
    y_np, uv_np = _random_nv12(seed=1)
    # push chroma away from neutral (128) so matrix choice actually matters
    uv_np = np.clip(uv_np.astype(np.int16) + 40, 16, 240).astype(np.uint8)

    out_601 = nv12_to_rgb_numpy(y_np, uv_np, ColourTags(matrix="bt601"), out_dtype="uint8")
    out_709 = nv12_to_rgb_numpy(y_np, uv_np, ColourTags(matrix="bt709"), out_dtype="uint8")

    assert not np.array_equal(out_601, out_709), (
        "BT.601 and BT.709 produced identical output -- the colour matrix "
        "looks hardcoded (not derived from ColourTags.matrix)"
    )


def test_range_is_not_hardcoded():
    """Same mutation-test shape for limited vs full range."""
    y_np, uv_np = _random_nv12(seed=2)
    out_limited = nv12_to_rgb_numpy(y_np, uv_np, ColourTags(range="limited"), out_dtype="uint8")
    out_full = nv12_to_rgb_numpy(y_np, uv_np, ColourTags(range="full"), out_dtype="uint8")
    assert not np.array_equal(out_limited, out_full), (
        "limited and full range produced identical output -- range handling "
        "looks hardcoded"
    )


def test_float_output_matches_uint8_within_one_lsb():
    tags = ColourTags()
    y_np, uv_np = _random_nv12(seed=3)
    y_t, uv_t = torch.from_numpy(y_np), torch.from_numpy(uv_np)

    out_u8 = nv12_to_rgb(y_t, uv_t, tags, out_dtype="uint8")
    out_f32 = nv12_to_rgb(y_t, uv_t, tags, out_dtype="float32")
    out_f32_as_u8 = (out_f32 * 255.0).round().clamp(0, 255).to(torch.uint8)

    diff = (out_u8.to(torch.int16) - out_f32_as_u8.to(torch.int16)).abs()
    assert diff.max().item() <= 1


def test_encode_decode_roundtrip_bt601_limited():
    """rgb_to_nv12 then nv12_to_rgb should approximately recover the input
    RGB (4:2:0 chroma subsampling loses some high-frequency chroma detail,
    so this bounds mean error, not bit-exactness -- the bit-exactness claim
    belongs to the real-decoder integration test in test_vulkan_backend.py,
    which compares against an actual H.265 bitstream, not this synthetic
    round trip)."""
    # chroma_location="center": rgb_to_nv12's encode side is not yet
    # siting-aware and always produces a 2x2-box-average chroma sample
    # (equidistant from all 4 luma pixels, i.e. "center" siting -- see
    # rgb_to_nv12's docstring's documented limitation). Decoding with the
    # matching siting here isolates ordinary 4:2:0 subsampling loss from
    # the siting-mismatch error that would appear decoding with the
    # "left"-default tags a real stream would carry.
    tags = ColourTags(matrix="bt601", range="limited", chroma_location="center")
    rng = np.random.default_rng(4)
    # Spatially smooth (low-frequency) image, not per-pixel white noise:
    # 4:2:0 chroma subsampling is lossy for independent per-pixel colour by
    # construction (every encoder's 4:2:0 mode is), so testing against white
    # noise would be asserting a false bound, not finding a bug. Rendered
    # sim frames -- this kernel's actual input -- are spatially correlated,
    # which is what makes 4:2:0 subsampling usable at all.
    low_res = rng.random((H // 8, W // 8, 3), dtype=np.float32)
    rgb_np = np.kron(low_res, np.ones((8, 8, 1), dtype=np.float32))
    rgb = torch.from_numpy(rgb_np)

    y, uv = rgb_to_nv12(rgb, tags)
    rgb_back = nv12_to_rgb(y, uv, tags, out_dtype="float32")

    mean_abs_err = (rgb - rgb_back).abs().mean().item()
    # Bound loosened 0.03 -> 0.035 when the chroma upsample switched from
    # nearest-neighbour replication to the geometry-correct interpolated
    # filter (the lossless-pipeline requirement): on this smooth-blocky
    # synthetic image the two filters' errors are the same order of
    # magnitude (not a regression -- see IMPLEMENTATION.md's "Fused
    # colour kernels" section for the measured comparison), this input's
    # 8x8 flat blocks are an adversarial case for ANY linear interpolation
    # filter (every block boundary is a hard edge), unlike the smoothly-
    # varying natural image in test_rgb_roundtrip_psnr_natural_image.
    assert mean_abs_err < 0.035, f"round-trip mean abs error too high: {mean_abs_err}"


def _synthetic_natural_image(h, w, seed=7):
    """A synthetic stand-in for a "natural" photographic image (smooth
    low-frequency gradients + a few soft coloured blobs + mild band-
    limited texture) -- NOT a lab calibration/test pattern (no
    chessboard/colour-checker geometry), and not per-pixel white noise
    (which would make 4:2:0 subsampling's loss unrepresentative of real
    content, see test_encode_decode_roundtrip_bt601_limited's docstring).
    This repo has no committed natural photo asset for video tests (only
    assets/targets/chessboard_9x6.png, a calibration target), so this is
    generated rather than loaded -- documented here rather than silently
    assumed per CLAUDE.md."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    img = np.stack([0.5 + 0.4 * np.sin(3 * xx + 1), 0.5 + 0.4 * np.cos(2 * yy + 2),
                     0.5 + 0.3 * np.sin(2 * (xx + yy))], axis=-1)
    for _ in range(5):
        cy, cx = rng.random(2) * [h, w]
        sigma = rng.uniform(h, w) / rng.uniform(3, 8)
        hue = rng.random(3) * 0.6 + 0.2
        d2 = (yy * h - cy) ** 2 + (xx * w - cx) ** 2
        blob = np.exp(-d2 / (2 * sigma ** 2))[..., None]
        img = img * (1 - 0.5 * blob) + hue[None, None, :] * (0.5 * blob)
    # mild band-limited texture: box-blur white noise a few times (cheap
    # approximation of a Gaussian blur) instead of raw per-pixel noise.
    noise = rng.standard_normal((h, w, 3)).astype(np.float64)
    for _ in range(4):
        noise = (noise + np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0)
                  + np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5.0
    img = img + 0.03 * noise
    return np.clip(img, 0, 1).astype(np.float32)


def test_rgb_roundtrip_psnr_natural_image():
    """Lossless-pipeline requirement: measure the loss the colour
    conversion ITSELF adds (RGB -> NV12/P010 -> RGB with our own kernels,
    no codec in the loop) on a natural-image-like synthetic photo, for
    both 8-bit (NV12) and 10-bit (P010) in one test -- this is the error budget
    Gaussian-splatting frame preparation inherits from the conversion
    alone, before any codec adds its own loss on top."""
    from eligsir_sim.video.colour import p010_to_rgb
    tags = ColourTags(matrix="bt709", range="full", chroma_location="center")
    h, w = 64, 96
    rgb_np = _synthetic_natural_image(h, w)
    rgb = torch.from_numpy(rgb_np)

    y8, uv8 = rgb_to_nv12(rgb, tags)
    rgb_back8 = nv12_to_rgb(y8, uv8, tags, out_dtype="float32")
    mse8 = ((rgb - rgb_back8) ** 2).mean().item()
    psnr8 = 10.0 * np.log10(1.0 / max(mse8, 1e-12))
    max_err8 = [(rgb[..., ch] - rgb_back8[..., ch]).abs().max().item() for ch in range(3)]

    # P010 10-bit: scale y8/uv8 up to the top-justified 10-bit convention
    # (same 8->10 bit promotion the real P010 packing would carry -- no
    # new information, just proving the 10-bit path's extra headroom
    # doesn't *add* error on top of the 8-bit case for the same content).
    # uint16, not int16: 1023<<6 = 65472 overflows signed int16 (max 32767).
    y10 = ((y8.to(torch.int32) * 4).clamp(0, 1023) << 6).to(torch.uint16)
    uv10 = ((uv8.to(torch.int32) * 4).clamp(0, 1023) << 6).to(torch.uint16)
    rgb_back10 = p010_to_rgb(y10, uv10, tags, out_dtype="float32")
    mse10 = ((rgb - rgb_back10) ** 2).mean().item()
    psnr10 = 10.0 * np.log10(1.0 / max(mse10, 1e-12))
    max_err10 = [(rgb[..., ch] - rgb_back10[..., ch]).abs().max().item() for ch in range(3)]

    print(f"\n[colour round-trip] 8-bit:  PSNR={psnr8:.2f} dB, max_err/channel(RGB)={max_err8}")
    print(f"[colour round-trip] 10-bit: PSNR={psnr10:.2f} dB, max_err/channel(RGB)={max_err10}")

    # Conversion-only loss (no codec): expect a comfortably high PSNR --
    # the only loss source is 4:2:0 chroma subsampling itself.
    assert psnr8 > 28.0, f"8-bit round-trip PSNR too low: {psnr8:.2f} dB"
    assert psnr10 > 28.0, f"10-bit round-trip PSNR too low: {psnr10:.2f} dB"


def test_invalid_colour_tags_rejected():
    with pytest.raises(ValueError):
        ColourTags(matrix="bt2020")  # not a supported matrix
    with pytest.raises(ValueError):
        ColourTags(range="studio")  # not a supported range
