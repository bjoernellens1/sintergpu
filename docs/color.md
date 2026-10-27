# Camera color and bilinear sampling profiles

`kernels/image/camera_color.slang` implements six experimental Vulkan compute
entrypoints. Every input/output is borrowed. The caller allocates resources,
validates requests, records dispatch, supplies producer/consumer barriers and
owns submission, synchronization and any transfers. The reference module creates
no runtime resources. The test/benchmark harness intentionally owns its runtime.

| Entry | Input | Output | Thread extent |
|---|---|---|---|
| `nv12Decode` | Y plane + interleaved UV plane | packed RGB uint8 | camera width,height |
| `nv12Encode` | packed RGB uint8 | Y plane + interleaved UV plane | width/2,height/2 |
| `yuyvDecode` | Y0,U,Y1,V pairs | packed RGB uint8 | camera width,height |
| `remapBilinear` | packed RGB + calibrated float2 map | packed RGB uint8 | output width,height |
| `resizeBilinear` | packed RGB | packed RGB uint8 | output width,height |
| `fusedYUYVRemap` | Y0,U,Y1,V pairs + calibrated map | packed RGB uint8 | output width,height |

All entries use `[numthreads(16,16,1)]` and bounds-check logical coordinates.
NV12 width and height must be even; YUYV camera width must be even. Remap/resize
output dimensions may be odd. A 1-pixel source dimension is valid for sampling.
NV12 and YUYV conversion preserve camera extents; they are not resize operations.

`Params` is a constant buffer of ten uint32 fields, in this order:
`srcWidth,srcHeight,dstWidth,dstHeight,sourcePitch,uvPitch,outputPitch,outputUVPitch,matrix709,limited`.
Pitches are bytes. `matrix709` and `limited` are exactly zero or one. Set constants
through `references.color.validate_request(..., for_dispatch=True)` and supply
all relevant buffer capacities. The helper rejects wrong geometry, undersized or
overflowing pitches/capacities, missing calibration, and wrong map shape/type.
CPU oracle calls perform their own capacity checks and output allocation; their
metadata-only validation does not require a preexisting output allocation.

The source/UV bindings are `ByteAddressBuffer sourceBytes,uvBytes`; destination
bindings are `RWByteAddressBuffer outputBytes,outputUV`. Bind only resources used
by the chosen entry. `StructuredBuffer<float2> mapXY` is tightly packed row-major
output-height by output-width, with coordinates measured in source pixel-center
indices. Its storage is contiguous float32 pairs. Calibration generation and
camera model interpretation remain caller-owned. A map is mandatory for remap
and fusion; no identity calibration is silently assumed.

Source, destination, map and intermediate resources must not alias. Output rows
must not overlap. Every byte-addressed allocation must have **4-byte-padded
capacity**, even when the last logical pixel needs fewer bytes. Non-word-aligned
row pitches are supported. RGB uses exactly three packed bytes per pixel; it is
not a four-byte uint32-pixel format. Only live pixel bytes are written; row and
allocation padding retain their original values, including nonzero sentinels.

## Numerical contract

`ColorSpec` requires BT.601 (`Kr=.299,Kb=.114`) or BT.709
(`Kr=.2126,Kb=.0722`), full or limited range. Primaries/transfer conversions are
outside this operation. Full chroma offset is128. Limited decode uses Y offset16,
luma scale255/219 and chroma scale255/224. Encode applies the inverse range scale.
Both directions clamp to[0,255] then round to nearest integer with ties to even.
No gradients are claimed.

The initial NV12 decode chroma profile is explicitly `cosited-nearest`: each
interleaved UV pair is replicated over its2x2 luma block. This does not implement
interpolated MPEG left/center reconstruction. Encode averages four unrounded
chroma contributions over the2x2 block before quantizing: its chroma siting is
**centered box**, so encode followed by nearest decode is a lossy subsampled
profile, not a lossless siting-preserving round trip. Locus planar float YUV is a
different format and is not substituted for NV12.

The new shader profile is `portable-fp32`; compile with precise floating-point
mode. Independent scalar/vector NumPy float64 oracles allow a maximum difference
of **one uint8 LSB**. The retained EliGSiR-sim HIP exact implementation and its
fixed-point/FP64 guard-band contract remain separate; requesting `hip-exact` here
fails explicitly. No portable FP32 result satisfies an exact-HIP request.

Sampling uses clamp-to-edge for finite coordinates. Nonfinite map coordinates
produce black pixels. Resize uses half-pixel centers:
`((output_index+.5)*source_extent/output_extent)-.5`.
Bilinear interpolation uses four decoded/loaded RGB uint8 corners and rounds the
final result to nearest-even. Fusion first decodes and quantizes **each corner**
to uint8, then filters those corners. Filtering unrounded decoded YUV/RGB would
change the contract and is forbidden. Shared device helpers preserve exact
fused-versus-separate output bytes, independently of the FP64-oracle tolerance.

## Packed-byte baseline and performance boundary

Each output byte has one writer. A byte write uses atomic word AND to clear only
its byte mask, then atomic OR to install that byte. Disjoint byte masks commute,
so adjacent pixels or rows sharing a32-bit word cannot erase one another.
No output zero initialization is required. An ordinary word read/modify/write
would race and is guarded against by mutation-sensitive tests.

This is a correctness-first experimental baseline: packed RGB costs six atomic
operations per pixel, neighboring byte accesses repeat word loads, and fusion
may decode four corners per output. No optimized packed-word path is claimed.

Independent source and harness review cleared the bounded local experiment
before GPU execution. On AMD Radeon8060S/RADV Mesa26.2.3 with SlangPy0.43.1,
bundled Slang2026.12 in precise mode, all six entries passed small GPU fixtures:
four matrix/range combinations, padded/unaligned pitches, odd outputs, nonfinite
and large finite coordinates, exact fused/separate bytes and preserved padding.
Both pinned standalone Slang2026.12 and2026.19 compile/reflection lanes passed.
Standalone2026.19 execution and NVIDIA execution remain pending.

The 720p and1080p experiments each ran three runs of100 samples after20warmups.
All run outputs were byte-identical between fused and separate, including
padding. Resident timestamps cover the entire pipeline, including an explicit
separate decode-to-remap global barrier. Host recording/submission/completion is
reported separately. Transfer-inclusive samples upload both packed source and
map, execute, wait and download the packed result, with reused allocations.

| Size | Resident p50 gain, runs1/2/3 | Transfer-inclusive p50 gain, runs1/2/3 | Selection |
|---|---|---|---|
|720p|39.3%,48.0%,44.6%|1.3%,1.0%,3.6%|separate fallback|
|1080p|41.4%,59.1%,59.7%|-0.2%,2.7%,0.6%|separate fallback|

These are **exploratory observations**, not qualified speedups. No exclusive GPU
window was established. ROCm process evidence identified concurrent native
GPU-test Python processes in another workspace; contamination cannot be
excluded. At1080p, resident runs1 and3 also had bootstrap lower gain bounds below
zero. Every transfer-inclusive run failed the required5%median gain. The adoption
gate remains at least5%median improvement, positive95%bootstrap lower bound and
at most5%p95 regression across all three runs, with qualified controlled evidence.
The default remains the separate pipeline; no auto-selection is enabled.

Raw local records are `/tmp/sintergpu-color-720p.json` and
`/tmp/sintergpu-color-1080p.json`; the implementation report records hashes,
compiler identity, review pointer and limitations. The synthetic calibrated map
is a shared test fixture, not real camera calibration or application evidence.

```bash
PYTHONPATH=src python -m pytest tests/test_color.py tests/test_color_hot_path.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src /home/bjoern/git/caspar-slang/.venv/bin/python \
  -m pytest tests/test_color.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src /home/bjoern/git/caspar-slang/.venv/bin/python \
  benchmarks/color_pipeline.py --width 1280 --height 720 --runs 3 --samples 100 \
  --warmups 20 --output /tmp/color-720p.json
```

