# Experimental VKL forward profiles

The initial four donor modules have reusable Slang profiles. Three more donor
modules (four new entries) are covered by [geometry/depth fusion profiles](vkl-geometry-depth.md). This is **not** a port of
all 47 substantive retained reference modules (48 Python files including the package initializer) or of the complete tracker. All profiles remain
experimental; there is no runtime selection. The donor revision is
`3384023a311651ea4260cf12ddab1567972f89f3`. Original owner code is Apache-2.0 by
explicit authorization; imported dependency notices remain applicable.

The caller owns validation, device, buffers, allocations, command encoding,
submission and synchronization. Product shaders use only typed structured buffers
and constant parameters. Device creation and readback occur only in opt-in tests
and benchmark code. Public names below are shader binding names.

All buffers are contiguous scalar arrays. Floating storage is IEEE float32;
logical storage is **uint32**, where zero is false and nonzero is true. Outputs
are canonical zero/one. Convert Torch bool masks with a typed conversion to
uint32 before the transaction; never bind a one-byte boolean allocation as a
uint32 buffer. `sintergpu.references.vkl.mask_u32` performs explicit CPU value
conversion for fixtures. No product host adapter copies or repacks inputs.
Buffers must not alias outputs or each other in ways that produce data races.
The caller skips zero-sized dispatches. Every active output element is overwritten;
tail elements beyond the logical count are untouched.

## Projection and depth evidence

`project_depth_evidence.slang::projectDepthEvidence`, 256 threads per group;
dispatch `(B*N,1,1)` logical threads. This is the float32 port of the actual
retained Triton kernel, checked against an independent scalar NumPy oracle and
the retained Torch reference.

| Binding | Type and layout |
| --- | --- |
| `xyz` | float32 `[B,N,3]`, xyz interleaved |
| `visible`, `provisional`, `depthCandidate` | uint32 `[B,N]` logical masks |
| `worldToCamera` | float32 `[B,4,4]`, row major, world-to-camera rigid transform |
| `intrinsics` | float32 `[B,3,3]`, row major; uses fx,fy,cx,cy |
| `sensorDepth` | float32 `[B,1,H,W]`, metric depth |
| `eligibleOut`, `projectedOut`, `sensorOut`, `supportOut`, `conflictOut` | uint32 `[B,N]` |
| `residualOut` | float32 `[B,N]`, absolute metric residual or zero |

`ProjectionParams`: uint32 `batches`, `points`, `height`, `width`; float32
`supportThreshold`, `conflictThreshold`, `epsilon`. Set epsilon to input dtype
machine epsilon (`1.1920928955078125e-7` for this float32 profile), **not** an
arbitrary geometry tolerance. Thresholds must be positive finite. Validate camera
shape, finite rigid matrices and positive finite focal lengths at the caller's
transaction boundary, as for donor `validate=False`.

Eligibility is visible AND (provisional OR depthCandidate) AND finite xyz.
Finite z must exceed epsilon. Nearest pixel lookup uses ties-to-even, including
negative half ties. Nonfinite projections and out-of-range rounded coordinates
produce no sensor evidence. A float bounds test precedes integer conversion,
avoiding undefined casts for huge finite projections. H and W must be positive
and <=2^24; flattened buffer offsets/products must fit uint32. Sensor depth must
be finite and positive. Support uses strict `abs(z-depth) < supportThreshold`;
conflict uses strict `z < depth-conflictThreshold` and the depthCandidate mask.
No landmark mutation, persistence, per-view de-duplication or scheduling is included.

## Photometric ISP observed RGB

`photometric_isp.slang::photometricISP`, 256 threads per group;
dispatch `(B*H*W,1,1)`. Bind float32 `rgbLinear` and `rgbObserved` in planar
`[B,3,H,W]` order; `exposureEV` `[B]`, `rgbGain` `[B,3]`, `vignette` `[B,3,3]`,
`colorMatrix` `[B,3,3]` row major, `responseGamma` `[B]`.

`PhotometricParams`: uint32 `batches`, `height`, `width`, `useExposure`, `useGain`,
`useVignette`, `useColor`, `useGamma`; float32 `epsilon` (float32 machine epsilon).
Each flag zero disables its option; disabled inputs still need valid dummy buffer
bindings. Validate finite RGB in [0,1], finite exposure and matrix, nonnegative
finite gains/vignette, positive finite gamma. Offset products must fit uint32.

Order is exp2 exposure, per-channel gain, exp(-vignette polynomial in normalized
radius squared), color matrix, lower clamp, optional
`pow(raw+epsilon,gamma)-pow(epsilon,gamma)`, final clamp [0,1]. Singleton normalized
image dimensions use -1, matching Torch linspace. No image-wide diagnostic
reductions, clipping ratios or backward/gradient profile are supplied. Gradient
support is untested and is **not claimed**.

## Depth numeric conversion

`depth_convert.slang::depthConvert`, 256 threads per group; dispatch `(count,1,1)`.
Float32 `values`, `confidence`; uint32 `valid`. Float32 `depthOut`, `inverseOut`,
`confidenceOut`; uint32 `validOut`, all `[count]`.

`DepthConvertParams`: uint32 `count`, `representation`; float32 `scale`,
`focalBaseline`. Representation 0 multiplies source depth by scale, 1 reciprocates
scaled inverse depth, 2 divides focalBaseline by disparity. Supply positive finite
scale/focalBaseline, finite confidence in [0,1], representation in {0,1,2}.
Validity requires input mask, finite positive source and finite positive converted
depth. Invalid outputs are zero. Metric-status tags, defaults and validation are
caller policy, separate from this numeric profile.

**Known failing domain:** SlangPy 0.43.1's compiler defaults flush denormals on the
local RADV GPU. The strict 1e-38/1e38 regression remains expected-failure in all
three representations; it is not a pass or qualification. Precise floating-point
mode alone does not preserve these values. Numerical comparisons pass for the
normal-range fixture (1e-20..1e20 plus invalid values). General float32 conversion
parity remains pending explicit denormal preservation and reciprocal verification.
The Slang CLI provides `-denorm-mode-fp32 preserve`; SlangPy's public 0.43.1
compiler-options binding does not expose this switch. See the
[Slang compiler reference](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md).

## One depth-pyramid level

`depth_pyramid.slang::depthPyramidLevel`, 256 threads per group;
dispatch `(B*ceil(H/2)*ceil(W/2),1,1)`. Float32 `depth`, `confidence` and uint32
`valid` are `[B,1,H,W]`; matching outputs `depthOut`, `confidenceOut`, `validOut`
are `[B,1,ceil(H/2),ceil(W/2)]`.

`DepthPyramidParams`: uint32 `batches`, `height`, `width`; float32 `epsilon`.
Validate confidence finite in [0,1], positive dimensions and uint32-safe products.
The four children are visited row-major; first minimum wins confidence ties.
Invalid/padded children do not count. All-four-valid depth range <=32*epsilon
uses mean depth/confidence; otherwise closest depth and its confidence win.
This deliberately preserves the donor's masked-INF maximum behavior: a missing
child disables the mean even if every remaining depth agrees. Empty blocks emit
zeros and invalid. Base sanitization and the caller's /2,/4,/8 dispatch chain
remain separate; this is one reusable level, not the complete donor function.

## Verification and remaining work

Independent hot-path review preceded GPU tests (`/tmp/sintergpu-vkl-perf-review.md`).
`tests/test_vkl_hot_path.py` guards typed mask ABI, complete stores, strict
thresholds, projection cast bounds and absence of unbounded shader loops; deliberate
mutations must fail. CPU command:

```
PYTHONPATH=src python -m pytest tests/test_vkl_ports.py tests/test_vkl_hot_path.py -q
```

On the host Torch environment: 15 passed, 15 GPU cases skipped. GPU command:

```
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src /home/bjoern/git/caspar-slang/.venv/bin/python -m pytest tests/test_vkl_ports.py tests/test_vkl_hot_path.py -q
```

On local AMD Radeon 8060S / RADV: 21 passed, 6 Torch-oracle skips (Torch absent
from that read-only interpreter), **3 expected failures** for extreme depth
conversion. Fixtures cover irregular counts, output padding, two camera batches,
random rigid transform, invalid/nonfinite points/sensors, huge projections,
positive/negative half ties, strict threshold ties, options and singleton images.
All four shaders compile to SPIR-V with standalone Slang 2026.12 and 2026.19.
SlangPy 0.43.1 uses bundled compiler `libslang-compiler.so.0.2026.12`, SHA256
`195f1fbbf7342517aed3064657c8cfaaae5791065158189610253dce070c4678`.

The opt-in `benchmarks/vkl_projection.py` times resident dispatch via GPU timestamps
and separately reports host record/submit/wait. One 307201-point run measured
0.02238 ms median, 0.02308 ms p95 resident GPU and 0.06971 ms amortized host wall.
These are single-device measurements, excluding upload/readback/transfer latency;
there is no baseline, confidence bound, speedup, fusion or cross-vendor claim.
Run three controlled trials and compare valid baselines before performance adoption.

The other 40 substantive retained reference modules have no Slang translation here, including alignment, temporal/reprojection, compaction,
voxel/mesh, pose reductions and tracking/landmark policy. The retained Triton
argmin has no new Slang port here. The photometric reductions/backward and full
pyramid dispatch wrapper are also pending. RTX3090 verification is pending.
