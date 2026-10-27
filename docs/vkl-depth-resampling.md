# VKL depth resampling and propagation

These experimental float32 forward profiles port depth lifting, conservative hole
filling, native pyramid initialization and direct-provenance labels from VKL
`3384023a311651ea4260cf12ddab1567972f89f3`. Full pyramid/downsample plans reuse the
existing `vkl.depth_pyramid.depthPyramidLevel` shader unchanged. Frozen Torch
functions, validation and autograd routes remain available unchanged.

## Storage and ownership

Depth/confidence images are contiguous float32 B1HW; RGB is planar BCHW with one
or three channels. Logical/provenance images use uint32 zero/nonzero input and
canonical zero/one output, not Torch's byte-sized bool/uint8 allocations. Support
is int32, numerically matching the donor's int64 counts in [0,9]. Provenance is
0=invalid, 1=direct and 2=propagated. Confidence inputs must be finite [0,1].

The application owns device, compilation/setup, buffers, descriptors, encoder,
submission, synchronization and lifetimes. Setup helpers produce parameters,
scalar32 capacities and logical dispatch sizes; they perform no GPU work and
import no GPU runtime. All entrypoints use 256x1x1 workgroups. Supply the actual
existing device X workgroup limit. `validate_capacities` does not establish types,
descriptor range limits, alignment, content contracts or alias safety: the binding
adapter/caller must validate those separately. Inputs and outputs cannot overlap
in ways that race. Zero-batch plans record no work.

`ResamplingParams` declares ten uint32 fields followed by four float32 fields:
`batches,height,width,outputHeight,outputWidth,channels,firstStep,useDirect,`
`useConfidence,minSupport,scaleX,scaleY,maxRGBDifference,maxRelativeSpread`.
Use compiler reflection for native binding and constant-buffer alignment. The
reused pyramid level retains its original four-field `DepthPyramidParams` ABI.

## Nearest depth lift

`depth_lift_layout(B,H,W,OH,OW,max_workgroups_x=...)` describes
`vkl.depth_resampling.depthUpsample`. Inputs: `depth`, `valid`, `confidence`.
Outputs: `depthOut`, `validOut`, `confidenceOut`, `direct_validOut`,
`propagated_validOut`, `provenanceOut`. It also supports a smaller requested size,
as does Torch's nearest interpolation.

Use `floor(float32(destination)*float32(source_extent/output_extent))`, clamped
to the last source coordinate. Setup supplies the scale so approximate GPU
reciprocal does not change lookup coordinates. Dimensions are limited to 2^24
for this float32-coordinate profile; uint32 element-product and dispatch limits
also apply. There is no half-pixel offset or bilinear mixing. Invalid,
nonpositive/nonfinite depth is zeroed; confidence is masked with the same usable
sample. Every lifted valid value is propagated, even for equal source/output
sizes: direct=false, propagated=true, provenance=2. This profile cannot replace
Locus image resize or bilinear remap merely because all resample an image.

## Full pyramid and downsample composition

`depth_pyramid_plan` returns `/1,/2,/4,/8` stages, stopping before a redundant
level after reaching 1x1. `/1` uses
`vkl.depth_resampling.initializeDepth` to sanitize native depth/valid/confidence.
If confidence is omitted (`confidence=False`), bind a caller-owned one-element
dummy; native confidence becomes usable validity converted to float32. Every
level has only the original three numerical outputs.

Subsequent stages use the released `depthPyramidLevel` source, ceil-half sizes,
first-minimum confidence and the donor's masked-INF maximum. Only a fully usable
2x2 neighborhood whose spread is within 32*float32_epsilon uses the mean.
Missing/invalid samples force minimum selection; padded edges are not silently
averaged. The full plan makes the previously standalone level usable without
host data round trips. The caller allocates/retains each requested level.

`depth_downsample_plan(...,scale=2|4|8,...)` returns `stages`, final `shape` and an
`outputs` map of output name to `{stage,binding}`. It appends
`vkl.depth_resampling.labelDirectDepth`, which reads the final `valid` mask and
writes `direct_validOut`, `propagated_validOut`, `provenanceOut`. Final depth,
validity and confidence **alias the existing level outputs**; no second copy is
recorded. Unavailable scales are rejected just as in the donor. The label pass
requires an already-sanitized level, not raw depth validity.

## Conservative RGB-guided filling

`depth_densify_plan(...,channels=1|3,steps=...,direct=False,...)` describes repeated
`vkl.depth_resampling.densifyDepth` calls. Inputs are `depth`, `valid`,
`confidence`, `direct_valid`, `rgb`. Outputs are depth/valid/confidence,
direct/propagated/provenance and `support`, with `Out` suffixes on every binding.

On the first pass, sanitize finite-positive depth. Default direct validity is
that initial usable mask; an explicit direct mask is intersected with initial
usability. The disabled direct input accepts a one-element caller-owned dummy.
Later passes bind the previous direct output as a full image. **Never rederive
direct validity from newly filled pixels.** `firstStep=0` requires initialized
state from a previous pass. Two distinct output sets suffice for ping-pong; the
caller retains RGB and establishes each storage dependency on the same encoder.

At each pixel, count usable neighbors in the clipped 3x3 stencil whose maximum
absolute per-channel RGB difference is inclusively within the RGB threshold.
For holes with enough support, select the lower median (index (support-1)//2),
then accept an inclusive relative spread bound. The acceptance test preserves
rounded float32 division using an exact integer comparison to the midpoint
between the threshold and its next float, including ties to even. It uses two
uint32 limbs, requiring neither float64 nor int64 shader capabilities; a raw
cross-product inequality would incorrectly omit the donor rounding step. Filled confidence retains the
donor expression `(sumConfidence/support)*(support/9)`, without algebraically
replacing it with `/9`. Current valid values/confidences remain unchanged.
Support is written for every pixel, including already-valid pixels. Zero
confidence does not independently invalidate depth. Nonfinite RGB comparisons
and NaN/Inf thresholds retain the donor behavior: NaN RGB thresholds reject
neighbors, NaN spread thresholds reject fills, and positive infinity is unbounded.
Python/NumPy real scalar thresholds are normalized during setup; negative and
boolean thresholds are rejected.

The stencil is O(pixels) per pass, with a fixed nine-float private array and a
hole-only 36-comparison sort. A source mutation guard enforces the exact 3x3
bounds, channel-loop form, sort bounds, tail checks and complete output stores;
there is no shared scratch, atomic, barrier or image-sized shader loop. Dynamic
private indexing may spill on a backend; register-only execution is not claimed.

## Verification and measurements

Differential tests use frozen Torch and an independent NumPy oracle. Downsample
oracles reuse the existing independent per-level reference rather than duplicating
its rules. GPU cases check all outputs and poisoned padding, 1/3-channel RGB,
1/2/4 passes, optional direct masks/confidence, NaN/Inf inputs/thresholds, exact
inclusive thresholds with adjacent float32 values, singleton/odd sizes, tails,
complete borrowed-encoder compositions and every retained pyramid level. Flat
0.5/1/2/4 depth is exact at every level. Float limits for other tested outputs are
rtol=8e-6, atol=5e-7; logical/provenance/count values are exact. General extremes,
gradients, other runtime/compiler/device configurations and application results
remain unqualified. No consumer default is changed.

```sh
PYTHONPATH=src python -m pytest tests/test_vkl_depth_resampling.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python -m pytest tests/test_vkl_depth_resampling.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/vkl_depth_resampling.py \
  --operation densify --steps 4 --pattern holes --output resampling.json
```

The explicit standalone benchmark preallocates and warms, records each isolated
kernel and complete resident pipeline, and separates host record/submit/wait from
full input upload/final output readback. Full-pyramid readbacks include all levels;
downsample readbacks include only its six final fields. Isolated-pass transfer
lists are empty and must not be confused with full transfer-inclusive pipeline
measurements. Raw samples, byte counts and source/setup/compiler/device/driver
identities are retained. Ten shared-device configurations cover steps 1/2/4,
dense/invalid/sparse/edge/hole fields, lifting, odd-size pyramids and direct labels.
These are measurements, not controlled-baseline speedup or adoption qualification.

C++ consumers use the same installed Slang modules through `sintergpu::kernels`
and their existing Vulkan/Daxa context. Use the reflected typed ABI and the stage
sizes above; Python is not required to compile or record these kernels. No native
runtime library is introduced by these profiles.
