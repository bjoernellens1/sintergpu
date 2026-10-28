# Experimental VKL depth and temporal reprojection

This0.2.5 working batch implements the complete numerical forward output sets of
frozen VKL `reproject.py` and `temporal_reproject.py` at
`3384023a311651ea4260cf12ddab1567972f89f3`. Independent scalar NumPy references
pass102 literal/frozen-donor comparisons. All nine new entries compile under
standalone Slang2026.12 and2026.19. New GPU execution, full-domain and controlled
performance qualification are still pending; consult the versioned qualification
record rather than inferring runtime support from compilation.

The donor's `depth_reproject` alias names the same projection operation; it is not
a fifth algorithm. There are four forward operations, two newly translated source
modules and nine dispatch stages, with released geometry/fusion stages reused.
No gradient, frame scheduling, keyframe ownership or tracking acceptance is added.

## Numerical contracts

| Operation | Every returned numerical output |
| --- | --- |
| Projection | projected_xy, projected_depth, visible, displacement, overlap, parallax; with target depth: depth_consistency, depth_consistency_valid, occluded, occlusion_valid |
| Rasterization | depth, valid, confidence, support_count |
| Temporal RGB alignment | aligned_rgb, aligned_valid, alignment_confidence |
| Prior-only temporal RGB-D reference | rgb, depth, valid, confidence, support_count, temporal_support_count, source_disagreement |

Depth backprojection reuses `vkl.geometry.depthGeometry` camera XYZ/validity.
The next stage transforms even an invalid source's zero camera point through the
source pose and target inverse rigid transform. Its raw diagnostic z/xy are
therefore retained. Projection divides by max(raw_z,float32 epsilon), while
visibility requires raw_z>0, source validity and inclusive floating pixel bounds.
Displacement compares against the original source pixel. Overlap divides visible
count by all source pixels; parallax divides the donor's displacement-times-mask
sum by max(count,1). Consequently0*Inf/NaN can remain in diagnostic aggregates.

Target-depth lookup preserves the explicit normalized float32 coordinate
roundtrip and nearest ties-to-even, zero padding, align_corners=True. Direct
rounding is not equivalent: pixel0.5 at width4 normalizes and returns as0.50000006,
selecting pixel1. Singleton axes retain the grid sampler's defined behavior.
Consistency requires visibility and finite positive sampled target depth;
occlusion is strict/directional: sample < raw_z-threshold.

Rasterization checks floating visibility before rounding/clamping. Staged uint32
atomics find minimum positive target-camera Z. Every source within
float32_epsilon*max(abs(min_z),1) contributes to support; confidence is their
maximum. Empty pixels emit zero depth/confidence/support and false validity.
Clear/minimum/support/finalize stages are mandatory on every invocation, including
reused scratch. No floating-point atomic extension is required.

Temporal alignment projects target depth into each source camera. RGB and optional
source confidence use planar float32 bilinear zero-padding and the normalized
roundtrip. `aligned_rgb` is not masked when aligned_valid=false. With source depth,
validity additionally requires consistency validity and no directional occlusion;
confidence multiplies visibility, exp(-residual/threshold), that mask and sampled
source confidence. This sampler cannot substitute for the quantized, clamped-border
YUYV/RGB remap profile.

Reference construction rasterizes priors, explicitly packs D9 source-major slots
and invokes unchanged `vkl.depth_fusion.temporalDepthFuse`. Slot0 is zero/invalid;
trusted flags are false. The current target depth guides RGB warping but never
supplies reference depth. RGB weights use all alignment_confidence*raster_confidence
*raster_valid*aligned_valid values, independently of D9's winning depth cluster.
Final validity also requires positive RGB weight. RGB/depth/confidence are masked;
support counts and disagreement retain D9 outputs even when only RGB invalidates
the final reference.

## Layouts and caller ownership

Float storage is contiguous float32. Logical masks are typed uint32, not Torch's
one-byte bool. Raster support is bounded uint32; D9 support outputs are bounded
int32, not the donor's int64 storage ABI. Convert explicitly at an output boundary
when a consumer requires int64. Values match within the documented capacity
profile; binding an int64 allocation to a scalar32 descriptor is invalid.

Projection uses source `[B,1,Hs,Ws]` depth, `[B,3,3]` intrinsics and `[B,4,4]` c2w
poses; target dimensions may differ. Diagnostic vector maps are planar B2/B1HW.
Raster outputs are target-grid B1HW. T0 uses target B3/B1HW and prior
`[B,N,C,H,W]` storage; camera arrays are B,N,row-major. A device packing stage maps
to D9 `[N+1,B,1,H,W]`, without host expansion or transpose.

`reprojection_plan` and `temporal_reprojection_plan` in `sintergpu.reprojection`
return setup-only JSON-compatible resource/stage/output dictionaries. Resources
declare scalar32 element capacities, dtype and role; each stage gives its exact
operation, parameters, thread count, named resource bindings and storage barriers.
`raster=True` exposes exactly the four raster outputs; intermediate projection
arrays remain available as scratch. `reference=True` requires source-depth buffers
regardless of the optional alignment flag, and exposes exactly seven reference
outputs. Zero batches skip all dispatches. Positive dimensions, every camera,
vector, partial, packing and raster scalar-offset product and device X group
limit are checked before recording.

The caller validates finite camera matrices/positive focal lengths, content and
storage types, buffer alias hazards, descriptor limits and calibration meaning.
The setup graph never creates a device, allocation, submission or synchronization.
Allocate/prepare bindings outside the record loop; use the existing
`SlangPyAdapter` with the caller's device/encoder and record declared barriers.
The same `.slang` source is installed through Python and CMake for native users;
these profiles do not establish Rust/Daxa/native adapter qualification.

## Evidence and performance

The independent references and frozen differential tests are
`tests/test_vkl_reprojection_reference.py` and
`tests/test_vkl_temporal_reprojection_reference.py`. Setup contracts, opt-in full
GPU outputs/padding/scratch reuse and hot-path guards are separate tests. Both
pinned compilers pass all146 catalog entries and matching reflection layouts.
Initial GPU attempts exposed canceled normalization rounding on both devices and
suppressed nonfinite diagnostic products on Radeon. The source corrections now
have actual/mutated compiler-emission guards; fixed-source GPU verification is
still pending. Original failure logs and partial-buffer diagnostics are retained.

The donor reference runtime is part of the numerical identity. The two hosts'
Torch2.14.1 CPU builds produced slightly different projections for identical
inputs; bilinear sampling amplified those differences beyond the unchanged RGB
comparison threshold. CPU-only probes reproduce this without GPU execution.
The differing live CPU-wheel behavior remains failed interoperability evidence;
it is not reported as a Slang or adapter qualification.

For reproducible differential qualification, all63 cases and113 invocation states
use one portable frozen-donor archive exported by
`tools/export_reprojection_fixtures.py` in the pinned Torch2.14.1+rocm7.2 CPU
environment with one intraop/interop thread. Export calls unchanged frozen donor
functions and checks every output against the independent NumPy oracle. The
manifest binds exact typed input/argument/graph bytes, complete outputs, donor
and generator source closure, actual installed Python/Torch binaries and
distribution records, build configuration and CPU ISA. This identifies installed
artifacts; it does not invent an upstream wheel checksum.

GPU tests validate all archive entries before creating a device, then compare
every output against both independent NumPy and the identical canonical donor
outputs on each vendor. Float thresholds remain rtol8e-6/atol8e-7 with equal NaN
locations; logical/count comparisons remain exact. Missing, stale or corrupt
fixtures fail closed; GPU qualification never regenerates them or skips a missing
Torch installation. Qualification against this canonical profile does not imply
compatibility with every live Torch build or application trajectory.

`references/reprojection_stage_numeric.py` supplies separate CPU computations for
the nine new dispatch boundaries. Each consumes prepared inputs and mutates only
caller-preallocated written/inout arrays. Projection starts from camera XYZ;
sampling starts from projected maps; reductions start from partials. CPU prefix
evaluation and state resets belong outside each stage clock. Complete public
function CPU timings remain a separate scope.

New source changes the conservative global timing identity. Earlier samples are
retained at their original source/compiler/device context and are historical until
fresh collection; no raw samples are rekeyed. GPU timestamps for one dispatch,
resident complete pipelines and transfer-inclusive wall time are separate costs.
Stateful per-stage timing must replay/reset prerequisites outside each measured
GPU interval. CPU whole-operation time is never copied onto a stage. There is no
optimized default selection, speedup, full-domain, tracking or trajectory claim.
