# VKL geometry and depth fusion forward profiles

These four experimental Slang entry points port the complete **forward numerical
outputs** of `geometry.depth_geometry`, `geometry.world_transform`,
`depth_fuse.fuse_depth` and `temporal_depth_fuse.temporal_depth_fuse` from VKL
`3384023a311651ea4260cf12ddab1567972f89f3`. They do not port Torch autograd,
application scheduling, metric-status acceptance policy or transaction ownership.
Frozen executable Torch references and their gradients remain available unchanged.

## Shared resource contract

All floating storage is float32. Logical buffers are uint32 (zero/nonzero input,
canonical zero/one output), not Torch's one-byte bool allocations. Source IDs and
counts use signed int32 buffers, preserving numeric values and `-1` sentinels;
callers explicitly convert to/from Torch int64 if needed. Inputs and outputs must
not overlap in ways that create write/read or write/write races.

The caller owns existing device/encoder, allocation, initialization, submission
and storage visibility. No product code compiles, allocates, submits, waits,
repackages tensors or reads GPU data while recording. SlangPy's encoder tracks
storage dependencies; native Vulkan/Daxa callers establish their own hazards.

`sintergpu.vkl.geometry_layout` and `fusion_layout` return setup-time parameters,
logical dispatch extents and scalar32 buffer capacities. Supply the actual existing
device's X workgroup limit. The planners reject uint32 offset-product overflow,
signed source/count overflow and excess dispatch groups. `validate_capacities`
checks supplied capacities without reading GPU contents. Adapters/callers must
also validate element types, descriptor range limits, alignment, content contracts
and aliasing; capacity metadata alone cannot establish those properties. Zero-batch
plans record no work. Positive image dimensions are required. Workgroups are 256x1x1.

## Geometry

Catalog IDs `vkl.geometry.depthGeometry` and `vkl.geometry.worldTransform` use
`kernels/vkl/geometry.slang`. `GeometryParams` consists of uint32 `batches`,
`height`, `width`, `useWorld`, `useConfidence`. Images use **planar B3HW** XYZ/ray/
normal storage, scalar B1HW depth/confidence/masks, row-major Bx3x3 intrinsics and
row-major Bx4x4 camera-to-world matrices.

`depthGeometry` reads `depth`, `intrinsics`, `pose`, `confidence`. Outputs are
`validOut`, `rays_cameraOut`, `xyz_cameraOut`, `normalsOut`,
`geometry_confidenceOut`, and optional `xyz_worldOut`. Logical threads are BHW.
Intrinsics must be finite with positive focal lengths; matrices must match the
caller-validated camera convention. Invalid/nonpositive/nonfinite depth is masked
to zero before backprojection. Rays remain the calibrated pinhole rays even at
invalid pixels. Normals use the right/down forward stencil, inclusive 25% relative
depth-jump rejection, length clamp 1e-8 and orientation toward the camera. Boundary
and rejected-stencil normals are zero. World XYZ is remasked after translation.

`useWorld=0` leaves world output storage untouched: bind a caller-owned one-element
dummy buffer rather than allocating/writing a full extra XYZ image. Disabled
optional inputs likewise accept existing one-element dummies. `useConfidence=0`
returns the stencil's float32 zero/one confidence. When enabled, confidence follows
the donor multiplication exactly, including signed zero and NaN from `0*NaN` or
`0*Inf`; unlike depth-fusion confidence, the donor geometry function imposes no
finite-[0,1] value validation. The shader preserves these cases explicitly because
a bool-to-float multiplication can otherwise be optimized into a zeroing select.

`worldTransform` reads only `xyz` and `pose` and writes `xyz_worldOut`, with the
same planar layouts and BHW dispatch. It always computes Rp+t, including zero or
nonfinite XYZ. It does **not** inherit depthGeometry's validity remasking. This
is a deliberate distinct contract, documented by the overlap audit.

## Same-grid depth fusion

`vkl.depth_fusion.fuseDepth` uses source-major `[S,B,1,H,W]` arrays `depths`,
`valids`, `confidences`. Outputs `depthOut`, `validOut`, `confidenceOut` and
`winning_sourceOut` have B1HW elements. Logical threads are BHW. `FusionParams`:
uint32 `count` (BHW), `sources`, `relative`, `minSupport`, `minTemporalSupport`;
float32 `threshold`. The support parameters are unused by this entry point.

Sources must be nonempty; confidences must be finite [0,1]. A usable observation
has an input valid mask, finite positive depth and positive confidence. Compute
its confidence-weighted mean, then require every usable source to lie within the
inclusive relative `threshold*max(mean,float32_epsilon)` interval. Consistent
sources are averaged, with summed confidence clamped to one. On conflict, choose
the first highest-confidence source; relative sources score half confidence.
The winner output always records this first maximum, including consistent pixels.
Invalid pixels store zero depth/confidence/valid and source `-1`.

`relative=1` represents all-RELATIVE input fields. `relative=0` represents fields
already accepted as metric by their consumer (estimated and/or anchored). The
consumer must reject mixed relative/metric inputs unless explicitly aligned and
retain the original metric-status result. The shader does not promote a field's
metric status or own its policy.

## Temporal depth fusion

`vkl.depth_fusion.temporalDepthFuse` reads the same source-major arrays plus
uint32 `visibility`, `consistency`, `trusted`. **Source zero is the target**;
remaining sources are already-reprojected temporal observations. Target visibility
and consistency are ignored, as in the donor. Missing optional evidence is supplied
by explicitly bound caller-owned ones/zeros, not host-generated copies during
recording. S>=2, positive finite threshold, minSupport>=1, minTemporalSupport>=0.

Usability adds temporal visibility/consistency. If any usable observation is
trusted, select the first maximum trust score (`1+confidence` trusted, half
confidence untrusted). Otherwise select the first highest-weight pairwise
agreement cluster. Pairwise thresholds are **asymmetric**, scaled by each candidate
reference depth. Agreement with the winning reference determines weighted depth,
variance, dominant contributor and support. Trusted support bypasses minimum
support requirements. Otherwise both total and temporal support minima apply.

All eleven donor numerical outputs are produced: `depthOut`, `validOut`,
`confidenceOut`, `support_countOut`, `temporal_support_countOut`,
`depth_varianceOut`, `winning_sourceOut`, `conflict_owner_sourceOut`,
`dominant_contributorOut`, `temporal_weight_fractionOut`,
`source_disagreementOut`. Support counts are retained even when output validity
fails. Invalid numerical outputs are zero; all three source-ID outputs are `-1`.
Winning source and dominant contributor share values but have explicit bindings.

The kernel streams scalar state without pairwise scratch, sorting or atomics.
Same-grid fusion is O(P*S); temporal consensus is O(P*S²) in the no-trust branch.
S is caller-controlled within offset/count limits, with no arbitrary compile-time
cap. Large source sets need measured budgets; no optimized variant is selected.

## Numerical and verification scope

Tests compare every numerical output against the frozen donor and an independently
implemented NumPy oracle, with exact logical/provenance values and untouched output
padding. Standalone and borrowed-encoder fused-depth→geometry pipeline cases are
included. Float comparison limits are rtol=8e-6, atol=5e-7 for tested GPU cases;
constant depths 0.5, 1, 2, 4 are additionally exact fixed-point regressions.

GPU division can approximate even equal numerator/denominator ratios. A raw
implementation created false normals after fusion. `stableRatio` preserves normal
exact power-of-two quotients by checking equal significands and constructing their
exponent difference, without requiring float64. Other ratios retain the target's
float32 division accuracy; this is not general bitwise parity or correctly-rounded
division certification. Extreme/denormal behavior, backward gradients, other
compiler/device configurations and full application results remain unqualified.
No tolerance was loosened to repair the new pipeline or constant-depth regressions.

Run CPU contracts and explicit GPU checks:

```sh
PYTHONPATH=src python -m pytest tests/test_vkl_geometry_depth.py tests/test_vkl_layouts.py tests/test_vkl_hot_path.py -q
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python -m pytest tests/test_vkl_geometry_depth.py -q
```

Independent performance review preceded GPU execution. Static mutation guards
reject image-count shader loops, extra source passes, missing tail/provenance
stores, atomics, workgroup scratch and barriers. The benchmark preallocates,
warms up and records GPU timestamps, host record/submit/wait, and separate full
upload/readback samples. It covers standalone operations and the GPU-resident
fusion→geometry pipeline, trust patterns, conflicts, invalid fields and source-
count/image-size scaling. Shared-device results are exploratory measurements;
there is no controlled-baseline speedup, optimization adoption or vendor claim.

```sh
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/vkl_geometry_depth.py \
  --height 480 --width 640 --sources 4 --trust mixed --iterations 30 \
  --output geometry-depth-results.json
```

## 0.2.0 camera-ray rounding migration

New seeded camera workloads exposed native GPU reciprocal approximation in camera-ray division: one-ULP ray differences amplified into normals outside the existing tolerance. Camera rays now use a resource-independent scalar32 integer division helper, rounded to IEEE binary32 nearest/even, including subnormals and signed zero. It introduces no shaderInt64 capability. Point and normal output bits can change; re-run downstream geometry/residual/pose parity before adopting 0.2.0. The original geometry masks, stencil, layouts and floating tolerances remain unchanged. Frozen Torch and independent NumPy references are preserved. Native arithmetic elsewhere retains its separately documented qualification limits.

Point products also carry explicit SPIR-V `NoContraction`, preserving the multiplication boundary before tangent subtraction even under pinned Slang 2026.12. Fused-depth ratios reuse the same rounded division helper; weighted depth products retain their multiplication rounding before accumulation. This is required by the borrowed fusion→geometry pipeline, where a one-ULP predecessor depth difference can exceed the normal tolerance. The composed pipeline must pass independently of isolated per-entry tolerances.
