# Standalone xsplat math contracts

These experimental operations retain the pinned xsplat-vk source behavior. They
are reusable math/dispatch components; the library does not own an optimizer,
renderer, frame scheduler or camera policy. The original Apache-2.0 sources,
notices and exact paths/hashes remain in `provenance/xsplat-vk.json`.

## Adam

`xsplat_vk.adam.main` reads float32 gradients and updates float32 values, first
moments and second moments in place for `count` elements. All four buffers must
have that capacity. The entry-point uniform `p` contains count/step (uint32) and
learningRate/beta1/beta2/epsilon (float32). Bias correction uses `max(step,1)`.
The standalone default is32 threads per group. Caller policy must provide finite,
valid optimizer parameters and nonnegative corrected variance; convergence and
invalid hyperparameter handling are not qualified.

SlangPy binds these resource buffers through `vars` and entry-point `p` as a
keyword argument to dispatch. Repeated optimizer iterations intentionally update
state; isolated benchmarks reset all three inout buffers between samples.

## Gaussian projection

`xsplat_vk.project.projectMain` consumes means, log-scales and normalized-on-use
quaternions in xyzw order. Camera data holds a world-to-camera matrix, focal and
principal point, image dimensions and Gaussian count. It transforms means to
camera space, projects by perspective division, transforms anisotropic covariance,
adds0.3 diagonal dilation and writes the inverse symmetric2D covariance as conic.
The standalone workgroup is128 threads. Opacity and SH fields remain in the donor
input layout even though this entry does not use them.

Visibility requires camera-space depth greater than float32(1e-4), and both pixel
coordinates within `[0,imageSize)`. Depth-rejected points have zero pixel/conic fields
and visibility, while depth retains the transformed camera-space value. The
frozen donor C++ reference returns all-zero records for depth-rejected points; callers
must preserve that distinction. Off-screen points with accepted depth retain
computed pixel/conic fields and have `visible=0`. Nonzero finite quaternions and finite,
well-defined covariance arithmetic are the exercised domain. Zero quaternions,
nonfinite/extreme inputs and complete renderer/camera accuracy remain unqualified.

Layouts belong to a compiler and adapter configuration. Do not interchange them:

| Record | Standalone pinned compiler layout | SlangPy0.43.1 Vulkan scalar profile |
| --- | --- | --- |
| Gaussian |80 bytes; field offsets0/16/32/48/64 |56 bytes;0/12/24/40/44 |
| Projected |48 bytes;0/16/28/32 |28 bytes;0/8/20/24 |
| Camera |96 bytes;0/64/72/80/88 |92 bytes;0/64/72/80/88 |
| Gradient diagnostic |32 bytes;0/12/16 |20 bytes;0/12/16 |

The benchmark requests row-major matrix compilation, validates actual bound
resource layouts before dispatch, and supplies camera constants through the
SlangPy field dictionary API. Pinned standalone2026.12/2026.19 compilation and ABI
checks are separate from the bundled compiler identity. Consumers must reflect
and validate their own profile, not assume one of these tables applies elsewhere.

The [timing methodology](xsplat-timing-methodology.md) and [pixel error model](xsplat-projection-error-model.md)
record the numerical evidence. Visible pixel coordinates, depth and conic have
2e-6 absolute/relative fixture checks. Every pixel coordinate additionally fits an
input-derived FP32 envelope, with ambiguous depth/viewport classification blocking
qualification. Large off-screen errors and broad enclosures are retained explicitly;
this evidence does not make projected coordinates metric ground truth.

## Compositing gradient diagnostic

`xsplat_vk.backward_probe.backwardProbeMain` launches one thread and writes three
RGB derivatives, one alpha derivative and one incoming-transmittance derivative
for the fixed donor example. The underlying compositing helpers are importable
Slang math. This dispatch is a toolchain diagnostic, not a rasterizer or scalable
full-gradient implementation. Defined outputs retain2e-6 fixture comparisons;
complete tail records are exact, and padding inside live structs is unspecified.

## References and integration

Source bundles retain the unmodified C++ reference implementation and original
headers under `references/donors/xsplat-vk`; original finite-difference and centered
projection tests execute without a GPU. Independent NumPy benchmark oracles live
in `benchmarks/xsplat_reference.py`. These are reference implementations, not an
optimized CPU runtime fallback. The Python wheel exposes shader sources/catalog;
C++ source/header consumers use the existing CMake integration. Callers retain
allocation, command submission and synchronization ownership.
