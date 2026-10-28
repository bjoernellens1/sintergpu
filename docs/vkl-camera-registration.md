# VKL camera registration forward profile

SinterGPU provides an experimental float32 forward profile of the five numerical
functions from VKL's [`camera_registration.py`](https://github.com/bjoernellens1/vkl/blob/3384023a311651ea4260cf12ddab1567972f89f3/src/vkl/kernels/reference/camera_registration.py).
The frozen donor remains available independently. The profile consists of nine
dispatches, not five separate one-dispatch implementations:

| Public function | Dispatches |
| --- | --- |
| `camera_rectify_grid` | `rectifyGrid` |
| `camera_rectify_from_grid` | `rectifyFromGrid` |
| `camera_rectify` | `rectifyGrid`, `rectifyFromGrid` |
| `depth_camera_rays` | `depthCameraRays` |
| `depth_register` | `depthCameraRays` when rays are not cached; `clearRegistration`, `projectRegistration`, `registrationMin`, `registrationWinnerCount`, and `finalizeRegistration` or `finalizeRegistrationDebug` |

The importable `common.brown_conrady` Slang module provides the forward Brown
polynomial and five-step inverse with `k1,k2,p1,p2,k3` coefficient order. The
Locus undistort and depth-registration operations retain their own sampling,
rounding, initialization and storage contracts. The two profiles are not
interchangeable merely because they share a mathematical purpose.

Images use planar `[B,C,H,W]` float32; grids and rays are interleaved float32,
and masks use uint32 values 0 and 1. The setup-only functions in
`camera_registration.py` describe typed resources, extents, dispatches and
barriers. Calibration, cache identity and lifetime, allocation, command encoding,
submission and synchronization belong to the caller. Device-resident grids and
rays can be reused; changed calibration or dimensions require a matching new
cache. No validation readback is performed in the recording path.

Grid creation preserves raw coordinates and the donor's border tolerance.
Cached sampling sanitizes nonfinite coordinates, clamps them and uses the eager
normalized align-corners nearest or bilinear profile. Depth rays are normalized
XY, not unit XYZ. Registration includes the configured depth endpoints, requires
finite positive transformed Z above float32 epsilon and rounds pixels ties to
even. It selects minimum RGB-camera Z, then the smallest source index on exact
ties; collision counts include all valid projections, not just winners. Empty
pixels are invalid and have zero depth. Raw projection diagnostics remain visible.

Debug winner indices and collision counts are logical signed int64 values encoded
as **little-endian low/high uint32 words**, with an eight-byte stride; empty
winner `-1` is two all-one words. Consumers must interpret that layout explicitly.
The implementation uses 32-bit atomic scratch and does not require 64-bit atomics
or shaderInt64 for this encoding. Plans validate capacity and dispatch limits;
callers must provide appropriately sized and initialized buffers and barriers.

The canonical eager donor archive contains 80 states and 347 output arrays. The
full fixture lane passed 89 tests on each Radeon 8060S and RTX 4070 Ti, including
scratch reuse and diagnostic-word checks, against the pinned donor and independent
NumPy references. The original 72-state archive is retained as historical evidence:
its source closure omitted an executed package initializer and `data/VERSION` and
it was not used to qualify the GPU profile. See
[`vkl-camera-fixture-profile-0.2.6.json`](../qualification/evidence/vkl-camera-fixture-profile-0.2.6.json)
for identities and logs. These results establish fixture parity only; wider
numerical domains, gradients, live camera/application adoption and controlled
no-regression comparisons remain pending. No optimized default is selected.


The current fixture baseline contains 32 paired camera stage rows across seven workloads on each target device. CPU reference, isolated GPU dispatch, complete resident pipeline and transfer-inclusive timings are separate; the full CPU pipeline oracle uses three samples and each paired stage uses 50. See the [timing table](kernel-timings.md) and [Radeon collection receipt](../qualification/evidence/vkl-camera-collection-8060s-0.2.6.json) and [RTX collection receipt](../qualification/evidence/vkl-camera-collection-4070ti-0.2.6.json). These small shared-host fixtures establish a reference, not a controlled no-regression comparison.
