# Current full-pipeline performance

The default GPU pipeline records model raycast, source/reference pyramids, 17 ICP
iterations, GPU world-pose composition, hash activation and indirect TSDF fusion in
one command submission. No intermediate dense readback or CPU solve/fusion occurs.
Two image uploads total 16 bytes/pixel; final diagnostics total 96 bytes/frame.
The CPU reference uses NumPy float32 geometry/map storage and float64 normal equations.

`slang-rgbd-slam compare` measures both implementations on identical decoded arrays.
Wall time includes frame transfers, raycast, tracking, mapping and final diagnostics;
construction/compilation, decoding, quality evaluation and export are excluded.
First-frame and subsequent-frame distributions are reported separately. Track loss,
capacity errors, common-frame ATE, reconstruction coverage and depth agreement are
reported beside speed. Independent static guards and numerical tests precede timing.

## Historical tracking-only benchmark

The measurements below describe the previous tracker and CPU point-map prototype;
they exclude the TSDF mapping/raycast now present in the default pipeline.

## Original measurement scope

Independent review completed before numerical GPU validation. The recording
path retains compiled kernels and GPU buffers; Python loops cover three
pyramid levels, fixed ICP iterations and logarithmic reduction stages. No
per-pixel Python loop, GPU allocation, compilation, CPU readback, intermediate
submission or device wait occurs in `VulkanTracker._record_track`, its delegated
`_record_maps`, or the external-encoder path through `GaussNewton.dispatch`.
Small Python dictionaries bind constants and existing buffers per dispatch.

`track()` uploads depth and two 4x4 poses, resets an 8-byte status buffer, then
submits one frame. Its final host transfer is 76 bytes: pose (64), count/status
(8), and weighted residual sum (4). These frame-boundary transfers are excluded
from the resident GPU benchmark. The depth finite/nonnegative validation is
vectorized on the host. TUM image decoding is lazy, one frame at a time; index
metadata and trajectory/report records are retained on the CPU.

Voxel insertion is a CPU keyframe operation using vectorized projection,
`np.unique` and `np.add.at`. Retained voxels are capped, and temporary arrays
are bounded by that cap plus one image's pixels. CPU mapping and PLY export
are excluded from GPU tracking timings. Export uses a host loop outside the
tracking path. This prototype is not a GPU mapping implementation.

## AMD Vulkan measurements, 2026-10-05

[Raw JSON](benchmark-amd-20261005.json) records five measured repetitions per
resolution, device/version metadata, raw timestamps converted with
`device.info.timestamp_frequency`, and numerical checks for every repetition.
The fixture uses the second independently ray-rendered synthetic room view
against the first view, starting from identity each time. Three validated
warmup frames precede timing. Reference maps, source depth upload, compilation,
allocation and rendering happen before measurement.

| Resolution | GPU median ms/frame | GPU min | GPU p95 | Host amortized ms/frame | Final correspondences | Maximum pose element error |
|---|---:|---:|---:|---:|---:|---:|
| 160 x 120 | 0.703016 | 0.690312 | 1.394329 | 3.477353 | 13,029 | 0.000204558 |
| 640 x 480 | 3.124056 | 3.090111 | 3.768905 | 5.675073 | 299,826 | 0.0000266432 |

Device: AMD Radeon 8060S Graphics, RADV STRIX_HALO, Vulkan. The supervisor
serialized physical GPU validation and this benchmark. These five-sample
measurements characterize this fixture, not throughput across a dataset or
sustained thermal operation.

GPU timestamps bracket source depth pyramid, vertex/normal maps, and all 17
coarse-to-fine ICP iterations: projective association, Caspar-generated
residual/Jacobian, hierarchical subgroup normal-equation reductions, damped
6x6 solve and SE(3) pose update. These timings are not comparable to the
`caspar-slang` benchmark of a single fixed-correspondence Gauss-Newton step.
Reference maps and CPU keyframe mapping are excluded.

Each repetition resets pose/status with GPU copies outside its timestamp pair.
Tiny GPU diagnostic snapshots also occur outside the pairs. All repetitions
are recorded into one encoder and submitted together. The separately labelled
host cost includes recording, reset/snapshot GPU copies, submission and wait,
amortized over five frames. It excludes diagnostic and query readback, and
input upload; it is not application frame latency.

Every measured status must succeed, correspondence count exceed the configured
minimum, weighted cost remain finite/nonnegative, and pose match independent
ray-rendering truth within 0.004 per matrix element. A failed fixture or measured
frame raises an error instead of returning a successful timing report.

`tests/test_hot_path.py` statically guards the tracker recording functions,
explicitly checks the external-encoder Caspar dependency when installed, and
rejects host loops in voxel integration. The GPU numerical tests passed before
these timings. NVIDIA and Intel Vulkan measurements remain uncollected.

Reproduce:

```bash
uv run slang-rgbd-slam benchmark --width 160 --height 120 --iterations 5
uv run slang-rgbd-slam benchmark --width 640 --height 480 --iterations 5
```
