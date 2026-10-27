# Portable GPU development base

Seventeen shared Slang kernels implement the first seven priorities: YUV conversion,
nearest resize, Brown–Conrady undistortion, invalid-depth filtering, 3×3 bilateral
cleanup, nearest-depth registration/resolve, quality maps/reduction, RGB-D
backprojection, rigid point transforms, bounded voxel insertion, coverage heatmap
and point preview, crop/normalization, cosine feature similarity, and inclusive
threshold / 3×3 dilation / erosion. Normals, surfels, novelty/next-view planning,
flow, learned inference and codec changes remain roadmap work.

`toolchain.json` pins Slang 2026.19 and upstream archive SHA256 checksums. Slang
Metal generation is experimental. `build-shaders.py` compiles **every** manifest
kernel to SPIR-V 1.3 and MSL, validates SPIR-V for Vulkan 1.1, and verifies both
reflected layouts. On macOS it additionally compiles host and iPhone metallibs.
An absent Apple compiler is recorded pending on Linux and fails `--metal-required`.
Build/reflection success is separate from numerical execution and camera ownership.

## Scope and integration status

* `VulkanBackend` submits bounded asynchronous compute using already-owned Vulkan
  buffers. It validates all kernel extents/parameters before encoding and owns the checked128-byte parameter buffer. It retains borrowed buffers through a fixed callback until its fence
  and every declared consumer complete. The caller serializes API calls and owns state initialization and epoch budgets before passing buffers. Pipeline
  creation is setup work. The destructor drains its own fences; dispatch never
  waits for a device, maps camera pixels, or reads images back.
* Android `AndroidImageImporter` imports AHardwareBuffer memory into VkImage,
  creates a YCbCr-aware sampler/view for external formats, waits an imported
  producer syncFD, transfers FOREIGN queue ownership, and exports a release syncFD
  after the final consumer and release barrier complete. Missing device features
  return unsupported. This is an optional native foundation, not a Camera2 or
  MediaCodec recording path. Import callers must supply consumer completion only
  **after the corresponding GPU work has completed**, and must pass the release
  syncFD to the native owner. Only compute-queue consumers are currently supported.
* iOS `MetalBackend.importFrame` retains a CVPixelBuffer and CVMetalTexture wrappers
  for BGRA or NV12 planes. Its bounded asynchronous buffer dispatch retains the
  five buffers, backend parameter metadata and optional owner through command completion and all consumers.
  Neither the coordinator nor encoder selects it yet.
* **Missing source integration:** GPU texture-to-float4-buffer conversion, native
  camera dispatch orchestration, preview/encoder consumers, persistent coverage
  management, and ML tensor handoff are not wired into either recording shell.
  Android preserves existing GLES/OES + MediaCodec. iOS preserves VideoToolbox.
  No live Slang camera path, universal zero-copy, ANE interop, or phone throughput
  is claimed. Native tracking continues to own ARCore/ARKit input.
* **Device gates:** Android external-format/fence imports, iOS pixel-buffer/encoder
  compatibility, numerical fixtures on phones, start/stop pressure, bandwidth,
  readbacks, latency and thermal stability require named physical devices.
  Lavapipe fixtures and hosted Metal fixtures only verify development compute.

## ABI 1

All kernels use bindings 0/1/2 for input `float4` structured buffers, 3 for output
`float4`, 4 for mutable `uint` state, and 5 for 128-byte parameters (`float4 p[8]`).
Strides are **float4 elements**, not bytes; each float4 is 16 bytes. Native NV12
textures are not these buffers. YUV buffer components carry unsigned byte values
as floats; chroma uses nearest cosited 4:2:0 sampling. Centered chroma resampling
is unavailable in this increment and must not be silently relabelled.

`p0=(width,height,inputStride,outputStride)` describes logical image extents;
point kernels require height=1. Padding is preserved, never interpreted as image
pixels. The host validator in `tests/gpu/reference.py` checks all fixture extents,
finite parameters, calibration, scales and accumulator limits before execution.
The native and Metal production dispatch adapters run equivalent extent/type/calibration checks and bind their own validated128-byte parameters. Camera integration must additionally associate production calibration identity and initialize/reset persistent state.

| Kernel | Additional parameters / state |
| --- | --- |
| yuv_to_rgb | p1.x chroma stride; p1.y limited(1)/full(0); p1.z BT709(1)/BT601(0); p1.w cosited(0), centered chroma unsupported |
| resize | p1.xyz input width/height/stride; nearest pixel centers |
| undistort | p1 fx/fy/cx/cy; p2.xyz k1/k2/k3; p3.xy tangential p1/p2; p7.x calibrated=1 |
| invalid_depth | p1.x metres/unit, p1.y/z inclusive depth min/max |
| bilateral | p1.x spatial sigma in pixels, p1.y range sigma in metres |
| depth_register | p0 depth extents; p1 color width/height/stride/metres-per-unit; p2 depth intrinsics, p3 color intrinsics; p4–6 depth-to-color affine rows; p7.x calibrated=1; state initialized to positive infinity float bits |
| depth_resolve | state positive depth float bits; writes depth and validity to output x/y |
| metrics | p1.x/y clipping low/high; p1.z depth stride; output gradient, absolute Laplacian, clipping, valid depth |
| reduce | state four zero-initialized counters; gradient/Laplacian clamped 0–4, quantized at256; count clipped/valid depth; max2^20 pixels per reduction, reset counters before next frame |
| backproject | p1 intrinsics; p2.x metres/unit; p7.x calibrated=1; optical coordinates right/down/forward; output w validity |
| transform_points | p4–6 affine rows; input/output w validity |
| voxel_insert | p1.x voxel metres, p1.y capacity, p1.z probe bound≤64; state 2×capacity key/count entries plus accepted/overflow/collision counters |
| coverage_heatmap | p1.x positive count scale; state key/count table; color encodes observed count, not scene completeness |
| coverage_render | p1.x voxel metres; occupied slots produce voxel centers + validity |
| normalize | p1 crop x/y/width/height; p2.xy source dimensions; p3.xyz mean; p4.xyz positive stddev; invalid sample has alpha0 |
| similarity | p1.x feature count0–256; p1.y feature pitch; input pixel stride p0.z; each scalar occupies float4.x; query input1; cosine similarity; zero/nonfinite vectors return0 |
| mask_postprocess | p1.x inclusive threshold; p1.y0 threshold,1 dilation,2 erosion; outside pixels false |

Registration uses atomic minimum on positive depth bits, so competing projections
retain the nearest positive depth. Missing calibration is rejected by the fixture
host contract; shaders also gate calibration validity. Raw sensor evidence is
never overwritten. Geometry carries frame/calibration/world epoch in shared
metadata; kernels do not infer temporal association or synchronization.

Voxel coordinates are floor(point/size), restricted to [-512,511] on each axis,
packed into a nonzero 30-bit key. Linear probing is bounded; occupancy collision,
capacity/probe exhaustion and out-of-range points are counted. NaN/invalid points
are skipped. Tables/counters must be reset on a new world epoch. The host must
bound cumulative accepted/collision totals below uint32 overflow, and cap per-slot
counts; this base does not silently wrap epochs or provide a persistent scene
manager. `shared/core/src/coverage.rs` is the checked CPU policy/reference. The
rendered table only describes observed coverage.

## Development verification

```sh
python3 tools/install-shader-toolchain.py --out /path/to/slang
python3 tools/build-shaders.py --slang /path/to/slang/bin/slangc
python3 -m unittest discover -s tests/gpu -p 'test_*.py' -v
cargo test --workspace
python3 tools/check-gpu-hot-path.py
cmake -S native -B build/gpu-native -G Ninja -DLOCUS_GPU=ON \
  -DLOCUS_JNI=OFF -DLOCUS_HOST_TESTS=OFF -DLOCUS_SHARED_CORE=OFF -DLOCUS_ROS2=OFF
cmake --build build/gpu-native --target locus_gpu_fixture
# Run only after the independent performance-refinement gate.
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  python3 tests/gpu/run-fixtures.py --backend vulkan --runner build/gpu-native/locus_gpu_fixture
# macOS: --metal-required compiles both host and iPhone metallibs first.
swiftc gpu/tests/metal_fixture.swift -o build/metal_fixture -framework Metal
python3 tests/gpu/run-fixtures.py --backend metal --runner build/metal_fixture
```

The test runners intentionally upload/read back tiny fixtures and record copied
bytes plus submit-to-completion latency. Those copies and waits are test-only;
fixture latency includes driver/submit cost and is not phone kernel throughput.
The macOS runner exits77 if no Metal device exists; CI records explicit pending
numerical evidence rather than claiming a runtime pass. Shader artifacts,
reflection, independent scalar vectors, compiler-failure/layout mutation tests,
resource-pool fence/consumer tests and performance guards are separate gates.
