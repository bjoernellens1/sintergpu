# Camera timing collection: prerequisites for RTX 4070 Ti

Status: design only. No GPU execution is authorized by this document. The
existing `benchmarks/vkl_camera_registration.py` recipe does not expose enough
controls for a wrapper to meet the requested validation boundary.

`run()` constructs its own Vulkan device and dispatches internally. It offers
no callback to establish the device's actual selected physical UUID and driver
before the first dispatch. Separately, `measure_graph()` calls
`benchmarks.vkl_geometry_depth.measure`: resident dispatches are batched, and
neither each resident nor each transfer-inclusive sample is followed by a full
input, output and tail validation before another sample runs. Stage samples do
have per-sample validation. A wrapper must not claim the stronger whole-pipeline
contract or monkeypatch the measurement function to simulate it.

## Required implementation and review before execution

1. Add a documented borrowed-device or device-validation hook to the recipe.
   Validate Vulkan API, exact RTX 4070 Ti physical UUID
   `96d2dba0-95fd-5c00-e849-2e0610f24e06`, current driver and the configured
   NVIDIA ICD against fresh device evidence before any workload dispatch. An
   adapter name or an ICD environment variable alone does not identify the
   actual selected physical device. Fail closed if the runtime cannot expose a
   trustworthy physical UUID; use a tested native Vulkan identity bridge if
   necessary.
2. Provide a checked measurement API which runs each warmup and each of 50
   measured samples independently. Before the next replay/reset, compare the
   full numerical outputs with the independent CPU oracle and pinned donor,
   check read-only input bytes and all resource poison tails. Account for
   allocation, reset, synchronization and validation outside the measured
   intervals. Check all stage state including atomics against the captured GPU
   prefix as the existing isolated-stage lane does.
3. Preserve separate raw samples for GPU timestamped dispatch, complete resident
   graph and upload/submit/wait/readback-inclusive wall time. Capture actual
   timestamp frequency/resolution and report quantization or zero-length samples
   as limitations; do not silently clamp or fabricate positive values. Match
   outputs, work and transfer boundaries across comparisons.
4. Before the job, verify the source commit and current v1 whole-shader and
   whole-reference tree digests, plus the exact harness, oracle, donor archive
   manifest, native libraries and actual loaded compiler/runtime identity.
   Recompute them after device close and fail closed on changes. Do not promote
   historical rows or claim an unvalidated v2 dependency closure.
5. Acquire the existing nonblocking canonical `/tmp/sintergpu-gpu.lock` on the
   selected host before device creation and retain it through device close.
   Check device availability and actual ICD; save pre/post device snapshots and
   record host load. Use exclusive creation for all raw outputs and receipts,
   without overwrite. Abort on failure; preserve partial evidence and errors.
6. Independently review the new control flow, source identity coverage, per-sample
   checks and device binding before any hardware collection. The result is a
   configuration-specific fixture baseline only, not an adoption or speedup
   decision.
