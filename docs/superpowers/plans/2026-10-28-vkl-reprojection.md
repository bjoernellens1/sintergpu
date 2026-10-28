# VKL reprojection and temporal reference implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Port the complete D4/T0 numerical forward family, including every output of reproject_rgbd, rasterize_reprojection, temporal_reproject_rgb and temporal_rgbd_reference, as reusable caller-owned GPU stages.

**Architecture:** Reuse released depthGeometry for camera XYZ/validity and temporalDepthFuse for D9 clustering. Add one shared reprojection/sampling implementation, explicitly staged raster minimum/support passes and setup-only resource graphs. Frozen Torch references remain unchanged; independent scalar NumPy references and literal fixtures define parity before GPU execution. No image scheduling, frame ownership or acceptance policy enters the library.

**Tech Stack:** Slang2026.12/2026.19, SlangPy0.43.1 Vulkan adapter, NumPy/Torch CPU references, Python package/CMake sources.

**Spec:** docs/superpowers/plans/2026-10-27-consolidation.md and user-approved expanded consolidation request. Exact numerical sources are frozen VKL3384023a311651ea4260cf12ddab1567972f89f3: references/donors/vkl/src/vkl/kernels/reference/{reproject,temporal_reproject}.py.

## Global constraints

- Borrow device/command encoder; caller owns allocation, visibility, submission and synchronization. No intermediate readback or hidden host transpose.
- Before1.0, incompatible numerical/API/layout changes increment minor. This additive experimental batch targets0.2.5; existing shaders/defaults stay unchanged.
- Typed uint32 logical buffers and bounded scalar32 counts; document conversion from frozen Torch bool/int64 ABI rather than silently binding incompatible storage.
- Qualify exact source/compiler/device/configuration on Radeon8060S and canonical RTX4070Ti; historical data and skips are not new qualification.
- Preserve all existing raw timing samples; refresh conservative whole-source baselines after new shader files. No optimized selection without controlled no-regression evidence; fusion adoption retains3runs/>=5%median/positiveconfidence/<=5%p95 gate.
- Versioned provenance, changelog/compatibility/migrations, immutable checksummed/attested release and seven downstream announcement PRs remain required.

## Review focus

1. Invalid depth with nonzero camera translation still emits donor raw diagnostic projection/z; only visibility is masked.
2. Nearest and bilinear sampling retain normalized-coordinate float32 roundtrip, ties-even, zero padding, align_corners=True and singleton-axis behavior.
3. Raster support includes every epsilon-close depth contributor; confidence is maximum, not mean or one winner.
4. T0 unmasked aligned_rgb and D9 source-major packing preserve actual semantics; target depth never enters prior-only reference depth.
5. Optional inputs, differing D4 image sizes, B/N indexing, overflow/device limits, zero batches, complete stores and poison tails are tested independently.

## Task1: Independent references and donor differential fixtures

Files: create src/sintergpu/references/reprojection_numeric.py, tests/test_vkl_reprojection_reference.py; create src/sintergpu/references/temporal_reprojection_numeric.py, tests/test_vkl_temporal_reprojection_reference.py.

Interfaces: NumPy functions use frozen public signatures, replacing Torch tensors with contiguous float32 ndarrays and boolean masks; return the identical output key sets/shapes/numerical meanings. Alias depth_reproject points to reproject_rgbd and is not counted as a fifth algorithm. Temporal oracle consumes the new D4 oracle and existing independent NumPy temporal_depth_fuse; no Torch implementation imported by the oracle.

- [x] Write literal tests first: invalid-depth translated projection, singleton sampler behavior, nearest half ties, collision minimum/maximum-confidence/all-support, visibility/threshold boundaries. Run and observe missing-reference RED.
- [x] Implement scalar NumPy projection, raster and zero-padded nearest/bilinear profiles with independently written arithmetic and explicit float32 rounding.
- [x] Differential tests compare every output against frozen donor modules, including asymmetric cameras, differing D4 sizes, B1/B8, N1/multiple, invalid depth, source confidence/depth options, threshold ties, cluster conflicts and zero-confidence cases. Pin masks/counts exactly, floats with justified tolerances; do not change frozen code or blanket-relax tolerances.
- [x] Test temporal-reference prior-only invariant, unmasked aligned_rgb, RGB weighting outside selected depth cluster and unmasked support/disagreement even when final valid=false. Run full CPU suite; report all skips/failures.

## Task2: Reusable GPU stages and setup graph

Files: create src/sintergpu/kernels/vkl/reprojection.slang; create src/sintergpu/reprojection.py; create tests/test_vkl_reprojection_layouts.py and tests/test_vkl_reprojection_gpu.py plus benchmarks/reprojection_support.py test/benchmark helper. Register metadata through one later integration owner.

Interfaces: reprojection_plan(batches,height,width,target_height,target_width,*,max_workgroups_x,target_depth=False,raster=False,consistency_threshold=.05) and temporal_reprojection_plan(batches,neighbors,height,width,*,max_workgroups_x,source_depth=False,source_confidence=False,reference=False,consistency_threshold=.05,min_support=1) return {resources,stages,outputs}. Each resource has elements (scalar32), dtype(float32/uint32/int32), role(input/scratch/output); each stage has operation, entrypoint, thread_count, params, bindings (shader names -> resource names); outputs map frozen output names to resource names. All dictionaries are setup-time values, no GPU imports. A graph explicitly describes producer/consumer storage barriers and scratch reset stages; caller supplies execution.

Checked implementation items describe source, CPU tests and compiler evidence only; they do not establish GPU numerical parity or device qualification. Latest full CPU run: 579 passed, 294 skipped, 151 subtests passed in 9.94s (build/reprojection-task2-cpu.log).

- [x] Write layout tests before code: literal stage/resource/output contract, invalid flags/thresholds, all scalar-product/camera/partial/raster/fusion capacity bounds, device limits and zero batches. RED must be real missing implementation, then GREEN.
- [x] Reuse existing geometry stage; project source camera XYZ unconditionally through source c2w then target transpose rotation. Support pair index repeat maps for shared target geometry/cameras in T0 without host expansion. Use per-batch/group partial count+parallax and deterministic reduction, not per-pixel Python loops or full-image serial reductions.
- [x] Implement clear -> atomic positive-float uint-min -> epsilon-close support/uint-float-max -> finalize raster stages. Clear every active scratch element per invocation. No float-atomic requirement; no framebuffer-size-per-output quadratic loop.
- [x] Implement exact-profile normalized-coordinate nearest/bilinear sampler and T0 RGB/optional depth/confidence outputs. Preserve aligned RGB outside validity. Compose prior raster output into explicit GPU D9 source-major slots with zero/invalid target slot and no implicit host transpose.
- [x] Reuse temporalDepthFuse unchanged. Final RGB uses all donor weights independent of winning cluster; zero final RGB/depth/conf when invalid; retain unmasked support counts and disagreement. Every frozen output has a resource mapping.
- [x] Write opt-in Vulkan tests using one module-scoped Device and one borrowed encoder per pipeline, with complete-output independent-reference/frozen-donor comparisons, poison tails and changed-input scratch reuse. This checks test implementation only; the 63 new GPU cases remain unexecuted.
- [x] Compile all 146 registered entries under pinned Slang2026.12 and Slang2026.19 and compare reflection across both versions. Owner-verified artifacts: build/reprojection-all-2026.12.log, build/reprojection-all-2026.19.log and build/reprojection-all-reflection.json.
- [ ] Execute the opt-in Vulkan tests on the required target devices; verify numerical parity, all output tails and repeated scratch without stale state. GPU execution and qualification remain pending.

## Task3: Integration, hot-path review, qualification and release

Files: metadata catalogs/modules/provenance/candidate inventory, qualification/0.2.5.json, docs/vkl-reprojection.md and related inventory/overlap/status/migration docs; changes/0.2.5-vkl-reprojection.feature.md, CHANGELOG.md, VERSION and packaged VERSION; benchmark recipe and generated timing data/evidence.

- [ ] Register each real dispatch and source hash (not aliases), preserve donor author/license/revision/path and profile contracts. Update dispositions without reclassifying consumer policy as missing shader work. Include postrelease delivery0.2.4 evidence.
- [ ] Separate worker performance/refinement pass before GPU execution: check sync/allocation/transfer behavior, source/AST guards and actual hot-path microbenchmark recipe. No default selection or speedup inferred from new port.
- [ ] Coordinate GPU owners, run CPU + two compiler/reflection/install lanes, differential GPU tests on8060S/4070Ti. Diagnose root causes and preserve strict invalid/count/ordering/capacity and existing tolerances.
- [ ] Collect per-entry CPU-reference/GPU timing plus resident complete pipeline and transfer-inclusive cost, raw samples/source/workload/compiler/hardware metadata. Refresh all six existing baseline recipes for changed conservative tree identity; retain historical rows unchanged except classification.
- [ ] Full branch review once; one fix pass for Important/Critical with RED->GREEN and green suite, ledger minor/declined judgments. Exact-head hosted CI, verified merge/tag and immutable release with automated notes, independent fresh download verification and seven idempotent update PRs. Broader consolidation remains active until original gates pass.
