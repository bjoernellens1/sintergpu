# VKL Camera Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute the user-authorized consolidation with independent review and evidence at each gate.

**Goal:** Add reusable, experimental float32 Slang forward profiles for all five frozen VKL camera-registration functions, preserving every numerical output and retaining existing consumer contracts.

**Architecture:** Separate calibration-only grid/ray preparation from per-frame sampling and staged depth registration. A small Brown distortion helper owns shared mathematical expressions; named VKL and Locus operation profiles retain their different rounding, sampling, validity and storage contracts. Setup-only resource plans record on caller-owned Vulkan resources without allocations, submissions, synchronization or readback in the record loop.

**Tech Stack:** Python >=3.11; independent NumPy CPU references; unchanged Torch donor references; Slang 2026.12 and 2026.19; existing SlangPy 0.43.1 adapter; Vulkan typed buffers and uint32 atomics; logical int64 diagnostics use an explicitly little-endian two-word uint32 shader ABI; no shaderInt64 is required.

**Spec:** The parent request for the next camera-registration batch, the accompanying source-grounded analysis in this task, and pinned donor `references/donors/vkl/src/vkl/kernels/reference/camera_registration.py`. This plan contains the proposed numerical and integration boundaries so execution does not depend on chat recollection. Supporting audit: `docs/kernel-overlap.md:G03,G10`; provenance: `provenance/vkl-candidates.json:32-87`.

## Global Constraints

- Implementation checkout: `/home/bjoern/.codex/worktrees/camera-registration/sintergpu`, branch `impl/vkl-camera-registration`, based on released commit `c66df57d77a2929968e5635a01adea684a240733`.
- Execute source work in the attached managed worktree from released 0.2.5. Preserve dirty peer work and never revert another worker's changes.
- Proposed compatible-addition version: 0.2.6. At execution start verify the released base still has VERSION 0.2.5; if it differs, the supervisor updates version references before assigning edits. Version changes and release publication belong to the final integration owner.
- Frozen donor repository: `bjoernellens1/vkl`; revision `3384023a311651ea4260cf12ddab1567972f89f3`. Camera file SHA256: `648628355f4b86cc708f0c22d04e3cb6bd21ad85a8cd51868b37b1e1fd05ad3c`.
- Original VKL code is Apache-2.0 by owner authorization recorded in `NOTICE:4-9` and `provenance/vkl-candidates.json:3-6`. Preserve headers, independent donor files, executable AST, source hashes and third-party notices. No new OpenCV runtime dependency: its optional donor comparison is corroboration only.
- Preserve the five numerical operations: `camera_rectify_grid`, `camera_rectify_from_grid`, `camera_rectify`, `depth_camera_rays`, `depth_register`. The direct rectification wrapper composes grid creation and sampling; it does not warrant a duplicate mathematical implementation.
- Experimental forward float32 only. Default eager donor semantics are the canonical numerical profile. Wider dtypes, gradients, native runtime adapters, real camera calibration acceptance and application adoption remain separately pending.
- Preserve host-side calibration validation and consumer policies. No GPU readback to validate camera contents, choose cache lifetime, decide frame acceptance or produce host scalars during recording.
- Existing Locus, image RGB8 remap, VKL reprojection and consumer defaults keep their own named numerical contracts. Sharing a polynomial or an atomic pattern does not establish complete-operation equivalence.
- Build/prepare kernels and bindings outside the record loop. Calibration grids/rays and registration scratch remain device-resident. Transfers/readback are explicit output/debug boundaries with separately reported cost.
- No fixed scientific speedup objective is invented here. Fixture parity, compiler agreement, baseline timing, no-regression and application adoption are distinct gates; failure of a later gate does not relabel an earlier one.

## Review Focus

1. A caller supplies cached NaN/Inf coordinates with either validity value: preserve donor coordinate sanitization, output masking and raw diagnostics; do not convert unchecked nonfinite coordinates to integers. Task 1 and Task 4 pin this.
2. Negative half-pixel edges and equal-depth collisions: preserve ties-to-even and smallest source index, including collisions from losing depths. Task 1 and Task 5 pin this.
3. Calibration or dimensions change while a caller retains a cache: reject structural mismatches and expose a cache identity; the caller owns cache replacement. Task 2 and Task 8 pin this.
4. A second invocation has fewer valid samples than the first: every live scratch/output must reset or overwrite correctly, including empty targets, debug outputs and padding. Task 2, Task 5 and Task 7 pin this.
5. Eager normalized grid_sample and compiled preclamped interpolation have different rounding: qualify each against its named profile, retain observed failures and never widen thresholds merely to merge results. Task 1, Task 4 and Task 7 pin this.

## Source Evidence and Consumer Boundary

- Donor Brown coefficients are `k1,k2,p1,p2,k3`; forward polynomial is at donor lines 105-118; inverse uses exactly five `x = x + xd - projected_x` iterations at 121-133.
- Rectification returns raw interleaved `source_uv [B,Ht,Wt,2]` and `valid [B,1,Ht,Wt]`; debug adds `target_ray [B,Ht,Wt,3]` and displacement. Border tolerance is `16 * float32_epsilon * max(Hs,Ws)` (164-229).
- Cached sampling replaces nonfinite coordinates with zero and clamps coordinates before sampling. Eager uses normalized align-corners grid_sample; compiled uses the direct preclamped sampler (136-161,232-265). A supplied true validity mask and NaN coordinate therefore still samples the sanitized location.
- Depth rays are undistorted normalized XY, not unit-length XYZ (310-348). Registration uses inclusive source depth min/max, finite RGB Z > float32 epsilon, rounded-pixel bounds, exact nearest-Z ties, smallest source index, and collision count over all source-valid projections (351-516).
- `recipes/splatograph/configs/pipelines/conditioned_fastpath_v1.yaml:3-28` in the pinned local `/home/bjoern/git/vkl` checkout consumes rectified RGB and registered depth. Registration `valid` is measurement occupancy passed as `depth_measurement_valid`; it is not geometric conditioning domain. `tests/test_conditioned_fastpath.py:33-115` pins this distinction.
- That VKL executor currently invokes Torch functions (`src/vkl/pipeline/executor.py:119-140`); `SlangPyAdapter` records on caller-owned SlangPy resources (`src/sintergpu/adapters/slangpy.py:8-48`). A qualified Torch/Vulkan resource-and-completion bridge is a prerequisite to replacing a live VKL node. No per-frame NumPy round trip is accepted as a resident integration.

## File Responsibilities and Ownership

| Surface | Proposed files | Responsibility |
| --- | --- | --- |
| Independent reference | `src/sintergpu/references/camera_registration_numeric.py` | All five NumPy float32 forward functions, shared scalar helpers, complete outputs |
| Portable frozen corpus | `benchmarks/camera_registration_fixtures.py`, `benchmarks/camera_registration_fixture_archive.py`, `tools/export_camera_registration_fixtures.py` | Deterministic typed payloads, strict archive loading, explicit CPU donor export |
| Shared pure math | `src/sintergpu/kernels/common/camera_distortion.slang` | Brown forward polynomial and five-step inverse, explicit arithmetic ordering |
| VKL device stages | `src/sintergpu/kernels/vkl/camera_registration.slang` | Typed grid/ray/sample/project/clear/min/winner-count/finalize entrypoints |
| Setup plans | `src/sintergpu/camera_registration.py` | Resource capacities, dtypes, stage bindings/barriers, cache identity |
| Recording/evidence helpers | `benchmarks/camera_registration_support.py`, `benchmarks/vkl_camera_registration.py` | Prepared resources, explicit parity/readback, independent per-stage and pipeline timing |
| Tests | `tests/test_vkl_camera_registration_{reference,layouts,fixture_archive,gpu,hot_path,stage_timings}.py` | Independent contracts, graph/storage gates, fail-closed corpus, execution parity and measurement integrity |
| Integration owner only | `src/sintergpu/data/{modules,catalog,timings}.json`, `provenance/vkl-camera-registration.json`, `provenance/vkl-candidates.json`, versioned qualification, docs, VERSION files, CHANGELOG, fragment | One owner coordinates generated/aggregate files and final evidence |

Workers own only their named surface. Every brief identifies the exact worktree/branch, dependencies, tests and commit authorization. Shared aggregate files have one integration owner. Current task makes no commits; future tested checkpoints are submitted to the supervisor before integration.

## Resource and ABI Decisions

- Float inputs/outputs: contiguous float32; masks: typed uint32 with canonical 0/1 outputs. Image layout is planar `[B,C,H,W]`; grid/ray vector outputs remain donor-interleaved as specified above.
- Preserve int64 `winner_source_index` with -1 sentinel and int64 `collision_count` in public diagnostic outputs. Use uint32 temporary winners/counts and positive-float depth bits internally; finalization widens values. No 64-bit atomics are required. Mark shaderInt64 explicitly for diagnostic entrypoints and validate its presence before creating their pipelines; a device without it remains unsupported for this complete diagnostic profile.
- Source pixel count `P` must satisfy `1 <= P < UINT32_MAX`, reserving `P` as a winner sentinel. All B/C/H/W/vector products, scratch offsets, byte sizes, padded thread counts and device workgroup limits are checked with Python integers before allocation. Capacity is measured in typed elements with explicit element size, not assumed scalar32 everywhere.
- Registration scratch holds raw source UV/Z, source-valid masks, safe destination indices, target min-depth bits, winner indices and collision counts. Invalid projections never enter atomics. Test bounds in floating space before any integer conversion, including huge finite projections.
- Registration graph: `clearRegistration -> projectRegistration -> registrationMin -> registrationWinnerCount -> finalizeRegistration`. Projection diagnostics may be produced before or after clear only if the declared dependency graph remains equivalent. Min and winner/count read the same prepared projections. Exact-Z winner equality uses the stored projected bits/value; epsilon-close supporters are NOT registration winners. Every source-valid projection increments collisions regardless of its Z rank.
- Declare storage barriers for scratch reuse, projection writes before consumers, clear before atomics, min before exact winner selection, winner/count before finalization, and output reuse. Caller owns execution dependencies across invocations and explicit output-boundary synchronization. A fake encoder test verifies recorded order; full reuse tests verify behavior.
- Cache metadata is setup-only: dimensions, camera/distortion inputs or caller-supplied immutable calibration identity, float32 profile, sampling profile and shader/source identity. A cache is rebuilt when its identity changes. Grid content remains on device; metadata validation never triggers a readback.

## Task 1: Pin Independent Numerical and Frozen-Donor Contracts

**Files:** create reference module, fixture payload/archive/export files, reference and archive tests from the ownership table. Read unchanged donor camera source and donor `tests/test_camera_registration.py`.

**Interfaces:** NumPy module exports the five donor function names with the same numerical arguments, optional distortion coefficients, `sampling`, `debug`, depth interval and cached `depth_rays`; arrays are float32 and masks are bool at this CPU oracle boundary. Exporter calls unchanged donor functions on CPU and writes a fresh archive; loader never imports Torch or regenerates expected outputs. `payloads()` yields typed input snapshots, call kwargs, function/profile identifiers and complete expected-output schemas.

- [ ] Write failing literal tests first: zero distortion identity B1/B8; independent focal/center and resolution; radial-only, tangential-only, all-five Brown coefficients; exact five-step inverse result (not an optimally converged inverse); direct/cached equivalence; singleton axes; C1/C3/C4 image layouts.
- [ ] Add output-schema assertions for each function and every debug key. Registration includes raw `[B,P,2]` UV, `[B,P]` projected Z/source validity, int64 winner/count images, plus depth/valid. Rectification debug shapes differ from registration shapes and must be asserted independently.
- [ ] Pin border tolerance immediately below/equal/above each side; nearest ties at -0.5,+0.5,1.5 and both outer edges; bilinear impulse/checkerboard; sanitized cached NaN/Inf with valid true/false; raw nonfinite diagnostics; inclusive min/max depth; transformed Z below/equal/above epsilon; all-invalid input; unequal-Z, equal-Z and one-ULP collision cases with distinct source indices.
- [ ] Run `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_vkl_camera_registration_reference.py tests/test_vkl_camera_registration_fixture_archive.py -q`; retain a RED log proving missing implementation or intended contract assertions fail.
- [ ] Implement independent scalar NumPy arithmetic and typed fixture identities. Keep the five-iteration recurrence, exact donor validity predicates and output masking; do not import or copy executable donor functions into the oracle.
- [ ] Export canonical eager CPU donor outputs using one explicitly pinned Torch runtime (reuse the established `2.14.1+rocm7.2`, one intraop/interop thread profile if available). Record actual binaries/distribution/config/CPU identities and source closure. The exporter rejects absent/wrong runtime, output drift, existing destinations and source changes during export. Write to `qualification/evidence/vkl-camera-registration-frozen-0.2.6/` in the future implementation worktree.
- [ ] Add corruption tests: modified inputs, graph/kwargs/profile/source closure, missing outputs, dtype/shape mismatch, duplicate identities, stale donor hashes and altered archive bytes must fail before device creation. Keep zero-tolerance logical/index/count comparisons; begin float comparison at the existing strict `rtol=8e-6, atol=8e-7`, equal NaN locations. Characterize any necessary change with independent error evidence and supervisor review rather than silently relaxing it.
- [ ] Run the focused CPU tests GREEN; retain source-bound RED/GREEN evidence and review this numerical boundary before shader work.

## Task 2: Specify Caller-Owned Resource Plans

**Files:** create `src/sintergpu/camera_registration.py` and layout tests; reuse established graph dictionary vocabulary from `src/sintergpu/reprojection.py` without adding a device dependency.

**Interfaces:** expose `camera_rectify_grid_plan(B,Hs,Ws,Ht,Wt,*,max_workgroups_x,distortion=False,debug=False)`, `camera_rectify_from_grid_plan(B,C,Hs,Ws,Ht,Wt,*,max_workgroups_x,sampling='bilinear')`, composed `camera_rectify_plan` with the union of those flags, `depth_camera_rays_plan(B,Hd,Wd,*,max_workgroups_x,distortion=False)`, and `depth_register_plan(B,Hd,Wd,Ht,Wt,*,max_workgroups_x,depth_distortion=False,rgb_distortion=False,cached_rays=False,min_depth=1e-6,max_depth=None,debug=False)`. Every return is `{resources,stages,outputs,profile}` with typed capacities, exact params, bindings, barriers and feature requirements. Camera values are caller-owned buffers; these arguments are sizes/options, not calibration arrays.

- [ ] Write RED tests for complete output sets, raw diagnostic shapes, direct/cached graph composition, int64 output element sizes and every mandatory stage/barrier. Require positive batches/dimensions like the donor; reject bool-as-integer, malformed flags, nonfinite/invalid float32 depth intervals, invalid sampling and capacity/group overflows before allocation.
- [ ] Add a fake recorder test which rejects device creation, allocation, CPU array expansion, submission, waits, to_numpy/cpu/item calls or rebuilding bindings within record. Record two invocations through the supplied encoder and inspect exact stage order.
- [ ] Implement the setup functions with checked typed sizes and profile/cache identity. Direct rectification uses the same grid and sampling entrypoints; cached rays skip only ray precomputation, not registration correctness stages.
- [ ] Run `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_vkl_camera_registration_layouts.py -q` RED then GREEN. Check AST imports remain CPU/setup-only and do not require NumPy/Torch/SlangPy at package import.

## Task 3: Own Brown Math Without Merging Numerical Contracts

**Files:** create common Brown helper; modify the new VKL shader. A later bounded edit may route `src/sintergpu/kernels/locus/preprocess/undistort.slang` through a compatible helper entry only after its own no-regression gate; no Locus ABI or sampling changes.

**Interfaces:** pure float32 forward Brown and five-step inverse helpers take XY plus explicitly ordered coefficients. If original Locus and canonical VKL require different evaluation/FMA schedules, expose clearly named schedule variants sharing only genuinely identical expressions; do not force one schedule across both consumers.

- [ ] Add RED scalar helper tests and actual compiler-emission checks for all coefficient terms and exactly five inverse iterations. Distinct fixtures must detect swapped k3/p1/p2, altered iteration count, normalized-ray versus unit-vector confusion, or an algebraic simplification that changes required rounding.
- [ ] Implement the pure helper and the grid/ray entrypoints. Preserve debug target rays, displacement, raw source UV and donor border tolerance. Use the established explicit float32 arithmetic helpers where evidence requires retained rounding boundaries.
- [ ] Compile helper consumers under both pinned compilers; inspect actual SPIR-V rather than only source text. Keep mutation tests which prove omission/reordering-sensitive changes are detected.
- [ ] For any Locus helper migration, run existing Locus literal/random and complete-output tests against its prechange source, including nearest behavior, coefficient parameter mapping and borders. Compare supported outputs exactly or at the already established original threshold; no threshold expansion. If schedules differ, retain explicit helper variants and report why.
- [ ] Run the reference/layout/hot-path CPU suites GREEN before preparing an execution qualification request.

## Task 4: Implement Complete Rectification Sampling

**Files:** new VKL shader and `tests/test_vkl_camera_registration_{reference,hot_path}.py`; archive payloads from Task 1.

**Interfaces:** `rectifyFromGrid` consumes float32 planar image, interleaved raw source UV and typed validity; writes complete planar image. The graph exposes the original validity image. Nearest and bilinear modes are explicit profiles; direct rectification composes Task 3's grid output with this stage.

- [ ] Add RED tests that distinguish normalized float32 eager sampling from direct preclamped sampling: width-four half-tie, odd sizes, singleton axes, tiny tolerated outside-border coordinates and one-ULP perturbations. Test NaN/Inf coordinates with caller validity true to retain sanitized-location behavior; false validity must zero the final image.
- [ ] Implement canonical eager semantics: sanitize nonfinite coordinates, clamp to source bounds, retain the normalized align-corners rounding boundaries, ties-even nearest or bilinear zero-padding, then apply the supplied validity mask. Preserve NaN behavior of floating image samples and zero-weight arithmetic as established by the frozen profile.
- [ ] Characterize unchanged donor compiled `_sample_preclamped_grid` separately against the same literal inputs. Store both profile results and error counts; do not label a tolerated compiled/eager difference as byte equality or combine their profile identities.
- [ ] Run CPU contract tests and compile-only checks GREEN. Ensure cached/direct call graphs use this single sampler implementation and output padding is never written.

## Task 5: Implement Deterministic Registration and Complete Diagnostics

**Files:** new VKL shader, setup plan, reference/layout/hot-path tests.

**Interfaces:** entrypoints `clearRegistration`, `projectRegistration`, `registrationMin`, `registrationWinnerCount`, `finalizeRegistration` follow the ABI and ordering above. `depth_register` without cached rays composes `depthCameraRays` before these stages; cached rays consume the same interleaved format.

- [ ] Write RED tests where the smallest source index is not the first encountered thread, depths are exactly equal or one ULP apart, projections land at -0.5/0.5/1.5, nearest source depth differs from nearest transformed RGB Z, and many samples collide on one target pixel. Assert every debug output, not only depth.
- [ ] Add invalid-depth tests with nonzero transform translation so raw projected diagnostics remain observable while source_valid stays false. Include huge finite/nonfinite projections and ensure bounds precede integer casts. Depth ranges use >=min and <=max; transformed Z uses >epsilon.
- [ ] Implement clear/project/min/winner-count/finalize with uint32 atomics. Count all source-valid collisions, select exact minimum-Z source indices, emit -1 for empty winners, zero empty depth and false validity, and widen public diagnostics to int64. Do not reuse reprojection epsilon-close support logic as registration winner logic.
- [ ] Add reuse tests with dense -> empty -> sparse -> changed calibration invocation states. Validate poison tails for every input, scratch and output, and source buffers unchanged after recording.
- [ ] Run CPU/static/layout tests GREEN and verify that stale counts/winners or omitted clear/min/winner barriers are caught by deliberate mutations.

## Task 6: Catalog, Provenance, Compilation and Package Closure

**Files:** integration-owner catalog/modules/provenance files; `docs/vkl-camera-registration.md`, `docs/{vkl-source-inventory,kernel-overlap}.md`; tests and installed-source checks. Existing frozen donors stay unchanged.

- [ ] Add RED provenance/catalog tests proving every new entry has explicit float32/mask/int64 ABI, capacities, the little-endian two-word diagnostic representation, profile documentation and experimental status. Include the common helper, all independent references, recipe/helper and fixture generators in source closures.
- [ ] Populate `provenance/vkl-camera-registration.json` from the pinned source/hash/license evidence. Update only the five actual numerical candidate dispositions. Consumer-only contract/tracking/keyframe/scheduling functions remain reference-only by design; backend-specific Triton retain_vendor/retain_compile dispositions do not remove Slang backlog.
- [ ] Generate the catalog with `PYTHONPATH=src:. .venv/bin/python tools/build_catalog.py`; run `PYTHONPATH=src:. .venv/bin/python -m sintergpu.cli validate` and the full CPU suite. Treat opt-in skips as skips in reports.
- [ ] Compile every portable entry under both pins using `tools/compile_shaders.py --slangc .tools/slang-2026.12/bin/slangc --output build/camera-compile-2026.12` and the analogous 2026.19 command. Require all expected entries and source hashes; retain actual compiler executable hashes, output, reflection and nonzero exits.
- [ ] Run `PYTHONPATH=src:. .venv/bin/python tools/compare_reflection.py --left build/camera-compile-2026.12/results.json --right build/camera-compile-2026.19/results.json --output build/camera-reflection.json`. Require passing resource/parameter ABI agreement for every entry, including typed int64 storage.
- [ ] Build wheel/sdist and verify tracked source closure through existing release source checks. Check archive inclusion of all fixture NPZ/manifests, compiler evidence, receipts and source helper. Install wheel and sdist into isolated environments outside the checkout; verify catalog, imports and new shader paths. Install/relocate CMake and compile every new installed shader entry with both pins. This establishes package/link/source closure, not native GPU execution.

## Task 7: Independent Review and Dual-Vendor Numerical Qualification

**Files:** new GPU tests and versioned `qualification/evidence/vkl-camera-registration-*` records; integration owner updates versioned qualification.

- [ ] Before GPU execution, assign a separate performance-refinement review of the data path: no full-frame host round trips, device syncs, per-pixel Python loops in production/recording, per-record allocations or resource/binding rebuilds. Add a CPU fake-encoder recording microbenchmark and mutation guards for forbidden calls. NumPy oracle loops are reference work and remain outside production recording.
- [ ] Independently review the complete numerical/layout/source diff and resolve Critical/Important findings. Each GPU brief identifies the actual host/device/driver, source closure, compiler profile, command, archive hash and the existing canonical per-device lock. Do not introduce a second lock namespace or race another owner's collector.
- [ ] GPU tests validate the entire frozen archive before device creation; compare all five output sets and diagnostics against both independent NumPy and canonical donor outputs. Record masks/indices/counts exactly and floats with the declared unchanged tolerance/equal-NaN rules. Missing/stale/corrupt fixture/runtime evidence fails closed rather than skipping or regenerating.
- [ ] Run `SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. <qualified-python> -m pytest tests/test_vkl_camera_registration_gpu.py tests/test_vkl_camera_registration_hot_path.py -q` separately on Radeon 8060S and RTX 4070 Ti after explicit execution authorization. Include B1/B8, different source/target sizes, all distortion flags, nearest/bilinear, cached/direct, debug/nondebug, singleton/tail/nonfinite/collision cases and repeated scratch reuse.
- [ ] Retain the first failure logs/partial diagnostics and exact pre/post-fix source identities. A fix changes the qualification source; rerun the affected full output/profile set. Do not relabel another Torch build's failed interoperability as a passed canonical profile.
- [ ] Run a dedicated Locus no-regression lane on both devices if its source/helper closure changed. Existing reprojection/camera-color behavior must remain at its original acceptance boundary. Hardware results qualify only their stated source/compiler/device/profile.

## Task 8: Genuine Stage/Transaction Timings and Adoption Decision

**Files:** create benchmark support/recipe and stage-timing tests; integration owner updates timing registry, generated table and qualification.

**Interfaces:** `PreparedCameraGraph` prepares typed buffers/kernels/bindings once and records only caller-supplied-encoder barriers/dispatches. Independent per-stage CPU functions consume precomputed stage inputs and preallocated written/inout arrays. Timing rows bind both metrics to one operation/workload/source/environment identity.

- [ ] Write RED fake-clock/fake-encoder tests proving prefix evaluation, precompute, persistent resets, allocation, validation and copies are outside isolated stage clocks; validate every measured write/read-only input/tail immediately after the isolated sample batch, before replay. Stateful atomic stages replay clear and prerequisites before every sample.
- [ ] Implement `benchmarks/vkl_camera_registration.py` with explicit cached and direct graphs, calibration rebuild workloads and separate precompute amortization. Use 640x480,1280x720,1920x1080 at B1/B8, unequal RGB/depth dimensions, distortion and collision-density variants, plus small parity fixtures. Validate dimensions/capacities before allocating large workloads.
- [ ] Measure isolated GPU timestamp intervals, complete resident pipeline, host record/submit/complete and transfer-inclusive wall time separately. Capture actual Vulkan device/driver snapshots, CPU/runtime/compiler identities, raw samples, state/profile identities and whole-job source-hash receipts. Keep debug diagnostics readback outside resident intervals and report its bytes/cost.
- [ ] Collect genuine independent per-stage CPU costs; keep whole-function donor/NumPy CPU timings separate. Use at least 50 samples and 5 warmups for fixture baselines, then three controlled matched old/new/old trials for any no-regression/adoption comparison. Report warmup, allocator/cache modes, concurrent load, uncertainty and transfer costs; no rate inferred by combining unrelated percentiles.
- [ ] Compare a consumer-equivalent baseline with identical resolution, distortion, caches, outputs and transfer boundary. Cached paths must report setup cost and frame-count break-even. Performance adoption requires full-output parity and a statistically supported improvement in the chosen transaction metric; an interval compatible with no change is inconclusive. Investigate reproducible regression before integration; supervisor explicitly accepts any remaining cost trade-off. A stage-only win cannot substitute for a slower complete transaction.
- [ ] Global shader/reference dependency identities will change. Preserve prior raw rows and contexts as historical; do not rewrite old metrics or mark stale rows current. Either rerun established recipes under the new closure or leave their current coverage pending honestly. Registry and Markdown generation must agree exactly and all Q-bound hashes must match final bytes.
- [ ] Exercise the conditioned pipeline boundary with raw RGB/depth of different sizes and finite unregistered holes. Preserve `depth_measurement_valid` separately from geometric domain. A live VKL replacement additionally requires a separately reviewed resident Torch/Vulkan ownership, aliasing, lifetime and completion bridge; there is no implicit approval to build that bridge here. If absent, report standalone profile qualification and keep application adoption pending.

## Task 9: Final Integration, Release Notes and Handoff

**Files:** integration owner only: `VERSION`, `src/sintergpu/data/VERSION`, `CHANGELOG.md`, `changes/0.2.6-vkl-camera-registration.feature.md`, `qualification/0.2.6.json`, provenance/catalog/timings/docs and retained evidence.

- [ ] Set matching versions only after the supervisor confirms the base release and integration scope. Add a dated populated exact changelog section and `Compatibility: compatible` fragment explaining all five numerical profiles, explicit ABI/feature requirements and qualification limits. Breaking consumer ABI/default changes are outside this batch and require a reviewed version/classification/migration decision.
- [ ] Generate preview with `PYTHONPATH=src:. .venv/bin/python tools/release.py notes --tag v0.2.6`; verify explicit Breaking changes None, migration status, actual test/case/state/device counts and every remaining gate. Include no speedup/default-selection/application claim without the corresponding Task 8 evidence.
- [ ] Run the full CPU suite, catalog/provenance/fixture/timing checks, both pinned compiler/reflection lanes and source/installed-package checks against final source. Any source/helper change after GPU/timing collection must be assessed for stale evidence and rerun as required. Final release artifacts are rebuilt from the exact clean tagged source; provisional wheels are not release assets.
- [ ] Have a fresh independent reviewer audit the complete branch and evidence boundaries, including shared-helper effects on Locus and existing timing identities. Resolve Critical/Important findings and document any accepted limitations. Supervisor owns commits, push, PR, merge, tag and publication under existing authorization; do not perform them as a side effect of a worker task.
- [ ] Final handoff lists implemented operations/entrypoints, source/profile/ABI, CPU and compiler results, each GPU's exact qualification, resident and transfer timing scopes, package closure, remaining scientific/native/application gates and any failed/inconclusive experiment. Keep kernel backlog separate from consumer policy.

## Plan Self-Review

- Spec coverage: all five functions and optional paths map to Tasks 1-5; provenance/package closure to Task 6; independent review and both compiler/device lanes to Tasks 6-7; timing/no-regression/application boundaries to Task 8; truthful release accounting to Task 9.
- Review-focus cases each have named assertion families in their owning tasks. No stage may be accepted on a final-output-only check that can hide corrupted intermediate state.
- Shared interfaces use one typed graph vocabulary; source_uv layouts are explicitly different for rectification and registration. Native int64 public diagnostics are not reinterpreted as scalar32 storage.
- Current activity is planning only. No command in Tasks 1-9 has been executed by writing this file, and no source/GPU/commit permission is implied by its existence.


## Qualification checkpoint

The first non-debug run passed 59 canonical states on each device at `cb40418`. The subsequently reviewed two-word debug ABI at `2b88170` passed the full GPU test file on Radeon 8060S and RTX 4070 Ti: 89 tests each, including 80 canonical states, scratch reuse and word checks. See `qualification/evidence/vkl-camera-fixture-profile-0.2.6.json`. Both compiler pins verified 8-byte diagnostic stride and no SPIR-V Int64 capability. These are fixture results; full-domain, gradient, performance and consumer adoption remain pending. The initial 72-state archive is retained as historical because its execution closure omitted the package initializer and version input; the pinned 80-state closure-v2 is the current profile.
