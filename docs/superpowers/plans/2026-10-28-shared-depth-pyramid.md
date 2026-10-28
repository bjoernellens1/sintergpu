# Depth Pyramid Profile Consolidation Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans; execute and verify tasks in order.

**Goal:** Consolidate the shared 2×2 depth reduction into a reusable general-purpose core while retaining the VKL and SLAM numerical contracts, established launch ABIs, and equal-or-better measured performance for each qualified configuration.

**Architecture:** Keep one importable per-block core with compile-time selected profiles, separately compiled entry wrappers and the existing buffer/dispatch contracts. No runtime branch to select a profile inside each pixel. Keep frozen donor implementations and independent references; do not replace consumer calls until parity and non-regression are demonstrated on each target configuration.

**Tech stack:** Slang/SPIR-V, Vulkan/SlangPy, Python/NumPy/Torch references, existing CMake and test/benchmark infrastructure.

**Spec:** `docs/kernel-overlap.md` G05 (lines 74–82), `docs/kernel-timing-methodology.md`, and the user-approved consolidation/no-regression requirements. This plan does not assert that the two outputs are equivalent.

## Source and evidence anchors

- VKL frozen revision `3384023a311651ea4260cf12ddab1567972f89f3`, original `src/vkl/kernels/reference/depth_pyramid.py`; license/owner authorization `provenance/vkl.json:3–7`. Independent imported reference `src/sintergpu/references/vkl/depth_pyramid.py:11–34`; current executable `src/sintergpu/kernels/vkl/depth_pyramid.slang:3–35`; host plan `src/sintergpu/vkl.py:128–145`.
- SLAM frozen revision `e35e52b35f692b8337696ea3c9d6721d1cc2bc6b`; license/owner authorization `provenance/slang-rgbd-slam.json:3–7`; imported frozen tracker shader `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:1–19` and frozen donor shader at `references/donors/slang-rgbd-slam/src/slang_rgbd_slam/shaders/tracker.slang`. The donor tracker composes source and dispatches the actual entry at `references/donors/slang-rgbd-slam/src/slang_rgbd_slam/tracker.py:110–128,182–200`.
- Existing VKL stage timing cases and numerical checks: `benchmarks/vkl_timings.py:80–102,145–159`; `benchmarks/vkl_depth_resampling.py`. Frozen SLAM resident tracking benchmark verifies its pose/status and retains timestamp samples: `references/donors/slang-rgbd-slam/src/slang_rgbd_slam/benchmark.py:10–33,44–102`. The latter does not separately time transfers or isolated downsample.
- Preserve owner-original Apache notices. No research-only simple-knn or unresolved RTG utility code may enter the implementation (`provenance/README.md:13`).

## Global constraints and numerical profiles

- **VKL:** Keep `depthPyramidLevel`, its 256×1×1 workgroup, `DepthPyramidParams`, exact bindings, float32 depth/confidence, uint32 validity, batches, ceil-half dimensions, traversal and accumulation order. Usable means valid and finite positive; if no usable value emit zeros/invalid; with four usable near-equal depths use mean and mean confidence; otherwise choose minimum and first-minimum confidence. Threshold is `epsilon * 32` inclusive; preserve masked-INF behavior, float order and output shape. Host `depth_pyramid_plan` and caller-owned resources/barriers stay ABI-compatible.
- **SLAM:** Keep `downsample`, 16×16×1 workgroup, `MapParams`, bindings, exact 2D dispatch and divisible 2× source sizing. Input validity is shader predicate `z > 0`, not VKL's finite/explicit mask rule; minimum initialization is `1e20`. Preserve source order, float reduction, inclusive `depthJump`, valid-subset mean or zero and original handling of NaN/Inf/large values. Do not introduce confidence, validity arrays, different border policy or GPU readback.
- No removal of original frozen shader, fixtures, original benchmark or downstream implementation; preserve notices, attribution, hashes, and exact shader/compiler/runtime identity. No new hidden allocations, submission, synchronization or host roundtrip in recording paths.
- Existing Vulkan layouts, stride, workgroup sizes, padding, limits and reflection must be unchanged; general-purpose extension is additive and specialization is compile-time. Keep faster specialized implementations selectable if the common version fails any gate.
- Version/classify changes with changelog and migration notes; consumer updates are separate and remain under downstream merge control. Numerical changes are interface changes.

## Review focus

1. Partial 2×2 masks and the VKL masked-INF rule differ from SLAM's average-or-reject; test all 16 masks per profile.
2. Positive infinity, NaN, huge values and signed zero follow donor-specific validity and IEEE behavior; test against independent frozen implementations rather than silently normalizing inputs.
3. Threshold equality and immediately adjacent float32 values change surface decisions; test both epsilon and depthJump boundaries and accumulation order.
4. Odd/singleton VKL borders, batch indexing, SLAM even dimensions and padded workgroups must neither read nor write outside capacity; poison guards and reuse scratch.
5. A visually matching kernel can alter downstream pose or hurt latency; gate complete intermediate outputs and application outputs, then controlled old/candidate/old comparisons.

## Task 1: Freeze baselines and build independent differential fixtures

**Files:** retain `references/donors/vkl/…/depth_pyramid.py`, frozen SLAM shader and benchmark; add `tests/test_depth_pyramid_consolidation.py`, `benchmarks/depth_pyramid_consolidation.py` and focused independent CPU reference as needed. Record hashes in provenance and per-run receipts.

- [ ] Pin the donor source/compiler/runtime and capture original source, entry reflection, workgroup, binding layouts and source closure hashes before edits. Record existing source and exact workload/fixture hashes; do not overwrite released evidence.
- [ ] Add independent SLAM scalar CPU oracle reproducing the literal shader predicate, bounds, accumulation and IEEE behavior; validate it against the unmodified frozen GPU shader before using it as a candidate oracle. Retain the unmodified VKL Torch donor, independent NumPy implementation, and frozen GPU implementation. If a CPU compiler/runtime changes numeric answers, use a source- and environment-bound canonical donor fixture and retain the discrepancy as evidence; do not loosen tolerances to pass.
- [ ] Generate all 16 2×2 validity combinations, distinct/equal minima with confidence ties, all-invalid, invalid/NaN/±Inf/±zero/negative/huge values, min/max and exact epsilon or jump plus `nextafter` endpoints, cancellation/high dynamic range and deterministic seeded random data. Test VKL odd/1×N/N×1 dimensions, edge padding, multiple batches and noncanonical nonzero uint masks; test SLAM valid even input dimensions, rectangular and workgroup-edge sizes.
- [ ] Validate every output and every native/half/quarter/eighth intermediate (where available), validity and confidence exactly where logical, declared floating tolerance and bitwise identity where the donor contract demands it; assert no guard writes, input mutation, overflow or stale scratch after repeated invocations. Add reflected layouts and launch ABI comparisons under Slang 2026.12 and 2026.19, distinguishing bundled SlangPy compiler identity.
- [ ] Run tests against each incumbent first. Record explicit fixture/donor/host identities and separate expected failures; fixtures must not compare one rewritten implementation to itself.

## Task 2: Implement shared, statically specialized core

**Files:** add `src/sintergpu/kernels/common/depth_pyramid_core.slang`; modify only the two wrapper shaders as needed; tests in `tests/test_depth_pyramid_consolidation.py`, existing VKL port/layout tests, and SLAM donor/differential harness.

- [ ] Write failing tests for shared-core import, exact entry reflection, resource binding names, workgroup/dispatch dimensions and both policy outputs. A shader-only import helper is not evidence of preserved dispatch ABI.
- [ ] Implement a reusable per-block load/reduction/decision core, with separate static policy specializations and exact per-profile arithmetic sequencing. Inline or compile-time specialize so unused masks/confidence/branches disappear; inspect both generated SPIR-V and source closure. Keep VKL and SLAM public entry points, argument layouts, scheduling, ownership and semantics unchanged.
- [ ] Run CPU/reference tests, both pinned standalone compiler/reflection lanes, and differential GPU tests on Radeon 8060S and RTX 4070 Ti. Test full per-stage state, tails, readonly inputs and reuse immediately before reset. Preserve failing runs, identity and logs. Any exactness, ABI or operational failure blocks selection.

## Task 3: Controlled no-regression qualification

**Files:** `benchmarks/depth_pyramid_consolidation.py`, versioned raw results and `qualification/` evidence; retain `benchmarks/vkl_timings.py` and original SLAM benchmark independently.

- [ ] Implement identical-input old/candidate/old paired collection for each profile, device and representative workload. Include odd/singleton and multibatch VKL shapes, SLAM even shapes including the donor 160×120 tracker workload and larger realistic frames; declare inputs, warmups, samples, paired order and representative shape set before timing. Reset outputs/state identically outside isolated timestamps and validate before and after every measured sequence. Use exclusive device coordination; record GPU UUID, compiler binary SHA, driver, CPU/load, source/input hashes, clocks, timestamps and raw samples.
- [ ] For each vendor, perform **three independent controlled old/candidate/old runs** per declared profile/workload; fail/retain samples and investigate rather than silently discard a run with interference, output mismatch or altered source/environment. Bracket candidate with both old measurements to detect drift; if controls fail, declare the run inconclusive and repeat only after identifying the cause.
- [ ] Report **separately**: isolated dispatch GPU time, complete resident pyramid / SLAM tracking GPU pipeline, host submission/completion, and transfer-inclusive end-to-end time including uploads and required readbacks. The existing SLAM benchmark excludes frame-boundary transfers, so add a separately labeled transfer-inclusive harness and do not relabel its existing GPU figure. Also record CPU reference timings independently; do not compare unlike boundaries as a speedup.
- [ ] Pre-register a paired statistical analysis before measuring: compare candidate with interpolated/bracketed old samples and report all three run medians, p95s and 95% confidence intervals. To meet the user's no-regression requirement, require an upper confidence bound for the candidate-minus-old median difference **≤ 0** and candidate p95 **≤ old p95** at each required profile/workload/device/boundary; no positive tolerance is assumed. Insufficient precision or noisy data is inconclusive, not a pass. For an actual fused variant additionally satisfy the approved fusion gate: three controlled runs, ≥5% lower median pipeline time, statistically positive improvement and ≤5% p95 regression; the stricter no-regression condition above still applies to rewritten existing variants.
- [ ] Preserve complete raw sample arrays and all metadata, source/compiler identities, full correctness output evidence and receipt hashes. Historical donor timings provide recipes only, not SinterGPU measurements. Any losing or inconclusive configuration retains the incumbent and remains experimental.

## Task 4: Integration, release and consumer migration

**Files:** `provenance/`, `docs/kernel-overlap.md`, numerical contracts, timing registry/table, changelog fragment, qualification metadata, consumer update manifests and tests.

- [ ] Record each policy's lineage, exact numerical contract, supported layouts, device-specific outcome, rejected variants and uncertainty. Update catalog/searchable docs without asserting unrelated profiles are equivalent. Ingest fresh measured values without deleting prior raw samples or mislabeling stale measurements as current.
- [ ] Create a versioned release only after scoped correctness, compile/reflection, package/install, CPU and device gates pass. Give each configuration a distinct qualification; incomplete configurations remain experimental. Release notes include changelog, compatibility, breaking changes, migrations and pending gates.
- [ ] Prepare separate VKL and SLAM update PRs pinned to released source/package hashes with relevant evidence. Preserve current consumer code until its own full differential checks and controlled performance gates pass; downstream retains merge control. The SLAM pose-checked synthetic benchmark alone does not establish trajectory accuracy; run its relevant trajectory/correspondence checks before adoption.
