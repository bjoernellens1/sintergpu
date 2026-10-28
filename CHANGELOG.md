# Changelog

## 0.2.8 — 2026-10-28

- Add an experimental, fixture-bound 256-thread point-plane AoSoA dispatch, checked caller-owned plan and packaged historical fixture. It requires fresh GPU parity and performance qualification for its imported compilation; existing defaults remain unchanged.
- Add experimental Caspar layout-conversion plans, references and shaders, plus provisional source-closure and timing diagnostics. These do not qualify a GPU path, performance or a new timing freshness policy.
- Add allocation-observation diagnostic patches and CPU-only orchestration checks. Production execution remains blocked pending source-backed provider, loader and ICD identity; no GPU qualification or physical allocation attestation is claimed.
- Add experimental same-backend Vulkan device-limit and dispatch-address observation patches with CPU-only checks. The diagnostic receipt schema changes and requires a paired runtime rebuild; provider identity, physical backing and GPU qualification remain unverified.
- The conservative timing registry covers 164 portable entries with zero source-current paired measurements. It retains 2,478 historical measurement rows and their samples; four native timing targets also remain pending.

## 0.2.7 — 2026-10-28

- Add experimental VKL stable bounded batched compaction and a separate logical backward Torch CPU-observed bit profile, with opt-in caller-owned recording, explicit resource and synchronization contracts, and Python and CMake shader discovery. Existing defaults remain unchanged.
- Retain RTX 4070 Ti forward and backward CPU-profile candidate probes and a Radeon 8060S forward candidate probe. The Radeon forward probe covers six fixtures, including five frozen Torch donor comparisons and one NumPy-only three-byte-width case. Relocated packaged shaders, Radeon backward execution, performance, framework autograd integration and downstream adoption remain unqualified; no speedup or optimized default is claimed.
- The timing registry now covers 163 portable entries with no source-current paired measurements. Its conservative whole-tree dependency hashes changed with added shaders and references, making the previous 816 current rows historical (the 72/155 paired coverage applied to v0.2.6 only); all 2,478 measurement rows and their raw samples remain in the registry. Four optional native timing targets also remain pending.

## 0.2.6 — 2026-10-28

- Add experimental VKL eager float32 camera rectification, reusable grid and ray preparation, and staged depth registration: five public functions use nine dispatches and caller-owned resources. Preserve complete diagnostics, collision and tie semantics, and the explicit little-endian two-word int64 diagnostic ABI. Existing Locus contracts and defaults remain unchanged.
- Validate the canonical 80-state, 347-output donor fixture profile with 89 tests on each Radeon 8060S and RTX 4070 Ti. The initial incomplete source-closure archive remains historical evidence. Wider domains, gradients, controlled no-regression and consumer adoption remain pending; no speedup or optimized default is claimed.
- Record 816 source-current CPU/GPU fixture timing rows across 72 of 155 portable operations, split evenly between Radeon 8060S and RTX 4070 Ti. The remaining 83 portable and four native timing targets remain pending. Compact the timing registry JSON without changing its schema, samples or identities.

## 0.2.5 — 2026-10-28

- Add experimental VKL reprojection and prior-only temporal RGB-D reference profiles, preserving collision support, diagnostic projections, normalized-coordinate sampling and every numerical output. Reuse released geometry and depth fusion with explicit caller-owned resources, independent references and source-bound validation. Existing dispatch contracts and defaults remain unchanged.
- Validate all 63 retained cases / 113 states on Radeon 8060S and RTX 4070 Ti against the same frozen CPU reference profile; retain initial failed attempts, precision fixes and exact compiler/device identities. Record 752 paired fixture timing rows across both devices, covering 63 of 146 portable operations; 83 portable and four native targets remain pending. These measurements do not select an optimized default.

## 0.2.4 — 2026-10-28

- Consolidate point-plane symbolic algebra behind shared vector/scalar generation adapters while retaining all five generated math modules byte-for-byte. Install reusable Slang residual/Jacobian functions with explicit preserved FMA schedules, actual import/instruction/layout checks, optional generation inputs and integration examples. Existing dispatches and defaults remain unchanged; qualification is configuration-specific.
- Generate GitHub release notes automatically from the exact version changelog and compatibility fragments, with explicit breaking changes, migration instructions and qualification limits. Bind the notes to the checksummed release artifacts and reject missing or invalid notes before publication.

## 0.2.3 — 2026-10-28

- Add independent timing recipes for the three existing xsplat math entries, with complete Adam inout resets, reflected projection storage layouts, explicit matrix layout and fixed-probe scope. Retain donor headers and compare projection against the unmodified C++ reference on their shared valid domain. Record17 workloads per Radeon8060S/RTX4070Ti, reaching54/137 paired entries and392 current records;83portable+4native targets remain pending. Retain original stress inputs, failed setup/numerical-assumption evidence, runtime ABI checks and actual-program precision guards. Document input-derived pixel error bounds while preserving strict visible/conic/depth/Adam/probe and exact flag/tail checks. Numerical shader sources are unchanged.

## 0.2.2 — 2026-10-28

- Add per-entry CPU/GPU timing recipes for all eight existing segmented scan and stable compaction entries, with independent stage oracles, source-bound raw samples and explicit initialization/tail checks. Record 73 workloads on each Radeon 8060S/RTX 4070 Ti: paired coverage is 51/137 entries with 358 current records; 86 portable and four native timing targets remain pending. Kernel sources and numerical contracts are unchanged.

## 0.2.1 — 2026-10-28

- Include retained qualification logs in source packages and verify complete package source bytes in PR CI. The 0.2.0 publication stopped before attestation/upload because the new NVIDIA log was omitted; its tag is preserved. This patch publishes the 0.2.0 numerical and timing batch with unchanged kernel sources.

## 0.2.0 — 2026-10-28

- Correct camera-ray and fused-depth float32 division, and preserve point/weighted-product rounding boundaries, so reciprocal/FMA differences no longer amplify into out-of-tolerance normals. This numerical-output change increments the pre-1.0 minor version; recheck geometry consumers before adoption.
- Add paired CPU/GPU baseline recipes for the remaining 13 already-ported VKL entries, with 38 seeded workloads covering projection, image conditioning, depth conversion/pyramids/geometry/fusion, temporal selection and depth resampling.
- Retain exact output/mask/tail and existing floating-point parity requirements, separate kernel/host/transfer timing, and full workload/source/compiler identities. No optimized CPU or complete pipeline claim.
- Record 212 current-source paired timing workload records across 43/137 catalog entries: 106 each on Radeon 8060S and the user-selected RTX 4070 Ti. Retain historical samples and explicit remaining coverage. Add source-bound NVIDIA Vulkan regression and fixture evidence.
- Document candidate draft PR intake, validation and separate promotion, including the no-regression gate for general-purpose consolidation.
- Clarify that the reusable binary32 helper does not specify input NaN payloads; feature numerical behavior is unchanged.

## 0.1.9 — 2026-10-27

- Add VKL BRIEF and bounded radius-gated binary matching as seven experimental Slang entries, preserving donor patterns, native int64 coordinates, exact masks/ties and float32 subnormal decisions.
- Add independent NumPy references, frozen Torch differential tests and borrowed-encoder recording recipes. Keep SLAM patterns and global-ratio matching as separate contracts.
- Add paired per-entry CPU/GPU timing recipes and refresh existing measurements for the expanded source tree; retain raw samples and explicit pending coverage.

## 0.1.8 — 2026-10-27

- Fix optional native radix CMake installation with caller-local `volk` targets, without requiring callers to export their dependencies.
- Resolve either installed `volk` target spelling and preserve relocated resource paths on CMake 3.25.
- Require real native archive, relocated consumer and CPU-only static-link checks under both pinned Slang versions. Native GPU runtime and timing qualification remain pending.

## 0.1.7 — 2026-10-27

- Add paired CPU/GPU timing recipes for all six camera color, resize and remap operations, retaining numerical-profile, padding and transfer-cost boundaries. Record 30 workloads; paired timing coverage is now 23/130 portable entries, with 107 portable and four native variants pending.

## 0.1.6 — 2026-10-27

- Add a source-bound CPU/GPU timing registry and generated table covering every catalog entry, with explicit missing/stale baselines.
- Record paired timings for 28 Locus fixtures across 17 operations, retaining raw samples, hardware/compiler/workload identity and separate kernel/transfer costs.
- Expose timing data through `sinter timings`, Python package resources and CMake data installation.

## 0.1.5 — 2026-10-27

- Compare shared reflection ABI fields across both pinned compiler lanes in required CI.
- Include hash-verified release changelog excerpts and explicit registration/adoption instructions in downstream update PRs.
- Preserve existing PR descriptions throughout verification, including legacy deliveries; write enriched context only on new PRs.
- Require one exact released changelog heading for publication and delivery, rejecting lookalike and duplicate versions.

## 0.1.4 — 2026-10-27

- Add native depth initialization, nearest depth lift, conservative RGB-guided filling and direct-provenance labeling.
- Compose full bounded pyramids and downsample outputs from the unchanged released pyramid-level source, with explicit caller-owned buffers and no intermediate readback.
- Preserve initial direct masks across repeated filling and verify threshold, invalid-value, capacity and exact flat-depth cases. Use exact rounded-ratio decisions to prevent approximate GPU division from changing fill masks.

## 0.1.3 — 2026-10-27

- Add the VKL temporal best-observation forward profile, independent NumPy oracle, bounded setup and full-output differential tests.
- Address freshly created draft releases using their creation response ID, tolerating release-list lag while requiring an exact tag ref and resolved source commit.

## 0.1.2 — 2026-10-27

- Add VKL depth geometry, rigid transforms, same-grid fusion and complete forward temporal depth fusion.
- Preserve NaN confidence, signed-zero and constant-depth invariants; add explicit capacity/dispatch planning.

- Address draft releases by immutable release ID instead of the published-tag endpoint.
- Require a dedicated publisher App for the administrative immutability preflight.

## 0.1.0 — 2026-10-27

- Consolidate Locus image/depth/geometry/perception kernels with scalar fixtures.
- Package Caspar/SLAM shader compositions, generated math and independent references.
- Preserve VKL reference/Triton and EliGSiR-sim color/HIP resources with provenance.
- Add standalone xsplat Adam/Gaussian resources and repair its differentiable probe.
- Add catalog, CLI, borrowed SlangPy/Vulkan/Daxa adapters and CMake export.
- Add fixed256-lane shared reduction and numerical GPU probes.
- Package sources for Python and C++ with one version and retained notices.
- Add four partial forward VKL Slang profiles and 47 optional Torch reference modules.
- Add hierarchical segmented scan, stable compaction and optional MIT vrdx source target.
- Add portable camera color/sampling profiles and experimental byte-parity YUYV/remap fusion.
- Document 22 semantic overlap groups and bind the review to its source input hashes.
- Require classified changelog fragments and version transitions in PR CI.
- Add immutable release, checksum/attestation verification and idempotent consumer PR tooling.

Qualification: experimental library. Local tests and exact hardware/compiler
records are scoped evidence. NVIDIA, consumer migrations and full campaign gates
remain pending until their records pass.
