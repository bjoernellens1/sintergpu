# Changelog

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
