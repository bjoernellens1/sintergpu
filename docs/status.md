# Implementation and qualification status

This is an experimental implementation in progress. Full consolidation and
cross-vendor qualification remain incomplete. Evidence below was obtained on
2026-10-27–28 and does not transfer automatically to changed sources or compilers.

| Area | Verified scope / remaining gate |
| --- | --- |
| Catalog | 137 dispatch entries with source hashes, numerical contracts and lineage; all remain experimental |
| Locus | 17 operations; 28 GPU fixture cases through SlangPy 0.43.1 on Radeon 8060S; original scalar suite passes |
| Shared reduction | Float SUM with 5 channels and uint MIN with 1 channel preserve the 256-thread tree in exact GPU probes |
| Consumer migrations | Released-wheel differential probes pass: [StereoForge opt-in PR](https://github.com/bjoernellens1/stereoforge/pull/3), [EliGSiR-native opt-in PR](https://github.com/bjoernellens1/eligsir-native/pull/10). Both pin exact release URL/hash; downstream review/adoption and cross-vendor qualification remain pending |
| VKL | 47 executable Torch reference modules packaged; eleven modules have forward Slang coverage (including geometry/depth and temporal evidence selection); 36 modules remain without Slang translations. Three strict expected failures document extreme-depth denormal flushing; no tolerance was weakened |
| Caspar/SLAM | 11 standalone composed modules, retained generators and frozen CPU references; 5 generated math modules reproduced exactly. GPU-dependent skips, full tracking/trajectory and gradient qualification remain pending |
| xsplat | Analytic compositing gradient probe passes on local GPU; full renderer qualification is outside this library |
| Primitives | Segmented hierarchical scan and stable compaction pass CPU/GPU probes, including multilevel irregular extents; forward tree differs from donor and gradients remain pending |
| Radix sort | Compatible MIT standalone vrdx sources retained; 4 shader variants compile. Optional C++ archive, relocated installed export and CPU-only consumer static linking pass with caller-provided or separately installed volk under CMake 3.25 and both pinned Slang versions; native GPU runtime and device qualification remain pending |
| Camera color | NV12/RGB, YUYV/RGB, bilinear resize/remap and fused YUYV/remap have independent CPU and local GPU parity tests; portable FP32 profile permits 1 LSB against float64 and does not replace exact HIP semantics |
| Fusion adoption | Separate/fused outputs match; exploratory shared-device runs do not meet the controlled three-run transfer-inclusive acceptance gate. Separate pass stays the default |
| Compiler matrix | All 137 entries compile and reflect under pinned Slang 2026.12 and 2026.19; shared reflected ABI fields match and are required by CI; SlangPy's bundled compiler identity is recorded separately |
| Packaging | Universal Python wheel installs and validates outside checkout without GPU dependencies; CMake source/header install available; optional native runtime packages remain separate |
| Timing reference | All 137 catalog entries listed; paired CPU-reference/GPU fixture timings for 54 operations (392 current workload records (196 per device on Radeon 8060S and RTX 4070 Ti), 50 samples; 518 historical rows retained), 83 portable entries and four optional native variants remain pending. Raw JSON, generated table, installed CLI/CMake data; no optimization or production-throughput claim |
| Candidate intake | [Dedicated candidate branches and draft PRs](superpowers/specs/2026-10-28-kernel-candidate-intake-design.md), validation evidence and separate reviewed promotion. Rewrites must outperform or stay on par; no automated candidate qualification service is deployed |
| Overlap audit | 22 semantic groups reviewed; one exact algebra duplicate, 14 same-operation/different-contract groups, 5 shared helpers and 2 related-only groups. Not an exhaustive audit of all 249 accessible repositories |
| NVIDIA | User-selected canonical `rtx4070` host reports RTX 4070 Ti (driver 595.99.02). Retained GPU regression: 563 passed, two skipped, three existing strict expected failures; 196 parity-checked timing fixtures across 54 operations. Full-domain/runtime/application and controlled-performance qualification remain pending |
| Release lifecycle | [0.2.3 is published and immutable](https://github.com/bjoernellens1/sintergpu/releases/tag/v0.2.3); 0.2.4 is a working batch pending publication. Seven registration PRs delivered per release; a real 0.1.0 retry reused all seven PR URLs and commit hashes. The retained 0.2.3 delivery and check annotations are in [qualification/delivery-0.2.3.json](../qualification/delivery-0.2.3.json). StereoForge 0.2.3 integrity verification and both CPU checks passed; six private consumers' hosted checks remain blocked before starting by account billing/spending restrictions. Automatic publisher-App and downstream-App execution require dedicated configuration and remain unverified |

The merged [0.2.0 batch](https://github.com/bjoernellens1/sintergpu/pull/12)
corrects geometry/fusion float32 rounding and records a43-operation timing baseline.
Its publication stopped because a retained regression log was omitted from source
packages. Published0.2.1 includes the log and requires package-byte checks in PR CI.
Published0.2.2 adds eight primitive entries and irregular carry fixtures, reaching
51 paired operations and358 current records. Published0.2.3 adds the three xsplat math entries. The working 0.2.4 batch
shares G18 symbolic ownership and adds the importable point-plane module; 32 GPU
cases per device passed. All six existing baseline recipes were refreshed,
reaching 54/137 paired entries and 392 current records, with 518 historical rows
retained and 83 portable entries plus four native variants pending. Fresh Python
and relocated CMake/source installation checks pass, including a freshly built
wheel with the corrected symbolic factory. The final reviews found caller-symbol
capture and untracked release-note inputs; both are fixed with regressions.
Release notes now include changelog, compatibility, breaking changes, migrations
and qualification limits automatically; publication remains pending. Six symbolic
tests pass and all five generated outputs reproduce byte-for-byte. Imported-module
performance, full-domain qualification and consumer acceptance remain unqualified
beyond this experimental release; the original consolidation requirements remain
incomplete.
See [xsplat contracts](xsplat-contracts.md) for adapter layouts and the explicit
input-derived pixel error envelope; wide off-screen bounds are weak accuracy evidence.

Do not infer full application, physical camera, tracking accuracy or cross-vendor
qualification from a kernel fixture, CPU test, compiler pass or historical result.
See [overlap review](kernel-overlap.md), [VKL scope](vkl-ports.md),
[color contracts](color.md), [primitives](primitives.md) and
[release requirements](releases.md) for the applicable boundaries.
