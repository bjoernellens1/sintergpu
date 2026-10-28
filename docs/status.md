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
| Timing reference | All 137 catalog entries listed; paired CPU-reference/GPU fixture timings for 43 operations (106 current workloads, 50 samples; 126 historical rows retained), 94 portable entries and four optional native variants remain pending. Raw JSON, generated table, installed CLI/CMake data; no optimization or production-throughput claim |
| Candidate intake | [Dedicated candidate branches and draft PRs](superpowers/specs/2026-10-28-kernel-candidate-intake-design.md), validation evidence and separate reviewed promotion. Rewrites must outperform or stay on par; no automated candidate qualification service is deployed |
| Overlap audit | 22 semantic groups reviewed; one exact algebra duplicate, 14 same-operation/different-contract groups, 5 shared helpers and 2 related-only groups. Not an exhaustive audit of all 249 accessible repositories |
| NVIDIA | Selected RTX 3090 workstation unreachable; execution pending |
| Release lifecycle | [0.1.9 is published and immutable](https://github.com/bjoernellens1/sintergpu/releases/tag/v0.1.9); hosted attestations and released checksums verified. Seven registration PRs delivered per release; a real 0.1.0 retry reused all seven PR URLs and commit hashes. The 0.1.9 delivery, verified changelog excerpts and fresh check annotations are retained in qualification/delivery-0.1.9.json. StereoForge integrity checks passed; its default CPU workflow still has a benchmark import-path failure addressed in the opt-in migration PR; six private consumers' hosted checks were blocked before starting by account billing/spending restrictions. Automatic publisher-App and downstream-App execution require dedicated configuration and remain unverified |

The upcoming [0.2.0 batch](https://github.com/bjoernellens1/sintergpu/pull/12)
corrects geometry/fusion float32 rounding and records the timing coverage above.
Those changes remain under review until that PR is merged and released.

Do not infer full application, physical camera, tracking accuracy or cross-vendor
qualification from a kernel fixture, CPU test, compiler pass or historical result.
See [overlap review](kernel-overlap.md), [VKL scope](vkl-ports.md),
[color contracts](color.md), [primitives](primitives.md) and
[release requirements](releases.md) for the applicable boundaries.
