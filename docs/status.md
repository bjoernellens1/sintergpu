# Implementation and qualification status

This is an experimental implementation in progress. Full consolidation and
cross-vendor qualification remain incomplete. Evidence below was obtained on
2026-10-27 and does not transfer automatically to changed sources or compilers.

| Area | Verified scope / remaining gate |
| --- | --- |
| Catalog | 130 dispatch entries with source hashes, numerical contracts and lineage; all remain experimental |
| Locus | 17 operations; 28 GPU fixture cases through SlangPy 0.43.1 on Radeon 8060S; original scalar suite passes |
| Shared reduction | Float SUM with 5 channels and uint MIN with 1 channel preserve the 256-thread tree in exact GPU probes |
| Consumer migrations | Released-wheel differential probes pass: [StereoForge opt-in PR](https://github.com/bjoernellens1/stereoforge/pull/3), [EliGSiR-native opt-in PR](https://github.com/bjoernellens1/eligsir-native/pull/10). Both pin exact release URL/hash; downstream review/adoption and cross-vendor qualification remain pending |
| VKL | 47 executable Torch reference modules packaged; nine modules have forward Slang coverage (including geometry/depth and temporal evidence selection); 38 modules remain without Slang translations. Three strict expected failures document extreme-depth denormal flushing; no tolerance was weakened |
| Caspar/SLAM | 11 standalone composed modules, retained generators and frozen CPU references; 5 generated math modules reproduced exactly. GPU-dependent skips, full tracking/trajectory and gradient qualification remain pending |
| xsplat | Analytic compositing gradient probe passes on local GPU; full renderer qualification is outside this library |
| Primitives | Segmented hierarchical scan and stable compaction pass CPU/GPU probes, including multilevel irregular extents; forward tree differs from donor and gradients remain pending |
| Radix sort | Compatible MIT standalone vrdx sources retained; 4 shader variants compile. Local optional C++ static archive compiles with caller-provided volk; installed native export, runtime and device qualification pending |
| Camera color | NV12/RGB, YUYV/RGB, bilinear resize/remap and fused YUYV/remap have independent CPU and local GPU parity tests; portable FP32 profile permits 1 LSB against float64 and does not replace exact HIP semantics |
| Fusion adoption | Separate/fused outputs match; exploratory shared-device runs do not meet the controlled three-run transfer-inclusive acceptance gate. Separate pass stays the default |
| Compiler matrix | All 130 entries compile and reflect under pinned Slang 2026.12 and 2026.19; shared reflected ABI fields match and are required by CI; SlangPy's bundled compiler identity is recorded separately |
| Packaging | Universal Python wheel installs and validates outside checkout without GPU dependencies; CMake source/header install available; optional native runtime packages remain separate |
| Overlap audit | 22 semantic groups reviewed; one exact algebra duplicate, 14 same-operation/different-contract groups, 5 shared helpers and 2 related-only groups. Not an exhaustive audit of all 249 accessible repositories |
| NVIDIA | Selected RTX 3090 workstation unreachable; execution pending |
| Release lifecycle | [0.1.4 is published and immutable](https://github.com/bjoernellens1/sintergpu/releases/tag/v0.1.4); hosted attestations and released checksums verified. Seven registration PRs delivered per release; a real 0.1.0 retry reused all seven PR URLs and commit hashes. The 0.1.3 delivery and fresh check annotations are retained in qualification/delivery-0.1.3.json. StereoForge integrity checks passed; its default CPU workflow still has a benchmark import-path failure addressed in the opt-in migration PR; six private consumers' hosted checks were blocked before starting by account billing/spending restrictions. Automatic publisher-App and downstream-App execution require dedicated configuration and remain unverified |

Do not infer full application, physical camera, tracking accuracy or cross-vendor
qualification from a kernel fixture, CPU test, compiler pass or historical result.
See [overlap review](kernel-overlap.md), [VKL scope](vkl-ports.md),
[color contracts](color.md), [primitives](primitives.md) and
[release requirements](releases.md) for the applicable boundaries.
