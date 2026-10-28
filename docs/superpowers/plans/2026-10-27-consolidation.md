# SinterGPU consolidation implementation plan

Approved by the user on 2026-10-27. Public destination: bjoernellens1/sintergpu.
License: Apache-2.0 with retained upstream notices. User authorizes publishing
their original code under Apache-2.0; that does not relicense third-party code.

NVIDIA target amended by the user on 2026-10-28: use the4070PC. The canonical
host `cps-wkstn-nvidia4070.local` reports an RTX4070Ti; qualification uses that
physical device alongside Radeon8060S. The earlier unreachable3090 is replaced.

## Global constraints

- Keep kernels independent of applications. Caller owns device, allocation,
  submission and synchronization. Optional Vulkan, Daxa and SlangPy adapters.
- Import independent references, tests, benchmarks and provenance with ports.
- Preserve exactness/gradient/ordering/layout/invalid-value and capacity semantics.
- Pinned compiler matrix includes Slang2026.12 and2026.19; identify SlangPy's compiler.
- No historical performance or skipped GPU check counts as new qualification.
- Strict SemVer from0.1.0; before1.0 incompatible changes increment minor;
  compatible additions/fixes increment patch. Unified source/package version.
- Immutable releases, checksums, attestations, changelog and downstream update PRs.
- Both8060S and the user-selected canonical RTX4070Ti qualification required.
- Fusion gate:3 controlled runs, >=5%median improvement, positive confidence bound,
  <=5%p95 regression. Report kernel/pipeline/transfer-inclusive separately.

## Task1: Portable foundation and Locus

Initialize isolated implementation worktree; build provenance ledger, catalog,
packaging/CMake, CLI and borrowed-context adapters. Import all17Locus shaders
with scalar references and fixture runner. Test artifacts and numerical fixtures.

## Task2: VKL and color/image additions

Import VKL references, tests, benchmarks,2Triton variants. Port projection/depth
evidence and reusable photometric/depth/reprojection/temporal/compaction/voxel
operations. Import EliGSiR-sim NV12/HIP references; add YUYV decode, bilinear
resize/remap and fused YUYV+calibration remap. Preserve existing numeric profiles.

## Task3: SLAM/Caspar and additional primitives

Import standalone feature/tracking/TSDF/BA/solver Slang kernels, independent
references and generation inputs. Preserve AoSoA padding and BRIEF identity.
Consolidate xsplat Adam/Gaussian math, segmented scans and compatible vrdx sort.
Keep full renderers/tracker scheduling/loop policy/application state outside.

## Task4: Real consumers

Shared256-thread Slang reduction in StereoForge(uint-min) and EliGSiR-native
(float-sum), preserving tree/barriers/identities. Bind source hashes into caches.
Migrate remaining consumers after differential parity; no forced host roundtrip.
Use isolated donor worktrees and reviewable PRs, never alter dirty main checkouts.

## Task5: Release system

Catalog/contracts/capability/provenance/license/migration/benchmark docs,
changelog fragments, CI CPU/compile/reflection/install, GPU qualification records,
versioned packages and immutable GitHub release with SHA256+attestations.
Registry of7consumers; idempotent automated update PRs pinned by version/hash,
relevant changes/evidence, consumer tests, visible failures, no auto-merge.

## Task6: Verification and publication

Separate performance review with hot-path guards and microbenchmarks before GPU
validation. Measure local8060S and canonical4070Ti, recording physical identity.
Release only accurately labeled evidence; complete expanded program only when
all port batches, real migrations, both-vendor qualification and lifecycle pass.
Independent final review required. Record unfinished gates explicitly.

## User extension: CPU and GPU timing reference

User requested a table or database with CPU and GPU execution times for every
kernel. Maintain a versioned machine-readable timing registry and a generated
searchable table covering every current operation/variant, with explicit pending
reasons until a valid workload is measured. Record actual CPU reference wall time
and GPU timestamp kernel time separately; retain resident-pipeline and full
transfer-inclusive costs when measured. Bind each record to exact source,
reference/workload, compiler, hardware/software environment and raw samples.
Never copy a pipeline time onto its stages or substitute donor timings. Reuse
existing correctness fixtures and benchmark recipes; extend workload coverage
alongside ports and qualification. Complete baseline coverage is part of the
end-to-end goal, not satisfied by an empty registry or unmeasured placeholders.
