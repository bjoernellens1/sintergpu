# Kernel family candidate ledger

## Current candidate checkpoint: native point-plane profile

The later iteration record is [NATIVE_ACCEPTANCE.md](NATIVE_ACCEPTANCE.md).
AoSoA point-plane at exactly 1,025 correspondences has three controlled passing
processes at GPU, resident completion and complete transfer boundaries on each
GPU, using Slang 2026.19, a 256-thread elementwise candidate launch, retained
caller-owned resources and matched polling completion. Thread CPU cost is
reported. This bounded result does not qualify Python or all supported sizes.

Iteration4 preserved the original 1024-thread adapter's load/store phases behind
shared residual, rotation and scalar functions. Its normalized instructions and
resource layout match the incumbent under both pins. Native iteration4 and5
retain every earlier failure. Iteration5 changed only the elementwise launch size
and added explicit matched wait/spin caller profiles. Both pins differ only in
the declared LocalSize; all arithmetic/order/decorations remain compared.

The longer precision experiments were declared before collection at 1,025 and
65,539 elements with 10,000 ABBA blocks and the unchanged gates. An initial local
AoSoA/1025 set overlapped report computation and is explicitly ineligible.
Iteration7 collected a fresh three-process set with analysis paused. A later
NVIDIA lock failure happened before GPU work; its log and completed runs remain,
and only the two missing processes were resumed with identical binary/artifacts.

Native reports require collection-control metadata and validate actual launch,
mode, count, affinity and groups against the declared plan. Source hashes match
each recorded Git revision. Final numerical tests pass 96 on each GPU. The fresh
review found no remaining Important defect within these source/reporting checks.
No family-wide/default/catalog/installed API/consumer/release change follows.

All history below retains its original decisions and source identities.

Owner: candidate implementation chat; managed worktree `kernel-families-candidates`.
Initial inspection base: `fe273e6946419c018c3a2dcdb20bb9c518c751e3` (parent timing branch).
Candidate rebased onto upstream main `536fbaa` per the new intake protocol. The
frozen point-plane and solve sources have identical hashes on both bases.
The parent owns catalog, candidate intake and releases. No promotion, migration,
version changes or donor edits are part of this branch.

## Ranked audit groups (before implementation)

Rank balances semantic equivalence, reusable value and an identifiable matched
GPU baseline. Different contracts stay named profiles; rank is not a claim of
performance or readiness. All 22 IDs are from `provenance/kernel-overlap.json`.

| Rank | Group | Reuse opportunity / disposition |
| --- | --- | --- |
| 1 | G18 | One symbolic point-plane expression, scalar and indexed AoSoA adapters; preserve two emitted FMA schedules |
| 2 | G21 | Left-looking Cholesky dimension/storage specialization; keep observation guards outside helper; PCG/Adam distinct |
| 3 | G07 | Pinhole scalar algebra with explicit rounding profile; parent rounded-division correction takes precedence |
| 4 | G08 | Rigid scalar transform, caller masks/layout/frame semantics retained |
| 5 | G03 | Brown-Conrady forward polynomial; inverse iteration and sampling remain separate |
| 6 | G11 | Exact fixed-tree helpers parameterized by channels/lanes; wave trees remain distinct |
| 7 | G06 | Explicit luminance weights/scale; Gaussian/box/antialiased filters stay distinct |
| 8 | G13 | Stable bounded compaction profiles with total/retained counts distinguished |
| 9 | G16 | Packed Hamming/search primitives; global-ratio and local-radius remain distinct |
| 10 | G15 | BRIEF sampling/packing helper; preserve original SLAM/VKL patterns and orientation |
| 11 | G17 | Point-plane term reuse; tangent/frame/action/robustification profiles required |
| 12 | G14 | FAST ring predicate shared; score/NMS/quota remain separate |
| 13 | G04 | Explicit finite-positive conversion profiles; bilateral separate |
| 14 | G05 | Depth pyramid profile helper; sparse support/confidence/jump policy differ |
| 15 | G09 | Normal-map stencil/orientation profiles, no sign or support substitution |
| 16 | G02 | Coordinate/border sampling helpers; quantized/fused/antialiased paths retained |
| 17 | G10 | Projection/z-buffer helpers; pixel ties/winner/provenance profiles required |
| 18 | G01 | Matrix/range metadata reuse; exact HIP/P010 and float/byte profiles retained |
| 19 | G12 | Public scan remains separate from pinned VRDX stable sort implementation |
| 20 | G22 | ISP forward profile remains distinct from normalization and Torch backward |
| 21 | G20 | Projection subexpressions only; sensor evidence/factor/Gaussian contracts unrelated |
| 22 | G19 | TSDF, voxel counts and same-grid/temporal fusion remain separate |

## First implementation tranche

G18 and the Cholesky subset of G21. Importable Slang functions have no resources,
allocation, submission, barriers, synchronization or readback hidden in them.
Adapters own layouts and guards. Frozen donors and canonical exports stay intact.
The independent float64/finite-difference references are not generated from the
candidate. Exact baseline differential tests additionally preserve FP32 ordering.

Concrete baselines: packaged Caspar composed point_plane_residual/computeMain
(1024 lanes), generated caspar_point_plane scalar function, gn.slang/solveSystem
(6x6) and dense_ba.slang/solveSchur (runtime 6/12/18 with stride18 local storage).
Sources and donor commits are frozen in the family manifest before execution.
The scalar and AoSoA residuals have different FMA accumulation order despite the
same symbolic expression; the candidate must preserve both schedules explicitly.

## Predeclared measurement and decision method

Use SlangPy 0.43.1's bundled Slang compiler, precise FP mode, Vulkan, hardware
adapter identity and compiler binary SHA256. Radeon8060S is available locally.
The initial selected RTX3090 host is probed separately; no replacement device counts.
Controlled collection owns an advisory lock and records PID, host and start time.
Reject software adapters. Check active GPU users before controlled execution.

Each matched workload uses at least three fresh processes, fixed seeded inputs,
10 warmups and 100 ABBA blocks (baseline,candidate,candidate,baseline), retaining
every sample and order. Report isolated GPU timestamps, resident recording +
submission + completion wall time, transfer-inclusive uploads + dispatch + full
readback, and independent CPU reference separately. Compilation/allocation are
outside timed regions and their changes are stated. No baseline reset is omitted.
Only compare identical operations/layouts/profiles and output boundaries.

Per run and boundary, resample whole ABBA blocks, retaining their four samples.
Fixed seed bootstrap (10,000 resamples) reports 95% upper bounds on ratios of
candidate/baseline raw-sample medians and p95s. On-par requires upper bounds <=1.03 median and <=1.05
p95, in every independent run and supported workload. Faster requires median
upper bound <1.0 and the same p95 gate. These margins are declared before timing,
not fitted to results. Retain baseline for failures, missing or noisy evidence.
Never aggregate configurations to cancel regressions. Batch capability is a
distinct profile; a batched solve cannot qualify the single-solve profile.

Representative point-plane counts: 1,63,1023,1024,1025,4099,65539; indexed duplicate
gathers and nontrivial tails. Solve dimensions 6,12,18 with SPD, singular,
indefinite, nonfinite and observation-count failures; single and batched work
are separate. Any changes to this recipe require a recorded reason before runs.

## Recoverable progress

- Worktree created from parent's numerical corrections; central worktree untouched.
- Audit groups ranked; baseline source schedules inspected.
- Two importable helper implementations and symbolic export adapters written.
- 35 numerical tests passed on Radeon8060S before timing (2026-10-28).
- Shared local timing lock: `/tmp/sintergpu-gpu.lock`; ownership evidence:
  `/tmp/sintergpu-gpu-owner.json`. No other timing process observed before testing.
- Candidate performance qualification pending.

## Iteration 1 rejection and iteration 2 declaration (before timing)

Three processes finished at 06:19:48 UTC before the parent regression notice.
Raw data is retained under `evidence/iteration1/`. No candidate replaces a baseline:
standalone point-plane tails were noisy and Cholesky median GPU latency regressed
approximately 3-6%. The original runner/support are recoverable at Git `24e74ad`
(before relocation); JSON retains exact hashes. This is rejection evidence.

SPIR-V inspection confirms four function-local arrays in helper6 versus three in
baseline6. Iteration2 writes directly into the returned solution array and removes
that extra local array/copy. Original loop order and failure outputs are retained.

G18/G11 iteration2 additionally fuses point-plane residual/Jacobian generation and
first-level normal equations, preserving both 64-lane fixed-tree and subgroup plus
fixed-tree profiles. Every original residual/Jacobian and partial remains exported;
the downstream hierarchical reduction and Cholesky remain the incumbents. Thus
the candidate removes one dispatch and intermediate reads without hiding a layout,
output, reduction-tree or solver change. The normal-term helper is reusable across
scalar residual dimensions. Full GN outputs must agree bitwise before/after timing.

Iteration2 retains the 100-block/10-warmup/three-process ABBA and bootstrap method.
It adds a stricter rejection for statistically credible slowdown (one-sided 95%
lower ratio >1 for median or p95) to the existing upper uncertainty gates.
Fusion additionally requires upper median ratio <=0.95 and upper p95 <=1.05 at
each applicable pipeline boundary. GPU producer and complete GN-step timestamps
are reported separately, alongside complete resident host wall and full transfer
cost. Counts 1,63,64,65,1025,4099,65539 cover both tree profiles. No result from
iteration1 is relabeled under this revised method; those original decisions stand.

User steering via parent: RTX4070 now replaces RTX3090 as the NVIDIA target.
The retained 3090 SSH failure is historical only. New 4070 execution is pending
its hostname/address and verified physical GPU/driver identity; parent is locating
the PC. No other NVIDIA device qualifies this target.

Parent verified canonical target: `bjoern@cps-wkstn-nvidia4070.local`, actual GPU
NVIDIA GeForce RTX4070Ti, UUID `GPU-96d2dba0-95fd-5c00-e849-2e0610f24e06`, driver
595.99.02, bus 00000000:01:00.0. Candidate will independently record this identity.
The realtime4070 machine is a different GPU and is not the selected target.

## Iteration 3 declaration (before timing)

Iteration2 removed the extra local array, confirmed by SPIR-V, but runtime Schur
profiles still measured 3-6% slower. Instruction counts identify added dimension
guard control flow and failure-zeroing loop. The incumbent already requires a
host-validated dimension. The candidate now separates a prevalidated factor/solve
core from a checked/zero-on-failure convenience function. Incumbent adapters use
the core and perform their original failure-select writes. Their external zero
failure updates and dimension preconditions remain unchanged. Checked convenience
semantics are tested separately and are not claimed performance-qualified.

Fresh three-process ABBA runs on both Radeon8060S and the selected RTX4070Ti use
the same iteration2 statistical and fusion gates, all 32 workload/profile cases,
100 blocks/10 warmups. Both standalone compiler lanes are rerun. Historical
iterations remain rejected and are retained without relabeling. Parent declared
its 4070Ti collection terminal before candidate GPU execution is started.
