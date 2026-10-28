# Kernel family candidate ledger

Owner: candidate implementation chat; managed worktree `kernel-families-candidates`.
Base: `fe273e6946419c018c3a2dcdb20bb9c518c751e3` (parent timing branch).
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
The selected RTX3090 host is probed separately; no replacement device counts.
Controlled collection owns an advisory lock and records PID, host and start time.
Reject software adapters. Check active GPU users before controlled execution.

Each matched workload uses at least three fresh processes, fixed seeded inputs,
10 warmups and 100 ABBA blocks (baseline,candidate,candidate,baseline), retaining
every sample and order. Report isolated GPU timestamps, resident recording +
submission + completion wall time, transfer-inclusive uploads + dispatch + full
readback, and independent CPU reference separately. Compilation/allocation are
outside timed regions and their changes are stated. No baseline reset is omitted.
Only compare identical operations/layouts/profiles and output boundaries.

Per run and boundary, pair each block's candidate mean with baseline mean. Fixed
seed bootstrap (10,000 resamples) reports 95% upper bounds on median and p95
candidate/baseline ratios. On-par requires upper bounds <=1.03 median and <=1.05
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
- Candidate implementation and qualification pending.
