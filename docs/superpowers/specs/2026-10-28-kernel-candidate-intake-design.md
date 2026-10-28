# Kernel candidate intake and promotion

Status: contribution protocol using existing GitHub draft PRs and existing release
checks. No automatic candidate qualification service is deployed by this document.
The repository owner requested candidate submissions from agents across projects,
validation before integration, and consolidation without performance regressions.

## Submission route available now

The canonical repository is https://github.com/bjoernellens1/sintergpu. Any agent
with repository write access may push a dedicated `candidate/<project>/<topic>`
branch and open a **draft PR** titled `[candidate] <operation>: <change>`. Without
write access, use a fork and draft PR. Never push candidates directly to `main`.
Use a separate worktree; preserve other agents' changes. Candidate branches start
from current upstream main and record that exact base commit.

Keep new unqualified implementations, tests, reference fixtures and evidence under
`candidates/<project>-<topic>/` on the candidate branch. That directory is a review
workspace, not an installed API. Do not merge the candidate PR, add its operations
to the runtime catalog, change default dispatch, replace consumer code, or modify
published releases. Candidate branches need no library version bump. Existing
normal PR CI may reject a missing version/changelog; this is expected for a draft
candidate and is not permission to weaken or bypass release checks. Record candidate
validation separately in the PR. Promotion uses a separate PR with all normal gates.

Existing library code being replaced remains immutable as the benchmark baseline;
put rewrites and required local wrappers in the candidate directory. Describe any
patch needed to exercise a candidate; never silently benchmark baseline code under
a candidate name. A candidate revision is its exact Git commit, not a moving branch.

## Required review packet

Put `candidate.json` and `README.md` beside the candidate source. The JSON is a
review manifest, manually reviewed for now, not an enforced published schema. It
must include:

- A stable candidate ID, upstream base commit, superseded candidate commit if any,
  submitting project and the operations/overlap groups addressed.
- Donor repository URL, full commit, source paths and SHA-256 hashes; authorship,
  license/copyright/notice paths and changes made. Preserve third-party notices.
  Owner-original code may be Apache-2.0. Research-only simple-knn and unresolved
  RTG utilities remain reference-only; do not copy them into distributable code.
- Candidate source paths/hashes, entry points, compiler versions/hashes/flags,
  requirements, layouts/strides/padding/capacities and optional dependencies.
- Numerical contract: types, units, coordinate conventions, invalid/NaN/Inf rules,
  rounding/reduction order, exactness/tolerances, masks, sorting/tie behavior,
  initialization/overflow requirements and gradients where applicable.
- Ownership of allocation, scratch, command recording, submission and completion.
  Reuse the caller's device/context; no hidden submission or full-frame readback.
- Independent CPU references, frozen donor differential fixtures, test commands,
  expected results and evidence paths/hashes. Include boundary and capacity tests.
- Benchmark recipe, frozen baseline source hashes, workloads/configurations,
  raw samples and hardware/driver/compiler identity; CPU reference, GPU kernel,
  GPU-resident pipeline and transfer-inclusive timing must be distinct.
- A gate table with evidence for each result and an explicit pending reason for
  missing hardware, unsupported adapters, unexecuted tests or failed gates.

README must give commands that run from the candidate directory without modifying
the baseline, including compiler import paths and reflection checks. Baseline
library tests do not validate an unregistered candidate: explicitly compile and
execute each candidate entry against its references and donor fixtures.

## Validation and states

Use this PR-body checklist, with exact evidence links and commits:

- [ ] Submitted: complete manifest, legal provenance and reproducible recipe.
- [ ] Static review: resource/synchronization/transfer paths, capabilities and ABI.
- [ ] CPU/reference parity: independent references and frozen donor differential
      tests, retaining existing strictness, ordering, gradients and capacities.
- [ ] Both pinned standalone compiler lanes (Slang 2026.12 and 2026.19): compilation
      and reflection/layout parity. Record SlangPy's actual compiler separately.
- [ ] Device parity: each proposed operation/variant/adapter/configuration on
      Radeon 8060S and the selected RTX 3090 workstation. Missing execution is pending.
- [ ] Performance: the matching controlled baseline/candidate comparison below.
- [ ] Maintainer review: scope and evidence accepted for a separate promotion PR.

The author may mark a test result with its evidence, but cannot declare the
candidate integrated or select it as a default. The maintainer records accepted,
needs-work, rejected or deferred in the draft PR, with reasons. Review does not
substitute for execution. Keep failed/noisy candidates available for iteration;
update provenance and rerun all affected checks after any source/compiler change.

Full cross-vendor/default qualification requires both specified devices. A release
may expose a clearly experimental variant with partial qualification if separately
reviewed and documented; it must not replace a qualified baseline on untested or
regressing configurations. No trajectory or application-quality claim follows
from kernel parity alone.

## General-purpose consolidation and no-regression gate

Start with the semantic overlap audit in `docs/kernel-overlap.md` and the
machine-readable provenance inventory. Share resource-independent math, contracts
and implementation where equivalent, with explicit compile-time profiles for
meaningful differences. Do not merge nearest/bilinear, packed/planar YUV, BRIEF
patterns, padded AoSoA layouts or reduction trees merely because names resemble.
Specialize reusable cores where needed; a universal runtime switch is not a goal.

Freeze the best semantically equivalent incumbent for each relevant workload,
not an artificially slow reference. Include representative sizes, awkward tails,
sparsity, layouts and optional profiles. Declare measurement scopes, the paired
comparison/statistical method and acceptable measurement uncertainty before runs.
Run at least three controlled paired/interleaved repetitions; keep raw samples,
run order and environment. Coordinate exclusive use of the physical GPU across
agents; a process-local lock alone does not prove the machine was idle.

A rewritten kernel must outperform or stay on par for median and p95 at the
relevant kernel and complete-pipeline boundaries, including transfer-inclusive
cost for CPU consumers. A positive aggregate speedup must not hide a regression
in a supported workload/profile. Report confidence intervals and uncertainty;
no statistically credible slowdown is acceptable, and an inconclusive result is
pending, not a pass. Preserve the incumbent when correctness/performance fails.
Do not relax numerical tolerance or omit allocations/transfers to pass this gate.
Controlled fusion has the additional existing gate: at least 5% lower median
pipeline time, statistically positive improvement and at most 5% p95 regression
across three runs. If a fusion also replaces an incumbent under this consolidation
policy, the stricter no-regression requirement applies.

## Promotion, versions and distribution

Only after evidence review, open a separate integration PR that moves accepted
source into the appropriate importable functions/dispatch/adapters/reference layers,
adds provenance and catalog records and installs only the accepted artifacts.
Attach the frozen candidate commit and evidence; close its draft PR by reference
after promotion. The integration PR must pass normal CI, current-source qualification
checks, installation examples and consumer differential checks, with a changelog
fragment and compatibility classification. Numerical-output changes are interface
changes. Version shaders, contracts, adapters and packages together; pre-1.0 breaking
changes increment minor, compatible additions/fixes patch.

Publish through the existing immutable release/checksum/attestation process and
registered consumer update PRs; consumers retain merge control. Candidate pushes
are never releases and never automatic downstream updates. Python wheels/source
packages and CMake/C++ resource packages receive the same released kernel sources.
No duplicate Python-only implementation is required for C++ consumers.

## Implementation choices made here

Use existing GitHub branches/draft PRs as the durable intake queue, rather than
new credentials, a daemon or direct cross-repository writes. State is the PR body,
review and attached immutable evidence. No special labels or automated candidate
CI exemption are prerequisites. This keeps submission available immediately while
preserving the existing strict integration/release checks. A future dedicated
candidate workflow or machine-enforced schema is optional infrastructure and must
not be advertised as already running.
