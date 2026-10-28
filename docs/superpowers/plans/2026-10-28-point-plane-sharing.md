# Shared point-plane implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate G18's owned symbolic algebra and provide reusable Slang point-plane functions while preserving existing shaders and ABIs byte-for-byte.

**Architecture:** One optional symbolic factory feeds vector/AoSoA and scalar generation adapters; frozen donor generators remain independent references. Install the reviewed, resource-independent Slang core through the existing Python module locator and CMake source package. Preserve both FMA schedules as compile-time profiles, with no default dispatch replacement.

**Tech Stack:** Python>=3.11; optional SymForce0.12.0; Slang2026.12/2026.19; SlangPy0.43.1 for explicit device tests.

**Spec:** Approved `2026-10-27-consolidation.md` Task3; `../specs/2026-10-28-kernel-candidate-intake-design.md`; G18 in `docs/kernel-overlap.md`; user instruction to consolidate without performance regressions.

## Global constraints

- Candidate source frozen at PR14 commit2def63bb85adfcb0cb5b1b8d0910bed31dda0539.
- Preserve frozen references, padded AoSoA4+2 output, indexed gathers, rotation-first Jacobian, FMA phase order and invalid-value behavior.
- All five existing generated modules must retain exact source SHA256. Existing dispatch sources, entry points, capacities and default selections stay unchanged.
- Shared functions allocate no storage and submit/synchronize nothing. Optional generation dependencies are not runtime dependencies.
- Native256 performance evidence covers exact seeded AoSoA1025/Slang2026.19/polling/cached-staging configuration only; do not infer other sizes, inputs, Python or installed-import timing from it.
- Numerical-output changes require a minor version; compatible new modules/factories use patch0.2.4. Qualification and original unfinished program remain explicit.

## Review focus

- Vector/scalar algebra must share one maintained factory, while differently ordered emitted FP32 schedules remain distinct.
- Reflection and executable equivalence must cover an actual module import, not concatenated source alone.
- Padding/tails/duplicate gathers and exceptional values need direct differential checks; finite-difference checks do not substitute for bit parity.
- Installed wheel/source/CMake consumers must locate the same source without optional generation/GPU packages.
- Candidate timings must not become a blanket performance/default-selection claim after moving source or changing compilation.

### Task1: Shared symbolic generation

Files: create `src/sintergpu/codegen/{__init__,point_plane}.py`, `tools/codegen/caspar_point_plane.py`, `tests/test_point_plane_codegen.py`; modify `tools/regenerate_math.py`, `provenance/generated-math.json`.

Interfaces: `point_plane_expression(sf,point,target,normal)->(residual,V6)` and `point_plane_scalars(sf,px,py,pz,qx,qy,qz,nx,ny,nz)->tuple7`; sf is the caller's symbolic module. Owned optional generator exposes `residual_kernel()`, `factor_kernels()`, `dense_factor_source()` matching frozen export names/metadata. Port donor generator adapters with provenance; both point-plane exports call the shared factory. Remaining photometric/dense algebra is retained unchanged as generation input.

- [ ] Write tests for a hand-computed residual/Jacobian and scalar/vector parity; loading factory must import no optional runtime. Run RED for missing factory.
- [ ] Import reviewed shared factory and owned generator adapters; exact output bytes must match independent frozen donor generation for vector residual and all scalar dense factors. Run tests RED before adapters, then GREEN.
- [ ] Route regeneration through owned adapters. Run `PYTHONPATH=src <symforce-python> tools/regenerate_math.py --check`; require all5 modules reproduced exactly. Preserve tool/input/version identities in provenance. Verify no frozen source edits.
- [ ] Full CPU suite and commit.

### Task2: Installed Slang shared math

Files: create `src/sintergpu/kernels/optimization/point_plane.slang`, `examples/point_plane_import.slang`, `tests/test_point_plane_module.py`, appropriate CPU test/benchmark support; modify `src/sintergpu/data/modules.json`, `MANIFEST.in` for example shader inclusion.

Interface: preserve candidate `sinterPointPlaneResidual<let AoSoASchedule:bool>(float3,float3,float3)->float`, `sinterPointPlaneRotation(float3,float3)->float3`, `sinterPointPlaneScalars<let AoSoASchedule:bool>(nine floats,out seven floats)`. Locate via `sintergpu.module_path('optimization.point_plane')`.

- [ ] Write missing-module/import compiler tests RED. Copy exact reviewed core; register module and import example. Compile actual imports under both pinned compilers, compare executable instructions and resource layouts to unchanged baseline adapters; tolerate only explicit LocalSize differences in separate256 profile.
- [ ] Add opt-in physical GPU differential tests over both schedules, randomized/adversarial indexed/duplicate gathers, extents1/63/1023/1024/1025/4099/65539, byte padding/tails and exceptional values. Independent analytic/finite-difference residual/Jacobian checks supplement frozen baseline byte checks.
- [ ] Validate both physical vendors with coordinated GPU ownership; record compiler/device/source identities. No new optimization claim without separately controlled pipeline timing; use instruction-identical default-size proof for unchanged arithmetic and retain incumbents.
- [ ] Verify Python and relocated CMake source installation, full suite; commit.

### Task3: Versioned reviewed publication

Files: version/changelog/contracts/overlap/provenance/qualification/status; retain release0.2.3 delivery and owner candidate-review evidence.

- [ ] Update G18 disposition only after both maintained generation adapters actually share the factory. Document pure functions, schedule differences, generated identities, experimental status and limits on old candidate timings. Keep deferred solver/fusion variants quarantined.
- [ ] Set compatible version0.2.4, changelog fragment and release qualification with exact source/compiler identities; refresh registry metadata without dropping existing measured rows.
- [ ] Full CPU, generation, compiler/reflection, relevant real GPU, strict package-byte and fresh installation checks. One fresh whole-branch review, then one TDD fix pass. Exact-head hosted CI before merge/release.
- [ ] Publish immutable attested release, independently verify downloads, deliver seven update PRs and inspect consumer checks. No downstream auto-merge, no full-consolidation completion claim.

Self-review: this is an implementation breakdown of the already approved consolidation work; no new consumer behavior or default selection is authorized by this plan. Both maintained export owners and their independent references are explicit. Full-program requirements outside G18 remain tracked by the original plan.
