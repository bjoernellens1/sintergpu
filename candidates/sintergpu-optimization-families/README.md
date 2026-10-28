# Optimization family candidates

Unpromoted source-only candidates for G18, the small Cholesky subset of G21,
and a G18/G11 fused GN producer. Catalog, installed APIs, defaults, consumers and
release versions are unchanged. All incumbents remain available. See
[the recoverable ledger](LEDGER.md) for the ranking of all 22 audited groups,
predeclared gates, rejected iterations and GPU coordination.

## Reproduce

Run these commands from this directory in a separate checkout of the recorded
upstream base. Optional dependencies: Python >=3.11, NumPy >=2, pytest >=8,
SlangPy **0.43.1**, and SymForce for the symbolic factory test. SymForce is passed
by the caller; importing the candidate factory initializes no runtime.

```sh
python -m pytest tests -q
SINTERGPU_GPU_TESTS=1 python -m pytest tests -q
# Selected canonical NVIDIA machine only; pin its NVIDIA Vulkan ICD explicitly.
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  SINTERGPU_GPU_TESTS=1 SINTERGPU_TARGET=rtx4070 python -m pytest tests -q

python benchmarks/compile_families.py \
  --toolchains /path/containing/2026.12/and/2026.19 --output /tmp/family-compilers

# Own/exclusively coordinate the physical GPU before measuring. One process/run.
SINTERGPU_GPU_TESTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/family_timings.py --target radeon8060s --output /tmp/run-1.json
# Repeat in two further fresh processes, retaining each output.
# For the verified canonical RTX4070Ti use --target rtx4070 and its ICD above.
```

The benchmark lock is `/tmp/sintergpu-gpu.lock`; PID/host/start/output evidence is
`/tmp/sintergpu-gpu-owner.json`. This advisory lock does not exclude desktop GPU
activity or an agent ignoring it. Coordinate other measurements explicitly;
record foreign GPU users and retain noisy/failing outcomes. The parent released
both controlled-collection slots before candidate measurements on those devices.

The pinned archives are the existing repository CI pins: Slang 2026.12 archive
SHA256 `2409ecc7b6710783e05fd59792558c275c637e3e3c37fb5054fcfc858e570d82` and
2026.19 `bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e`.
The standalone script compiles every baseline/candidate profile plus a real module
import and applies the library's reflection schema allowlist. It does not edit the
library's compilation records or catalog. SlangPy's actual compiler is separately
hashed in every timing JSON; standalone compile success is not device execution.

## Import and ownership

`kernels/optimization/*.slang` contains reusable functions. Add `kernels/` to
the Slang module search path; see [the import example](examples/import_helpers.slang).
For a caller-owned C++/Vulkan context, compile the same sources to SPIR-V and use
their recorded reflection; no Python runtime is required by the shaders.

```sh
slangc examples/import_helpers.slang -I kernels -entry main -target spirv \
  -profile spirv_1_5 -fp-mode precise -O3 -o /tmp/helpers.spv \
  -reflection-json /tmp/helpers.reflection.json
```

Functions create no device, buffer, descriptor, command encoder, submission,
completion wait, transfer or host allocation. The host owns all resources and
lifetimes. Cholesky's fixed local arrays are shader invocation storage. The fused
dispatch uses the incumbent 43x64 float groupshared storage (11,008 bytes) and the
same output/scratch allocations. Its only pipeline changes are one fewer dispatch
and eliminating residual/Jacobian reloads in the first reduction. All exported
residuals, Jacobians, partials, final terms, delta and status remain materialized.
The incumbent higher reductions and solve are used unchanged in the fused test.
Raw Vulkan callers must provide storage-write/read dependencies between passes;
the harness records into one explicit encoder and submits/waits outside shaders.

## Numerical contracts

| Family/profile | Preserved contract |
| --- | --- |
| Point-plane `sinterPointPlane<true>` | FP32 AoSoA generated FMA schedule: initial z product, x then y accumulation; indexed points, sequential targets/normals in the adapter |
| Point-plane `sinterPointPlane<false>` | Same symbolic residual and Jacobian; scalar generated FMA schedule: initial z product, y then x accumulation |
| Cholesky core, capacity6 | Caspar lower-triangle row-major H then b; fixed dimension6; rotation-first pose convention belongs to caller |
| Cholesky core, capacity18 | SLAM runtime dimension6/12/18, local L stride18; same matrix/b layout and original observation-count guard in adapter |
| Checked Cholesky helper | Dimension guard and zero output on failure; convenience profile is correctness-tested, not performance-qualified |
| Fused workgroup/subgroup | Original 64-lane stride-halving tree, optionally original WaveActiveSum then tree; profiles are not interchangeable |

Point-plane evaluates `n dot (p + omega cross p + translation - q)` at zero.
The six Jacobian components are `cross(p,n), n`, rotation first, left action.
Normals are held constant and need not be normalized. No finite filtering,
robust weighting, coordinate/frame conversion, scaling or invalid-value policy is
added. Explicit FMA order is observable, including cancellation, signed zero,
NaN/Inf and overflow. Independent finite differences check the symbolic factory;
GPU Jacobians also match independent float64 analytic results. No general
autograd/backward or differentiable-solver API is claimed.

The point adapter retains padded AoSoA V6: first four components at `4*i+c`, last
two at `4*allocation+2*i+c`. Input xyz uses four-float rows. Indices are uint32;
duplicate gathers are valid. All sizes, index bounds and storage capacities are
caller-validated, as in the donor. The fused GN adapter uses allocation=count,
preserving the incumbent GN Jacobian layout. Out-of-range lanes stay active
through every reduction barrier and contribute zero. Poisoned padding is untouched.
Empty correspondence sets are outside the incumbent host contract; tested counts
are explicit in the evidence, not a claim that every uint32 size is supported.

Cholesky solves `(lower(H)+lower(H).T-diag(H)+damping*I)x=-b`. It reads only the
lower triangle, including when the unused upper triangle is NaN. Damping is a
caller-validated finite nonnegative FP32 value. The left-looking i/j/k loops,
forward/backward substitution, nonpositive/nonfinite pivot checks and original
global failure flag are unchanged. Incumbent adapters emit status0/1 and exact
zero failed updates. The core requires `1 <= dimension <= Capacity`, sufficient
terms and prevalidated offsets; it returns validity and does not promise usable
x on failure. The checked wrapper rejects out-of-capacity dimensions before
reading terms and clears its entire returned array on failure. Observation gates
stay in adapters; VKL eigenvalue/condition policy, Schur assembly, PCG, rollback,
and Adam are not replaced. No sorting, atomics, count truncation or new uint-wrap
policy is introduced. Shader index products must fit uint32 and device dispatch
and storage limits; this is a caller precondition, not an untested capacity claim.

## References, baselines and performance

[candidate.json](candidate.json) freezes upstream/donor revisions, baseline hashes,
notices, requirements and evidence. Frozen donor implementations are unchanged.
The symbolic factory has vector and scalar adapters; both generated device
schedules are explicit because one symbolic expression does not imply one FP32
schedule. Float64 NumPy cross/dot, matrix multiplication and `linalg.solve` are
independent numerical oracles; they do not call candidate device math. Exact
full-output differential tests additionally compare original generated shaders and
solver loops, including tail lanes, padding, failures and observation guards.

The matched incumbents are packaged Caspar `computeMain`/`caspar_point_plane`,
`gn.slang/solveSystem`, and SLAM `dense_ba.slang/solveSchur`. Full GN comparisons
use both existing workgroup and subgroup reductions and the same higher reduction
and solve. They retain their different exact reduction trees; neither slower-tree
results nor CPU reference speedups qualify the other profile. The dense RGB-D
tracker has different inputs, association and weighting and is not a substitute
baseline for this prepared-correspondence GN operation.

Each final configuration has three independent processes, 10 warmups and 100
ABBA blocks, with every sample retained. CPU reference wall, GPU isolated family
timestamps (or fused producer and full GN timestamps), complete resident host
wall, and complete uploads/dispatch/full readback costs are distinct. Buffer
allocation and compilation are outside timing and unchanged between variants.
Full correctness is checked before and after all measured boundaries. No pose
tracking, trajectory, application throughput or consumer adoption follows.

Historical rejected iterations and their recipes are retained. Final per-device,
per-family gate tables and exact pending/retained decisions are in
[ACCEPTANCE.md](ACCEPTANCE.md). A failed, noisy, missing or untested configuration
keeps its incumbent. No positive aggregate cancels an individual regression.

Raw timing JSON is retained as deterministic, lossless `.json.gz` files to keep
the source diff reviewable. `evidence/compressed-samples.json` records compressed
and original byte hashes. `gzip -dc file.json.gz` restores the exact original JSON;
the report script reads either form. No sample or rejected iteration was dropped.
