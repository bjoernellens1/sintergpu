# Local qualification: 2026-10-05

Backend: AMD Radeon 8060S Graphics (RADV STRIX_HALO), Vulkan through
SlangPy/slang-rhi 0.43.1; SymForce 0.12.0; Python 3.13.12. Standalone
compiler: Slang 2026.19; SPIR-V validator: SPIRV-Tools 2026.1.

- Actual Caspar/DABSEG-generated scalar residual and six-column symbolic Jacobian
  dispatch and match independent float64 NumPy normal equations and update.
- Finite-difference derivatives match the independent analytic Jacobian.
- Workgroup and subgroup variants pass repeated persistent-buffer solves, including
  one-element problems and partial workgroups at sizes 65 and 4099.
- Original workgroup numerical cases: 1, 63, 64, 65, 257, 4099 correspondences.
- Singular undamped system reports status 1 and finite zero update.
- Indexed V3 gather including duplicates, shared indices, unique indexed stores/adds
  all execute with Caspar's actual padded vector layout.
- Residual/Jacobian, both first-stage reductions, hierarchical partial reduction,
  and solve shaders compile to SPIR-V and pass `spirv-val`.
- Lint and distribution build pass; the wheel contains `shaders/gn.slang`.
- Local validation passes; downstream RGB-D development proceeds.

Run:

```bash
uv sync --frozen --extra dev --extra gpu
export PATH=/path/to/slang/bin:/path/to/spirv-tools/bin:$PATH
CASPAR_SLANG_GPU_TESTS=1 uv run pytest -q
uv run python scripts/validate.py
```

Benchmark evidence and exact timing boundaries are in [PERFORMANCE_REVIEW.md](PERFORMANCE_REVIEW.md)
and [benchmark-amd-20261005.json](benchmark-amd-20261005.json).
These are fixed-input dense backend measurements, not nonlinear RGB-D tracking.

Pending: NVIDIA Vulkan versus Caspar CUDA correctness/performance qualification,
Intel hardware coverage, robust losses, iterative pose update, image association,
and the downstream tracker/map. GitHub CI's Mesa Vulkan job is software-driver
coverage and must not be presented as NVIDIA/AMD/Intel hardware qualification.
