# Shared point-to-plane math

`optimization.point_plane` is an importable, resource-independent Slang module.
It computes `n·(p-q)` and a rotation-first left-action Jacobian `(p×n,n)`.
All three vectors must share the caller's coordinate frame; units and normal
normalization belong to the caller. This module owns no buffers, allocation,
command recording, submission, synchronization or readback.

```python
from sintergpu import module_path
source = module_path('optimization.point_plane')
# Add source.parent.parent to the caller's existing Slang session include paths.
```

```slang
import optimization.point_plane;
float r = sinterPointPlaneResidual<true>(p, q, n);
float3 rotation = sinterPointPlaneRotation(p, n);
// Translation Jacobian is n; store it using the caller's chosen layout.
```

CMake installs the same file under `share/sintergpu/kernels/optimization/`;
`sintergpu::kernels` supplies the source directory. See
[`examples/point_plane_import.slang`](../examples/point_plane_import.slang) for
an illustrative256-thread caller. That example has its own buffers and is not
the ABI-qualified Caspar native adapter.

## Explicit schedules

The `AoSoASchedule=true` residual preserves the generated indexed Caspar
vector export's z/x/y FMA schedule. `false` preserves the scalar export's z/y/x
schedule. They represent the same real-valued algebra but need not produce the
same floating-point bits. Rotation retains the original FMA/product ordering.
`sinterPointPlaneScalars<AoSoASchedule>` returns residual and six separate
rotation-first Jacobian scalars through output parameters. There is no implicit
normal normalization, validity filtering or finite-value sanitization.

Functions do not impose a buffer ABI. The qualification adapters preserve the
original padded4+2 AoSoA Jacobian, indexed points, sequential targets/normals,
capacity guards and untouched tails. Existing dispatch shaders/default selections
remain unchanged. Standalone Slang2026.12 and2026.19 import checks compare the
complete normalized executable stream and reflected resources:1024-thread
profiles match;256-thread profiles permit only the actual LocalSize difference.
This is compiler evidence, not a fresh timing or full-domain device claim.
Physical GPU parity and source-current baseline recollection remain pending in
this working batch until their retained evidence is recorded.

## One generation owner

`sintergpu.codegen.point_plane` is an optional symbolic factory. Importing it
requires no NumPy, Torch, SlangPy or SymForce runtime. Callers pass their own
`sf` symbolic module to `point_plane_expression(sf,p,q,n)` or its nine-scalar
adapter `point_plane_scalars`. Both evaluate an infinitesimal left rigid action
at zero twist and return the rotation-first derivative.

`tools/codegen/caspar_point_plane.py` owns the vector and scalar Caspar generation
adapters; both consume this factory. Remaining dense/photometric expressions
retain their donor algebra. Frozen donor generators are independent references,
not a second maintained library owner. `python tools/regenerate_math.py --check`
requires optional SymForce0.12.0 and reproduces all five released generated math
modules byte-for-byte. Consumers of released shaders need none of those generation
dependencies. Provenance records the source and generation input identities.

## Qualification boundaries

Candidate PR14 at2def63bb85adfcb0cb5b1b8d0910bed31dda0539 has a controlled
passing native256 configuration: exact seeded AoSoA1025, Slang2026.19,
retained Vulkan resources/cached staging and matched polling completion, three
10000-block processes per Radeon8060S/canonical RTX4070Ti. Owner evidence replay
independently checks198 archived native processes and exactly reproduces18
boundary decisions from the qualifying six processes. This does not qualify
other inputs, size ranges, Python callers, example buffers or changed import
compilation for timing. No optimized default or consumer migration follows.

Other candidate solver and fusion families remain quarantined with their failed
or incomplete no-regression gates. Tracking poses, trajectories, convergence and
application accuracy remain separate acceptance requirements.
