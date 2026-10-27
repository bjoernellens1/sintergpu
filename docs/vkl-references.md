# Reusable VKL Torch references

`sintergpu.references.vkl` packages all **47 substantive VKL reference modules**
plus their initializer from revision
`3384023a311651ea4260cf12ddab1567972f89f3`. They are executable Python modules
included in the wheel, with the original public exports and sibling import paths.
Install the optional Torch dependency through `sintergpu[torch]` or provide a
compatible PyTorch installation for your device. Base `import sintergpu` does not
import Torch or create a runtime.

```python
import torch
from sintergpu.references.vkl import depth_convert, depth_geometry, photometric_isp

# The application chooses and owns the device. Nothing selects one implicitly.
device = torch.device("cpu")
depth = torch.ones((1, 1, 8, 12), dtype=torch.float32, device=device)
k = torch.tensor([[[10., 0., 5.5], [0., 10., 3.5], [0., 0., 1.]]], device=device)
canonical = depth_convert(values=depth, representation="DEPTH", scale=1.)
geometry = depth_geometry(canonical["depth"], k)
rgb = torch.full((1, 3, 8, 12), .25, device=device, requires_grad=True)
observed = photometric_isp(rgb, evidence=False)["rgb_observed"]
observed.sum().backward()
```

These are GPU-capable **Torch reference variants**, distinct from the four partial
forward Slang profiles in [vkl-ports.md](vkl-ports.md). They preserve the donor's
validation, dtype, invalid-value handling, layout, numerical operation order,
capacity, ordering, autograd graph, device and optional output semantics. The
implementations are unchanged except for explicit owner-authorized Apache-2.0
headers. No device-selection layer, hidden tensor copy, NumPy conversion, detach,
queue submission or alternative kernel dispatch was added. Existing donor
validation can synchronize when checking Torch tensor values; these functions
have the same behavior as the original references and are not a new asynchronous
execution guarantee. Function-specific gradient limitations remain those of the
donor; testing selected differentiable paths does not establish gradients for
all 47 modules or for the separate Slang kernels.

The initializer exposes the exact original `__all__`, including depth conversion,
fusion, evidence, pyramid and geometry; camera registration; temporal RGB/depth
restoration and reprojection; photometric ISP; stable compaction; keypoints;
pose and tracking helpers. Additional original modules not exported at package
level remain importable by module, for example:

```python
from sintergpu.references.vkl import temporal_restore_rgbd, select_compact
from sintergpu.references.vkl.sparse_geometry import project_landmarks
```

All dependencies trace to Python's standard library, Torch (including
`torch.nn.functional`) and sibling modules packaged here. There are no missing
application-store dependencies or implicit imports from a local VKL checkout.
Torch remains optional: if absent, public Torch entry points raise a clear
`ModuleNotFoundError` naming `sintergpu[torch]`; the independent shader NumPy
oracles remain importable as `sintergpu.references.vkl.numpy`.

The initial NumPy-only `sintergpu.references.vkl` module has moved to
`sintergpu.references.vkl.numpy`. Existing fixture users should update their
import. This avoids presenting the NumPy shader oracles as the original Torch
public API. `mask_u32` and `depth_pyramid_level` are NumPy fixture helpers; they
are not donor Torch exports or additional Slang ports.

`provenance/vkl-reference-package.json` binds every original path and hash to its
wheel package destination and hash. The package initializer adds a narrowly
scoped missing-Torch fallback; every substantive module has an AST identical to
its frozen donor. Third-party dependency licensing remains unchanged.

Verification commands:

```
PYTHONPATH=src python -m pytest tests/test_vkl_reference_package.py -q
PYTHONPATH=src /home/bjoern/git/caspar-slang/.venv/bin/python -m pytest tests/test_vkl_reference_package.py -q
```

System Torch CPU verification: **61 passed**. The read-only SlangPy environment
without Torch: **49 passed, 12 explicit optional-dependency skips**. Tests import
all 47 modules, compare exact public exports and executable ASTs, and perform
strict differential checks against frozen raw source for temporal validity and
trust gates, photometric forward/parameter gradients, depth conversion/status/
invalids and gradients, geometry/pose/normals and gradients, stable compaction/
capacity/action codes and gradients, and original validation errors. Base import
and the missing-Torch NumPy path are checked in fresh subprocesses.

These tests establish packaging integrity and the tested CPU/autograd paths.
They do not constitute new GPU execution, gradient or performance qualification
for all reference functions. Four substantive modules have partial forward Slang
translations; **43 substantive Slang translations remain pending**, together
with the previously documented strict extreme-depth failures and RTX3090 gate.
