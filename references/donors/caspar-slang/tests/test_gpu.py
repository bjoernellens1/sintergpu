import importlib.util
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CASPAR_SLANG_GPU_TESTS") != "1", reason="set CASPAR_SLANG_GPU_TESTS=1"
)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 257, 4099])
def test_gpu_gn_matches_independent_reference(n):
    from caspar_slang.gn import GaussNewton, reference, synthetic_problem

    points, targets, normals, indices = synthetic_problem(n, seed=41)
    solver = GaussNewton()
    result = solver.step(points, targets, normals, indices, damping=1e-3)
    expected = reference(points, targets, normals, indices, damping=1e-3)
    for name in ["residual", "jacobian", "hessian", "gradient", "delta"]:
        np.testing.assert_allclose(getattr(result, name), expected[name], atol=3e-5, rtol=3e-4)
    assert result.status == 0


def test_gpu_singular_solve_reports_failure():
    from caspar_slang.gn import GaussNewton

    z = np.zeros((65, 3), np.float32)
    result = GaussNewton().step(z, z, z, np.arange(65, dtype=np.uint32), damping=0)
    assert result.status == 1
    assert np.isfinite(result.delta).all()


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="symforce missing")
def test_indexed_vector_kernel_uses_upstream_layout():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    from caspar_slang.caspar_adapter import from_caspar_kernel
    from caspar_slang.layout import pack, unpack
    from caspar_slang.runtime import VulkanRuntime

    lib = CasparLibrary()

    @lib.add_kernel
    def gather(p: T.Annotated[sf.V3, mem.ReadIndexed]) -> T.Annotated[sf.V3, mem.WriteSequential]:
        return p * 2

    ir = from_caspar_kernel(lib.kernels[0])
    runtime = VulkanRuntime()
    p = np.arange(51, dtype=np.float32).reshape(17, 3)
    indices = np.array([16, 0, 4, 4, 2], dtype=np.uint32)
    out = runtime.buffer(np.zeros(20, np.float32))
    runtime.dispatch(
        ir,
        {"p": runtime.buffer(pack(p)), "p_indices": runtime.buffer(indices), "out": out},
        problem_size=5,
        allocations={"p": 17, "out": 5},
    )
    np.testing.assert_array_equal(unpack(out.to_numpy().view(np.float32), 5, 3), p[indices] * 2)
