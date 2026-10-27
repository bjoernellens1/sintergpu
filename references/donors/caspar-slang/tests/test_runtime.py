import importlib.util
import os

import numpy as np
import pytest

from caspar_slang.ir import AccessKind, Buffer, KernelIR
from caspar_slang.runtime import VulkanRuntime


def host_runtime():
    # Validation tests must fail before any device interaction.
    return VulkanRuntime.__new__(VulkanRuntime)


def scatter_ir():
    return KernelIR(
        "scatter",
        [
            Buffer(
                "out", AccessKind.WRITE, components=3, index_name="out_indices", unique_indices=True
            ),
            Buffer("out_indices", AccessKind.READ, num_alloc_param=False, scalar_type="uint"),
        ],
        [],
        [],
    )


@pytest.mark.parametrize(
    "indices,message",
    [
        (np.array([1, 1], np.uint32), "unique"),
        (np.array([0, 3], np.uint32), "out-of-bounds"),
        (np.array([0], np.uint32), "fewer entries"),
    ],
)
def test_host_scatter_rejects_invalid_indices(indices, message):
    with pytest.raises(ValueError, match=message):
        host_runtime().dispatch_arrays(
            scatter_ir(),
            {
                "out": np.zeros((3, 3), np.float32),
                "out_indices": indices,
            },
            problem_size=2,
        )


def test_host_rejects_wrong_component_count():
    with pytest.raises(ValueError, match="shape"):
        host_runtime().dispatch_arrays(
            scatter_ir(),
            {
                "out": np.zeros((3, 2), np.float32),
                "out_indices": np.array([0, 1], np.uint32),
            },
        )


def test_empty_dispatch_is_noop():
    arrays = {"out": np.zeros((0, 3), np.float32), "out_indices": np.array([], np.uint32)}
    result = host_runtime().dispatch_arrays(scatter_ir(), arrays)
    assert result["out"].shape == (0, 3)


@pytest.mark.skipif(
    os.environ.get("CASPAR_SLANG_GPU_TESTS") != "1" or importlib.util.find_spec("symforce") is None,
    reason="requires opt-in Vulkan and SymForce",
)
@pytest.mark.parametrize("mode", ["WriteIndexed", "AddIndexed"])
def test_real_indexed_scatter_and_shared_indices(mode):
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    from caspar_slang.caspar_adapter import from_caspar_kernel

    lib = CasparLibrary()

    @lib.add_kernel
    def scatter(p: T.Annotated[sf.V3, mem.ReadIndexed]) -> T.Annotated[sf.V3, mem.WriteSequential]:
        return p * 2

    kernel = lib.kernels[0]
    original = kernel.outputs[0]
    output = getattr(mem, mode)(
        "out",
        original.storage,
        dtype=original.storage_t,
        kernel_dtype=original.kernel_t,
        reuse_indices_from="p",
    )
    kernel.outputs = [output]
    kernel.accessors = [*kernel.inputs, output]
    ir = from_caspar_kernel(kernel)
    assert len([b for b in ir.buffers if b.scalar_type == "uint"]) == 1
    p = np.arange(15, dtype=np.float32).reshape(5, 3)
    indices = np.array([4, 0, 2], np.uint32)
    initial = np.ones((5, 3), np.float32)
    actual = VulkanRuntime().dispatch_arrays(
        ir, {"p": p, "out": initial, "p_indices": indices}, problem_size=3
    )["out"]
    expected = initial.copy()
    expected[indices] = p[indices] * 2 + (1 if mode == "AddIndexed" else 0)
    np.testing.assert_array_equal(actual, expected)
