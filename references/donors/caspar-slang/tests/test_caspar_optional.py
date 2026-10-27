import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="symforce not installed")
def test_real_caspar_sequential_kernel_can_lower():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    from caspar_slang.caspar_adapter import from_caspar_kernel
    from caspar_slang.emitter import emit_slang

    caslib = CasparLibrary(name="test")

    @caslib.add_kernel
    def axpy(
        x: T.Annotated[sf.Symbol, mem.ReadSequential],
        a: T.Annotated[sf.Symbol, mem.ReadSequential],
        b: T.Annotated[sf.Symbol, mem.ReadSequential],
    ) -> T.Annotated[sf.Symbol, mem.WriteSequential]:
        return x * a + b

    ir = from_caspar_kernel(caslib.kernels[0])
    source = emit_slang(ir)
    assert "RWStructuredBuffer<float> out" in source or "RWStructuredBuffer<float> y" in source
    assert "fma(" in source or "*" in source
