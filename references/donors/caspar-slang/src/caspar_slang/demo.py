from __future__ import annotations

from .ir import AccessKind, Buffer, KernelIR


def demo_kernel() -> KernelIR:
    # y = fma(x, a, b), represented in Caspar's SoA-compatible flat layout.
    return KernelIR(
        name="demo_fma",
        buffers=[
            Buffer("x", AccessKind.READ),
            Buffer("a", AccessKind.READ),
            Buffer("b", AccessKind.READ),
            Buffer("y", AccessKind.WRITE),
        ],
        registers=["r0", "r1", "r2"],
        code_lines=[
            "r0 = x[(0u * params.x_num_alloc + global_thread_idx)];",
            "r1 = a[(0u * params.a_num_alloc + global_thread_idx)];",
            "r2 = b[(0u * params.b_num_alloc + global_thread_idx)];",
            "r0 = fmaf(r0, r1, r2);",
            "y[(0u * params.y_num_alloc + global_thread_idx)] = r0;",
        ],
        block_size=256,
        metadata={"source": "built-in demo"},
    )
