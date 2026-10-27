from caspar_slang.demo import demo_kernel
from caspar_slang.emitter import emit_slang


def test_demo_emits_compute_shader():
    src = emit_slang(demo_kernel())
    assert '[shader("compute")]' in src
    assert "[numthreads(256, 1, 1)]" in src
    assert "StructuredBuffer<float> x;" in src
    assert "RWStructuredBuffer<float> y;" in src
    assert "fma(r0, r1, r2)" in src
    assert "params.problem_size" in src
