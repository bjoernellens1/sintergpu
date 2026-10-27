from caspar_slang.intrinsics import lower_scalar_line


def test_common_intrinsics():
    assert lower_scalar_line("r0 = sinf(r1);") == "r0 = sin(r1);"
    assert lower_scalar_line("r0 = fmaf(r1, r2, r3);") == "r0 = fma(r1, r2, r3);"
    assert lower_scalar_line("r0 = __frcp_rn(r1);") == "r0 = rcp(r1);"


def test_sincos():
    assert lower_scalar_line("sincosf(r0, &r1, &r2);") == "sincos(r0, r1, r2);"
