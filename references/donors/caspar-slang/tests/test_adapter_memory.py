import pytest

from caspar_slang.caspar_adapter import _access_kind, _lower_memory_line


def test_read_idx_lowering_uses_chunk_width_and_padding():
    line = "ReadIdx3<256, float, float, float4>(points, 2*points_num_alloc, global_thread_idx, r0, r1, r2);"
    assert _lower_memory_line(line) == [
        f"r{j} = points[(2u * params.points_num_alloc + global_thread_idx * 4u + {j}u)];"
        for j in range(3)
    ]


def test_write_idx_lowering():
    line = "WriteIdx2<256, float, float, float2>(out, 0*out_num_alloc, global_thread_idx, r0, r1);"
    assert _lower_memory_line(line) == [
        f"out[(0u * params.out_num_alloc + global_thread_idx * 2u + {j}u)] = r{j};"
        for j in range(2)
    ]


def test_indexed_add_is_read_modify_write():
    from caspar_slang.ir import AccessKind

    accessor = type("AddIndexed", (), {})()
    assert _access_kind(accessor) == AccessKind.READ_WRITE
    assert _lower_memory_line("AddIdx1<64,float,float,float>(out,0*out_num_alloc,out_idx,r0);") == [
        "out[(0u * params.out_num_alloc + out_idx * 1u + 0u)] += r0;"
    ]


def test_rejects_accessor_dimension_mismatch():
    with pytest.raises(ValueError):
        _lower_memory_line("ReadIdx3<64,float,float,float4>(p,0*p_num_alloc,p_idx,r0,r1);")
