import numpy as np
import pytest


def test_caspar_chunk_padding_roundtrip():
    from caspar_slang.layout import pack, unpack

    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    packed = pack(data)
    assert packed.size == 40
    np.testing.assert_array_equal(packed[:8], [0, 1, 2, 3, 7, 8, 9, 10])
    np.testing.assert_array_equal(packed[20:28], [4, 5, 6, 0, 11, 12, 13, 0])
    np.testing.assert_array_equal(unpack(packed, 5, 7), data)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7, 11, 16])
def test_pack_all_chunk_widths(d):
    from caspar_slang.layout import pack, unpack

    data = np.arange(3 * d, dtype=np.float32).reshape(3, d)
    np.testing.assert_array_equal(unpack(pack(data), 3, d), data)


def test_custom_chunk_order_roundtrip():
    from caspar_slang.layout import pack, unpack

    data = np.arange(15, dtype=np.float32).reshape(3, 5)
    layout = ((3, 1, 4), (2, 0))
    np.testing.assert_array_equal(unpack(pack(data, layout), 3, 5, layout), data)


def test_layout_rejects_wrong_storage_dtype_and_size():
    from caspar_slang.layout import pack, unpack

    with pytest.raises(TypeError):
        pack(np.zeros((3, 3), np.float64))
    with pytest.raises(ValueError):
        unpack(np.zeros(9, np.float32), 3, 3)
