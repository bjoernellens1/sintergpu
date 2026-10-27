import pytest
import torch

from vkl.kernels import REGISTRY
from vkl.kernels.reference import resolution_pyramid


def test_resolution_pyramid_returns_native_and_bounded_rgb_levels():
    rgb = torch.arange(1 * 3 * 5 * 9, dtype=torch.float32).reshape(1, 3, 5, 9)

    levels = resolution_pyramid(rgb=rgb)

    assert set(levels) == {"native", "half", "quarter", "eighth"}
    assert levels["native"] is rgb
    assert levels["half"].shape == (1, 3, 3, 5)
    assert levels["quarter"].shape == (1, 3, 2, 3)
    assert levels["eighth"].shape == (1, 3, 1, 2)
    assert all(level.dtype == rgb.dtype and level.device == rgb.device for level in levels.values())


def test_resolution_pyramid_rejects_non_rgb_float_batches():
    with pytest.raises(ValueError, match=r"shape \[B, 3, H, W\]"):
        resolution_pyramid(rgb=torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match=r"shape \[B, 3, H, W\]"):
        resolution_pyramid(rgb=torch.zeros(1, 1, 8, 8))
    with pytest.raises(TypeError, match="floating point"):
        resolution_pyramid(rgb=torch.zeros(1, 3, 8, 8, dtype=torch.int64))


def test_resolution_pyramid_rejects_empty_spatial_dimensions():
    with pytest.raises(ValueError, match="positive spatial dimensions"):
        resolution_pyramid(rgb=torch.zeros(1, 3, 0, 8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resolution_pyramid_cpu_low_precision_preserves_dtype(dtype):
    levels = resolution_pyramid(rgb=torch.arange(1 * 3 * 5 * 9, dtype=dtype).reshape(1, 3, 5, 9))

    assert levels["half"].dtype is dtype
    assert levels["half"].shape == (1, 3, 3, 5)


def test_resolution_pyramid_is_public_and_antialiased():
    rgb = torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0]]]]).repeat(1, 3, 1, 1)

    assert REGISTRY["resolution_pyramid"] is resolution_pyramid
    assert 0.0 < resolution_pyramid(rgb=rgb)["half"][0, 0, 1, 1] < 1.0
