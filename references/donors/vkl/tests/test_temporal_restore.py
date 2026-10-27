import torch
import pytest

from vkl.kernels import temporal_restore_rgbd


def test_temporal_restore_uses_only_valid_aligned_context_and_labels_mode():
    rgb = torch.zeros(1, 3, 2, 2)
    context = torch.ones(1, 2, 3, 2, 2)
    valid = torch.tensor([[[[[True, True], [False, False]]], [[[False, False], [True, True]]]]])

    out = temporal_restore_rgbd(rgb, context, context_valid=valid, alignment="rgbd_reprojected")

    assert torch.equal(out["rgb"], torch.full_like(rgb, 0.5))
    assert torch.equal(out["support_count"], torch.ones(1, 1, 2, 2, dtype=torch.int64))
    assert torch.equal(out["alignment_code"], torch.tensor([1]))


def test_temporal_restore_masks_nonfinite_or_invalid_confidence_context_without_nan_leakage():
    rgb = torch.full((1, 3, 2, 2), 0.25)
    context = torch.full((1, 1, 3, 2, 2), float("nan"))
    out = temporal_restore_rgbd(
        rgb,
        context,
        context_confidence=torch.full((1, 1, 1, 2, 2), 2.0),
    )
    assert torch.equal(out["rgb"], rgb)
    assert not out["confidence"].any()
    assert not out["support_count"].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_temporal_restore_fullgraph_compiles_on_gpu():
    rgb = torch.rand(1, 3, 8, 8, device="cuda")
    context = torch.rand(1, 1, 3, 8, 8, device="cuda")
    compiled = torch.compile(temporal_restore_rgbd, fullgraph=True)

    out = compiled(rgb, context, alignment="rgb_aligned")

    assert out["rgb"].shape == rgb.shape
