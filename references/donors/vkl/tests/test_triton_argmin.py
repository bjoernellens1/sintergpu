import torch
import pytest

from vkl.kernels import select_best_inverse_depth


def test_select_best_inverse_depth_matches_two_smallest_costs():
    costs = torch.tensor([[[3.0, 1.0, 2.0], [4.0, 5.0, 0.5]]])
    inverse_depths = torch.tensor([0.1, 0.2, 0.4])

    out = select_best_inverse_depth(costs, inverse_depths)

    assert torch.equal(out["best_index"], torch.tensor([[1, 2]]))
    assert torch.equal(out["inverse_depth"], torch.tensor([[0.2, 0.4]]))
    assert torch.equal(out["best_cost"], torch.tensor([[1.0, 0.5]]))
    assert torch.equal(out["second_cost"], torch.tensor([[2.0, 4.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_triton_select_matches_torch_on_gpu():
    cost = torch.rand(2, 17, 64, device="cuda")
    inverse_depths = torch.linspace(0.1, 2.0, 64, device="cuda")

    out = select_best_inverse_depth(cost, inverse_depths)
    expected, indices = cost.topk(2, dim=-1, largest=False)

    assert torch.equal(out["best_index"], indices[..., 0])
    assert torch.allclose(out["best_cost"], expected[..., 0])
    assert torch.allclose(out["second_cost"], expected[..., 1])
    assert torch.allclose(out["inverse_depth"], inverse_depths[indices[..., 0]])
