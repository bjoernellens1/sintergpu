import torch

from vkl.kernels import select_compact


def test_select_compact_preserves_order_and_capacity():
    values = torch.tensor([[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]])
    mask = torch.tensor([[False, True, True, True], [True, False, True, False]])
    actions = torch.tensor([[0, 4, 5, 6], [7, 0, 8, 0]], dtype=torch.int32)

    out = select_compact(values, mask, action_codes=actions, capacity=2)

    assert torch.equal(out["count"], torch.tensor([2, 2]))
    assert torch.equal(out["valid"], torch.tensor([[True, True], [True, True]]))
    assert torch.equal(out["indices"], torch.tensor([[1, 2], [0, 2]]))
    assert torch.equal(out["action_codes"], torch.tensor([[4, 5], [7, 8]], dtype=torch.int32))
    assert torch.equal(out["values"][..., 0], torch.tensor([[11.0, 12.0], [20.0, 22.0]]))
