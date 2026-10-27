import pytest
import torch

from vkl.kernels import deterministic_reduce_solve


@pytest.mark.parametrize("batch", [1, 8])
def test_deterministic_reduce_solve_reduces_fixed_partials_and_solves(batch: int):
    h = torch.eye(6).expand(batch, 3, -1, -1).clone()
    b = torch.zeros(batch, 3, 6)
    b[..., 0] = -torch.tensor([0.25, 0.5, 0.25])
    valid = torch.tensor([[True, True, True]]).expand(batch, -1)

    out = deterministic_reduce_solve(h, b, valid, damping=0.0)

    assert out["H"].shape == (batch, 6, 6)
    assert out["b"].shape == (batch, 6)
    assert torch.allclose(out["H"], 3 * torch.eye(6).expand(batch, -1, -1))
    assert torch.allclose(out["b"][:, 0], torch.full((batch,), -1.0))
    assert torch.allclose(out["delta"][:, 0], torch.full((batch,), 1 / 3))
    assert out["valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_deterministic_reduce_solve_rejects_degenerate_systems(batch: int):
    h = torch.zeros(batch, 2, 6, 6)
    b = torch.ones(batch, 2, 6)
    valid = torch.ones(batch, 2, dtype=torch.bool)

    out = deterministic_reduce_solve(h, b, valid, damping=0.0)

    assert not out["valid"].any()
    assert torch.equal(out["delta"], torch.zeros_like(out["delta"]))
    assert torch.isinf(out["condition_proxy"]).all()
