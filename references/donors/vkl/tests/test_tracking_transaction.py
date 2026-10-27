import pytest
import torch

from vkl.kernels import compose_tracking_solve


def _partials(batch: int, partials: int, *, bias: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(6).reshape(1, 1, 6, 6).expand(batch, partials, -1, -1).clone()
    vector = torch.full((batch, partials, 6), bias)
    valid = torch.ones((batch, partials), dtype=torch.bool)
    return identity, vector, valid


@pytest.mark.parametrize("batch", [1, 8])
def test_composed_tracking_solve_reduces_dense_and_sparse_partials_together(batch: int):
    dense_H, dense_b, dense_valid = _partials(batch, 4, bias=1.0)
    sparse_H, sparse_b, sparse_valid = _partials(batch, 2, bias=2.0)

    out = compose_tracking_solve(
        dense_partial_H=dense_H,
        dense_partial_b=dense_b,
        dense_partial_valid=dense_valid,
        sparse_partial_H=sparse_H,
        sparse_partial_b=sparse_b,
        sparse_partial_valid=sparse_valid,
    )

    assert out["H"].shape == (batch, 6, 6)
    assert out["b"].shape == (batch, 6)
    assert torch.allclose(out["H"], dense_H.sum(1) + sparse_H.sum(1))
    assert torch.allclose(out["b"], dense_b.sum(1) + sparse_b.sum(1))
    assert torch.equal(out["dense_partial_count"], torch.full((batch,), 4, dtype=torch.int64))
    assert torch.equal(out["sparse_partial_count"], torch.full((batch,), 2, dtype=torch.int64))
    assert out["valid"].all()


@pytest.mark.parametrize("batch", [1, 8])
def test_composed_tracking_solve_keeps_invalid_sparse_partials_out_of_the_system(batch: int):
    dense_H, dense_b, dense_valid = _partials(batch, 1, bias=1.0)
    sparse_H, sparse_b, sparse_valid = _partials(batch, 2, bias=100.0)
    sparse_valid.zero_()

    out = compose_tracking_solve(
        dense_partial_H=dense_H,
        dense_partial_b=dense_b,
        dense_partial_valid=dense_valid,
        sparse_partial_H=sparse_H,
        sparse_partial_b=sparse_b,
        sparse_partial_valid=sparse_valid,
    )

    assert torch.allclose(out["H"], dense_H.sum(1))
    assert torch.allclose(out["b"], dense_b.sum(1))
    assert torch.equal(out["sparse_partial_count"], torch.zeros(batch, dtype=torch.int64))
