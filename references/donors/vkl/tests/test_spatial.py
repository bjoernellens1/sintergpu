import torch

from vkl.kernels import spatial_bin_reduce


def test_spatial_bin_reduce_counts_sums_and_feature_means():
    points = torch.tensor([[[0.1, 0.1, 0.1], [0.8, 0.2, 0.2], [1.2, 0.1, 0.1], [9.0, 9.0, 9.0]]])
    valid = torch.tensor([[True, True, True, False]])
    features = torch.tensor([[[1.0], [3.0], [9.0], [0.0]]])

    out = spatial_bin_reduce(points, valid, voxel_size=1.0, origin=torch.zeros(3), features=features, capacity=3)

    assert torch.equal(out["count"], torch.tensor([2]))
    assert torch.equal(out["cell_ids"][0, :2], torch.tensor([[0, 0, 0], [1, 0, 0]]))
    assert torch.equal(out["point_count"][0, :2], torch.tensor([2, 1]))
    assert torch.allclose(out["point_sum"][0, :2], torch.tensor([[0.9, 0.3, 0.3], [1.2, 0.1, 0.1]]))
    assert torch.allclose(out["feature_mean"][0, :2, 0], torch.tensor([2.0, 9.0]))
