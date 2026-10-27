import torch

from vkl.kernels import residual_evidence


def test_residual_evidence_returns_rgb_depth_frame_and_tile_residuals():
    observed_rgb = torch.zeros(1, 3, 4, 4)
    predicted_rgb = observed_rgb.clone()
    predicted_rgb[:, :, :2, :2] = 0.5
    observed_depth = torch.ones(1, 1, 4, 4)
    predicted_depth = observed_depth.clone()
    predicted_depth[:, :, 2:, 2:] = 3.0
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    out = residual_evidence(observed_rgb, predicted_rgb, observed_depth=observed_depth, predicted_depth=predicted_depth, valid=valid, tile_rows=2, tile_cols=2)

    assert torch.equal(out["rgb_residual"][0, 0, :2, :2], torch.full((2, 2), 0.5))
    assert torch.equal(out["depth_residual"][0, 0, 2:, 2:], torch.full((2, 2), 2.0))
    assert torch.allclose(out["tile_residual"], torch.tensor([[[0.25, 0.0], [0.0, 1.0]]]))
    assert torch.allclose(out["frame_residual"], torch.tensor([0.3125]))
