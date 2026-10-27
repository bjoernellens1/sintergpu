import pytest
import torch

from vkl.bench.photometric import (
    fit_online_exposure_gain,
    fit_online_exposure_gain_response,
    fit_photometric_exposure_gain,
    fit_photometric_stages,
)
from vkl.kernels import photometric_isp


def test_photometric_fit_rejects_mismatched_dtype():
    clean = torch.rand(1, 3, 4, 4, dtype=torch.float32)
    observed = clean.to(torch.float64)

    with pytest.raises(ValueError, match="share device and dtype"):
        fit_photometric_exposure_gain(clean, observed, steps=1)


def test_disabled_optional_stages_preserve_batch_shapes():
    clean = 0.1 + 0.2 * torch.rand(2, 3, 4, 4)
    observed = clean * 1.05

    fitted = fit_photometric_stages(
        clean,
        observed,
        steps=2,
        fit_rgb_gain=False,
        fit_color_matrix=False,
        fit_vignette=False,
        fit_response=False,
    )

    assert fitted["rgb_gain"].shape == (2, 3, 1, 1)
    assert fitted["chroma_rg_bg"].shape == (2, 2, 1, 1)
    assert fitted["color_matrix"].shape == (2, 3, 3)
    assert fitted["vignette"].shape == (2, 3, 3)
    assert fitted["response_gamma"].shape == (2, 1, 1, 1)
    assert torch.equal(fitted["chroma_rg_bg"], torch.zeros_like(fitted["chroma_rg_bg"]))
    assert torch.allclose(fitted["color_matrix"], torch.eye(3).expand(2, -1, -1))


def test_full_stage_fit_uses_exact_two_dof_wb_latent(monkeypatch):
    optimizer_shapes: list[list[tuple[int, ...]]] = []
    real_adam = torch.optim.Adam

    def recording_adam(parameters, *args, **kwargs):
        params = tuple(parameters)
        optimizer_shapes.append([tuple(parameter.shape) for parameter in params])
        return real_adam(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    clean = 0.1 + 0.2 * torch.rand(1, 3, 4, 4)
    observed = clean * torch.tensor([1.05, 1.0, 0.95])[None, :, None, None]

    fitted = fit_photometric_stages(
        clean,
        observed,
        steps=2,
        fit_rgb_gain=True,
        fit_color_matrix=False,
        fit_vignette=False,
        fit_response=False,
    )

    assert optimizer_shapes == [[(1, 1, 1, 1), (1, 2, 1, 1)]]
    assert fitted["chroma_rg_bg"].shape == (1, 2, 1, 1)
    assert torch.allclose(fitted["rgb_gain"].prod(dim=1), torch.ones(1, 1, 1, 1), atol=1e-6)


def test_log_ratio_initializer_ignores_clipped_pixels():
    clean = torch.full((1, 3, 4, 4), 0.9)
    clean[:, :, :, :1] = 0.2
    exposure_ev = torch.full((1, 1, 1, 1), 0.6)
    rgb_gain = torch.tensor([1.2, 1.0, 1 / 1.2])[None, :, None, None]
    observed = photometric_isp(
        clean,
        exposure_ev=exposure_ev,
        rgb_gain=rgb_gain,
        evidence=False,
    )["rgb_observed"]

    fitted = fit_online_exposure_gain(
        clean,
        observed,
        updates_per_frame=0,
        initialization="log_ratio",
    )

    assert torch.allclose(fitted["exposure_ev"], exposure_ev, atol=1e-5)
    assert torch.allclose(fitted["rgb_gain"], rgb_gain, atol=1e-5)
    assert torch.equal(fitted["initializer_applied"], torch.ones((1, 1), dtype=torch.bool))
    assert fitted["initializer_valid_fraction"].item() == pytest.approx(0.25)


def test_log_ratio_initializer_preserves_previous_state_without_support():
    clean = torch.full((2, 3, 4, 4), 0.2)
    exposure_ev = torch.tensor([0.25, 3.0]).reshape(2, 1, 1, 1)
    observed = photometric_isp(clean, exposure_ev=exposure_ev, evidence=False)["rgb_observed"]

    fitted = fit_online_exposure_gain(
        clean,
        observed,
        updates_per_frame=0,
        initialization="log_ratio",
        min_initializer_valid_fraction=0.1,
    )

    assert fitted["initializer_applied"].flatten().tolist() == [True, False]
    assert fitted["initializer_valid_fraction"][1].item() == 0
    assert torch.allclose(fitted["exposure_ev"][1], fitted["exposure_ev"][0])


def test_online_refinement_cannot_move_state_without_trusted_pixels():
    clean = torch.full((2, 3, 4, 4), 0.2)
    exposure_ev = torch.tensor([0.25, 3.0]).reshape(2, 1, 1, 1)
    observed = photometric_isp(clean, exposure_ev=exposure_ev, evidence=False)["rgb_observed"]

    fitted = fit_online_exposure_gain(
        clean,
        observed,
        updates_per_frame=4,
        initialization="log_ratio",
        min_initializer_valid_fraction=0.1,
    )

    assert fitted["initializer_applied"].flatten().tolist() == [True, False]
    assert fitted["refinement_applied"].flatten().tolist() == [True, False]
    assert torch.allclose(fitted["exposure_ev"][1], fitted["exposure_ev"][0])
    assert torch.allclose(fitted["rgb_gain"][1], fitted["rgb_gain"][0])


def test_shared_response_uses_one_persistent_optimizer(monkeypatch):
    calls: list[int] = []
    real_adam = torch.optim.Adam

    def recording_adam(parameters, *args, **kwargs):
        params = tuple(parameters)
        calls.append(len(params))
        return real_adam(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    clean = 0.1 + 0.2 * torch.rand(3, 3, 4, 4)
    observed = photometric_isp(
        clean,
        response_gamma=torch.full((3, 1, 1, 1), 0.9),
        evidence=False,
    )["rgb_observed"]

    fitted = fit_online_exposure_gain_response(clean, observed, updates_per_frame=1)

    assert calls == [1, 2, 2, 2]
    assert torch.isfinite(fitted["response_gamma"]).all()


def test_shared_response_skips_unsupported_frame_without_momentum_drift():
    clean = torch.full((3, 3, 4, 4), 0.2)
    observed = photometric_isp(
        clean,
        response_gamma=torch.full((3, 1, 1, 1), 0.82),
        evidence=False,
    )["rgb_observed"]
    observed[1] = 1.0

    fitted = fit_online_exposure_gain_response(
        clean,
        observed,
        updates_per_frame=2,
        min_valid_fraction=0.1,
    )

    assert fitted["response_refinement_applied"].flatten().tolist() == [True, False, True]
    assert fitted["trusted_valid_fraction"][1].item() == 0
    assert torch.equal(fitted["response_gamma"][1], fitted["response_gamma"][0])
    assert torch.equal(fitted["exposure_ev"][1], fitted["exposure_ev"][0])
    assert torch.equal(fitted["rgb_gain"][1], fitted["rgb_gain"][0])


def test_full_stage_fit_cannot_absorb_spatial_misalignment():
    torch.manual_seed(41)
    clean = torch.rand(1, 3, 16, 16)
    shifted = clean.roll(2, dims=-1)
    identity_mae = (clean - shifted).abs().mean()

    fitted = fit_photometric_stages(clean, shifted, steps=128)
    fitted_mae = (fitted["rgb_observed"] - shifted).abs().mean()

    assert fitted_mae > 0.1
    assert fitted_mae > identity_mae * 0.5
