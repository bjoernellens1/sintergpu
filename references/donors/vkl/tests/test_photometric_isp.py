import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from vkl.bench.photometric import (
    fit_online_exposure_gain,
    fit_online_exposure_gain_response,
    fit_photometric_exposure_gain,
    fit_photometric_stages,
)
from vkl.kernels import photometric_isp

_pair_spec = importlib.util.spec_from_file_location(
    "ppisp_pairs", Path(__file__).resolve().parents[1] / "recipes/photometric/run_ppisp_pairs.py"
)
assert _pair_spec is not None and _pair_spec.loader is not None
_pair_recipe = importlib.util.module_from_spec(_pair_spec)
sys.modules[_pair_spec.name] = _pair_recipe
_pair_spec.loader.exec_module(_pair_recipe)
discover_ppisp_pairs = _pair_recipe.discover_ppisp_pairs


def test_ppisp_fit_resize_is_bounded_and_preserves_linear_batch_contract():
    rgb = torch.rand(8, 3, 120, 160)

    reduced = _pair_recipe._resize_max_side(rgb, 80)

    assert reduced.shape == (8, 3, 60, 80)
    assert reduced.dtype == rgb.dtype and reduced.device == rgb.device


def test_photometric_isp_applies_batched_ev_and_rgb_gain():
    rgb = torch.full((2, 3, 2, 2), 0.25)
    out = photometric_isp(
        rgb,
        exposure_ev=torch.tensor([[[[1.0]]], [[[0.0]]]]),
        rgb_gain=torch.tensor(
            [[[[1.0]], [[2.0]], [[1.0]]], [[[1.0]], [[1.0]], [[1.0]]]]
        ),
    )
    assert torch.allclose(out["rgb_observed"][0, :, 0, 0], torch.tensor([0.5, 1.0, 0.5]))
    assert torch.allclose(out["rgb_observed"][1], rgb[1])


def test_photometric_isp_rejects_non_linear_contract_shapes_and_keeps_gradients_finite():
    rgb = torch.full((1, 3, 2, 2), 0.25, requires_grad=True)
    ev = torch.zeros((1, 1, 1, 1), requires_grad=True)
    with pytest.raises(ValueError):
        photometric_isp(rgb, exposure_ev=torch.ones(1, 1, 1, 2))
    photometric_isp(rgb, exposure_ev=ev)["rgb_observed"].sum().backward()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(ev.grad).all()


def test_photometric_isp_optional_stages_preserve_identity_and_response_is_monotonic():
    rgb = torch.tensor([[[[0.1, 0.8]], [[0.2, 0.7]], [[0.3, 0.6]]]])
    out = photometric_isp(
        rgb,
        color_matrix=torch.eye(3)[None],
        vignette=torch.zeros(1, 3, 3),
        response_gamma=torch.ones(1, 1, 1, 1),
    )
    assert torch.allclose(out["rgb_observed"], rgb)

    curve = torch.linspace(0, 1, 9)[None, None].expand(1, 3, 1, -1)
    response = photometric_isp(
        curve, response_gamma=torch.full((1, 1, 1, 1), 1.4)
    )["rgb_observed"]
    assert torch.all(response[..., 1:] >= response[..., :-1])


def test_photometric_isp_optional_stage_evidence_is_batched_and_finite():
    out = photometric_isp(
        torch.rand(2, 3, 5, 7),
        color_matrix=torch.eye(3).repeat(2, 1, 1),
        vignette=torch.zeros(2, 3, 3),
        response_gamma=torch.ones(2, 1, 1, 1),
    )
    assert out["color_deviation"].shape == (2, 1)
    assert out["vignette_gain_range"].shape == (2, 2)
    assert out["crf_response_range"].shape == (2, 2)
    assert all(torch.isfinite(value).all() for value in out.values())


def test_vignetting_is_center_normalized_physical_attenuation():
    rgb = torch.ones(1, 3, 5, 5)
    out = photometric_isp(rgb, vignette=torch.full((1, 3, 3), 0.1))

    assert torch.all(out["rgb_observed"] <= 1)
    assert torch.allclose(out["rgb_observed"][..., 2, 2], torch.ones(1, 3))
    assert torch.all(out["vignette_gain_range"][:, 0] > 0)
    assert torch.all(out["vignette_gain_range"][:, 1] <= 1)


def test_vignetting_rejects_corner_brightening_coefficients():
    with pytest.raises(ValueError, match="non-negative attenuation"):
        photometric_isp(torch.ones(1, 3, 3, 3), vignette=-torch.ones(1, 3, 3))


def test_photometric_isp_reports_pre_response_lower_clipping():
    rgb = torch.full((1, 3, 2, 2), 0.25)
    matrix = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))[None]

    out = photometric_isp(rgb, color_matrix=matrix)

    assert torch.allclose(out["clipping_ratio"], torch.tensor([[1 / 3, 0.0]]))


def test_power_response_has_finite_black_gradients():
    rgb = torch.zeros((1, 3, 2, 2), requires_grad=True)
    gamma = torch.full((1, 1, 1, 1), 0.5, requires_grad=True)

    output = photometric_isp(rgb, response_gamma=gamma)["rgb_observed"]
    output.sum().backward()

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(gamma.grad).all()


def test_exposure_only_fast_path_skips_optional_stages_and_evidence_reductions():
    rgb = torch.full((1, 3, 2, 2), 0.25)
    with patch("vkl.kernels.reference.photometric_isp.torch.einsum", side_effect=AssertionError):
        out = photometric_isp(rgb, exposure_ev=torch.ones(1, 1, 1, 1), evidence=False)

    assert tuple(out) == ("rgb_observed",)
    assert torch.allclose(out["rgb_observed"], torch.full_like(rgb, 0.5))


def test_photometric_isp_known_exposure_and_gain_are_recoverable():
    torch.manual_seed(3)
    clean = 0.1 + 0.3 * torch.rand(1, 3, 16, 16)
    exposure_ev = torch.tensor([[[[0.35]]]])
    rgb_gain = torch.tensor([[[[1.1]], [[0.9]], [[1 / 0.99]]]])
    observed = photometric_isp(clean, exposure_ev=exposure_ev, rgb_gain=rgb_gain)["rgb_observed"]

    fitted = fit_photometric_exposure_gain(clean, observed, steps=128)

    assert torch.allclose(fitted["rgb_observed"], observed, atol=2e-3)
    assert torch.allclose(fitted["exposure_ev"], exposure_ev, atol=2e-2)
    assert torch.allclose(fitted["rgb_gain"], rgb_gain, atol=2e-2)


def test_exposure_gain_cannot_remove_spatially_displaced_structure():
    torch.manual_seed(4)
    clean = torch.rand(1, 3, 16, 16)
    shifted = clean.roll(2, dims=-1)

    fitted = fit_photometric_exposure_gain(clean, shifted, steps=128)

    assert (fitted["rgb_observed"] - shifted).abs().mean() > 0.03


def test_constrained_full_stage_fit_recovers_known_photometric_transform():
    torch.manual_seed(9)
    clean = 0.1 + 0.25 * torch.rand(1, 3, 16, 20)
    observed = photometric_isp(
        clean,
        exposure_ev=torch.tensor([[[[0.15]]]]),
        rgb_gain=torch.tensor([[[[1.08]], [[0.96]], [[1 / 1.0368]]]]),
        color_matrix=torch.tensor([[[1.0, 0.02, -0.01], [-0.01, 1.0, 0.02], [0.01, -0.02, 1.0]]]),
        vignette=torch.full((1, 3, 3), 0.015),
        response_gamma=torch.full((1, 1, 1, 1), 1.08),
        evidence=False,
    )["rgb_observed"]

    fitted = fit_photometric_stages(clean, observed, steps=384)

    assert (fitted["rgb_observed"] - observed).abs().mean() < 7e-3
    assert torch.allclose(torch.diagonal(fitted["color_matrix"], dim1=-2, dim2=-1), torch.ones(1, 3))
    assert torch.all(fitted["vignette"] >= 0)


def test_full_stage_fit_can_disable_unneeded_optional_stages():
    clean = 0.1 + 0.25 * torch.rand(1, 3, 8, 8)
    observed = photometric_isp(clean, exposure_ev=torch.full((1, 1, 1, 1), 0.2), evidence=False)["rgb_observed"]

    fitted = fit_photometric_stages(
        clean,
        observed,
        steps=64,
        fit_rgb_gain=False,
        fit_color_matrix=False,
        fit_vignette=False,
        fit_response=False,
    )

    assert (fitted["rgb_observed"] - observed).abs().mean() < 2e-3
    assert torch.allclose(fitted["color_matrix"], torch.eye(3)[None])
    assert torch.equal(fitted["vignette"], torch.zeros_like(fitted["vignette"]))
    assert torch.equal(fitted["response_gamma"], torch.ones_like(fitted["response_gamma"]))


def test_online_ev_wb_fit_improves_with_more_causal_updates():
    torch.manual_seed(19)
    clean = 0.1 + 0.2 * torch.rand(8, 3, 8, 8)
    truth_ev = torch.linspace(0, 0.4, 8).reshape(8, 1, 1, 1)
    truth_gain = torch.exp(torch.stack((truth_ev[:, 0], torch.zeros_like(truth_ev[:, 0]), -truth_ev[:, 0]), dim=1))
    observed = photometric_isp(clean, exposure_ev=truth_ev, rgb_gain=truth_gain, evidence=False)["rgb_observed"]

    one = fit_online_exposure_gain(clean, observed, updates_per_frame=1)
    four = fit_online_exposure_gain(clean, observed, updates_per_frame=4)

    one_error = (one["rgb_observed"] - observed).abs().mean()
    four_error = (four["rgb_observed"] - observed).abs().mean()
    assert four_error < one_error
    assert four_error < 1e-2


def test_online_ev_wb_log_ratio_initializer_recovers_known_state_without_updates():
    torch.manual_seed(29)
    clean = 0.1 + 0.2 * torch.rand(5, 3, 8, 8)
    truth_ev = torch.linspace(-0.2, 0.35, 5).reshape(5, 1, 1, 1)
    truth_chroma = torch.tensor([[0.1, -0.05], [0.04, 0.03], [0.0, 0.0], [-0.07, 0.08], [0.02, -0.06]])[:, :, None, None]
    truth_log_gain = torch.cat(
        (
            (2 * truth_chroma[:, :1] - truth_chroma[:, 1:]) / 3,
            (-truth_chroma[:, :1] - truth_chroma[:, 1:]) / 3,
            (2 * truth_chroma[:, 1:] - truth_chroma[:, :1]) / 3,
        ),
        dim=1,
    )
    observed = photometric_isp(
        clean,
        exposure_ev=truth_ev,
        rgb_gain=truth_log_gain.exp(),
        evidence=False,
    )["rgb_observed"]

    fitted = fit_online_exposure_gain(
        clean,
        observed,
        updates_per_frame=0,
        initialization="log_ratio",
    )

    assert torch.allclose(fitted["rgb_observed"], observed, atol=1e-5)
    assert torch.allclose(fitted["exposure_ev"], truth_ev, atol=1e-5)
    assert torch.allclose(fitted["chroma_rg_bg"], truth_chroma, atol=1e-5)


def test_online_ev_wb_shared_response_recovers_a_fixed_camera_curve():
    torch.manual_seed(31)
    clean = 0.12 + 0.16 * torch.rand(8, 3, 12, 10)
    truth_ev = torch.linspace(-0.12, 0.22, 8).reshape(8, 1, 1, 1)
    truth_chroma = torch.linspace(-0.06, 0.08, 8).reshape(8, 1, 1, 1)
    truth_gain = torch.cat((truth_chroma, torch.zeros_like(truth_chroma), -truth_chroma), dim=1).exp()
    truth_response = torch.full((1, 1, 1, 1), 0.82)
    observed = photometric_isp(
        clean,
        exposure_ev=truth_ev,
        rgb_gain=truth_gain,
        response_gamma=truth_response.expand(clean.shape[0], -1, -1, -1),
        evidence=False,
    )["rgb_observed"]

    one = fit_online_exposure_gain_response(clean, observed, updates_per_frame=1)
    eight = fit_online_exposure_gain_response(clean, observed, updates_per_frame=8)

    assert (eight["rgb_observed"] - observed).abs().mean() < (one["rgb_observed"] - observed).abs().mean()
    assert torch.allclose(eight["response_gamma"][-1:], truth_response, atol=0.08)


def test_ppisp_pair_discovery_keeps_only_exact_relative_paths(tmp_path):
    archive = tmp_path / "colmap" / "huerstholz_colmap.zip"
    archive.parent.mkdir()
    import zipfile
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("huerstholz/images/oms/shared.jpg", b"standard")
        out.writestr("huerstholz/images/oms/standard_only.jpg", b"standard")
        out.writestr("huerstholz_auto/images/oms/shared.jpg", b"automatic")
        out.writestr("huerstholz_auto/images/oms/auto_only.jpg", b"automatic")

    pairs = discover_ppisp_pairs(tmp_path, scene="huerstholz", camera="oms")

    assert [pair.relative_path for pair in pairs] == ["shared.jpg"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_photometric_isp_compiled_matches_eager():
    rgb = torch.rand(2, 3, 11, 13, device="cuda")
    matrix = torch.eye(3, device="cuda").repeat(2, 1, 1)
    eager = photometric_isp(rgb, color_matrix=matrix, validate=False, evidence=False)["rgb_observed"]
    compiled = torch.compile(photometric_isp, fullgraph=True)(rgb, color_matrix=matrix, validate=False, evidence=False)["rgb_observed"]
    assert torch.allclose(compiled, eager)


@pytest.mark.parametrize("bad_rgb", [float("nan"), -0.01, 1.01])
def test_photometric_isp_rejects_nonfinite_or_out_of_range_rgb(bad_rgb):
    rgb = torch.full((1, 3, 2, 2), 0.25)
    rgb[0, 0, 0, 0] = bad_rgb
    with pytest.raises(ValueError):
        photometric_isp(rgb)


@pytest.mark.parametrize("bad_gain", [float("nan"), -0.01])
def test_photometric_isp_rejects_nonfinite_or_negative_rgb_gain(bad_gain):
    rgb = torch.full((1, 3, 2, 2), 0.25)
    gain = torch.ones((1, 3, 1, 1))
    gain[0, 0, 0, 0] = bad_gain
    with pytest.raises(ValueError):
        photometric_isp(rgb, rgb_gain=gain)


def test_photometric_isp_rejects_nonfinite_exposure_ev():
    rgb = torch.full((1, 3, 2, 2), 0.25)
    with pytest.raises(ValueError):
        photometric_isp(rgb, exposure_ev=torch.full((1, 1, 1, 1), float("inf")))
