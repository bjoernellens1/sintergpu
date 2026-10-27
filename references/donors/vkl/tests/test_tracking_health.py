import pytest
import torch

from vkl.kernels import (
    TRACK_REASON_CONDITION,
    TRACK_REASON_GEOMETRY,
    TRACK_REASON_PARTIALS,
    TRACK_REASON_PHOTOMETRIC,
    TRACK_REASON_RESIDUAL,
    TRACK_REASON_SOLVE,
    tracking_health,
)


def _inputs(batch: int) -> dict[str, torch.Tensor]:
    return {
        "solve_valid": torch.ones(batch, dtype=torch.bool),
        "partial_count": torch.full((batch,), 4, dtype=torch.int64),
        "condition_proxy": torch.full((batch,), 32.0),
        "geometry_inlier_count": torch.full((batch,), 32, dtype=torch.int64),
        "photometric_inlier_count": torch.full((batch,), 32, dtype=torch.int64),
        "geometry_residual_abs_mean": torch.full((batch,), 0.01),
    }


@pytest.mark.parametrize("batch", [1, 8])
def test_tracking_health_accepts_well_constrained_rgbd_update(batch: int):
    out = tracking_health(**_inputs(batch), min_partials=4, min_geometry_inliers=32, min_photometric_inliers=32)

    assert out["valid"].all()
    assert torch.equal(out["reason_bits"], torch.zeros(batch, dtype=torch.int32))
    assert ((out["score"] > 0) & (out["score"] <= 1)).all()


def test_tracking_health_keeps_independent_failure_reasons():
    values = _inputs(8)
    values["solve_valid"][1] = False
    values["partial_count"][2] = 0
    values["geometry_inlier_count"][3] = 0
    values["photometric_inlier_count"][4] = 0
    values["geometry_residual_abs_mean"][5] = 1.0
    values["condition_proxy"][6] = 1e8

    out = tracking_health(**values, min_partials=4, min_geometry_inliers=32, min_photometric_inliers=32)

    assert out["valid"].tolist() == [True, False, False, False, False, False, False, True]
    for index, reason in enumerate(
        (TRACK_REASON_SOLVE, TRACK_REASON_PARTIALS, TRACK_REASON_GEOMETRY, TRACK_REASON_PHOTOMETRIC, TRACK_REASON_RESIDUAL, TRACK_REASON_CONDITION),
        start=1,
    ):
        assert int(out["reason_bits"][index]) & reason


def test_tracking_health_classifies_singular_solver_condition_as_unhealthy():
    values = _inputs(1)
    values["condition_proxy"].fill_(torch.inf)

    out = tracking_health(**values)

    assert not out["valid"].item()
    assert int(out["reason_bits"][0]) & TRACK_REASON_CONDITION
