import pytest
import torch

from vkl.kernels import REGISTRY, dense_track_reduce, deterministic_reduce_solve


def _inputs(batch: int, *, translation_z: float = 0.0):
    height, width = 6, 8
    depth = torch.full((batch, 1, height, width), 2.0)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 3.5], [0.0, 8.0, 2.5], [0.0, 0.0, 1.0]]]
    ).expand(batch, -1, -1).clone()
    source_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    source_pose[:, 2, 3] = translation_z
    reference_pose = torch.eye(4)[None].expand(batch, -1, -1).clone()
    return depth, intrinsics, source_pose, reference_pose


def _rgb(batch: int, height: int = 6, width: int = 8) -> torch.Tensor:
    x = torch.linspace(0.1, 0.9, width).view(1, 1, 1, width)
    y = torch.linspace(0.0, 0.2, height).view(1, 1, height, 1)
    return (x + y).expand(batch, 3, height, width).contiguous()


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_geometry_is_fixed_shape_and_identity_consistent(batch: int):
    depth, intrinsics, source_pose, reference_pose = _inputs(batch)

    out = dense_track_reduce(
        source_depth=depth,
        source_intrinsics=intrinsics,
        source_pose_c2w=source_pose,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
    )

    assert out["H"].shape == (batch, 6, 6)
    assert out["b"].shape == (batch, 6)
    assert torch.allclose(out["H"], out["H"].transpose(-1, -2))
    assert (out["geometry_inlier_count"] > 0).all()
    assert torch.allclose(out["geometry_residual_abs_mean"], torch.zeros(batch))
    assert torch.equal(out["photometric_inlier_count"], torch.zeros(batch, dtype=torch.int64))
    assert REGISTRY["dense_track_reduce"] is dense_track_reduce


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_geometry_detects_depth_axis_pose_error(batch: int):
    depth, intrinsics, source_pose, reference_pose = _inputs(batch, translation_z=0.2)

    out = dense_track_reduce(
        source_depth=depth,
        source_intrinsics=intrinsics,
        source_pose_c2w=source_pose,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
    )

    assert (out["geometry_inlier_count"] > 0).all()
    assert torch.allclose(out["geometry_residual_abs_mean"], torch.full((batch,), 0.2), atol=1e-5)
    assert (out["b"].abs().sum(-1) > 0).all()


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_accumulates_photometric_evidence(batch: int):
    depth, intrinsics, source_pose, reference_pose = _inputs(batch)
    reference_rgb = _rgb(batch)
    source_rgb = (reference_rgb + 0.1).clamp_max(1)

    out = dense_track_reduce(
        source_depth=depth,
        source_intrinsics=intrinsics,
        source_pose_c2w=source_pose,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        source_rgb=source_rgb,
        reference_rgb=reference_rgb,
    )

    assert (out["photometric_inlier_count"] > 0).all()
    assert (out["photometric_b"].abs().sum(-1) > 0).all()
    assert torch.allclose(out["H"], out["geometry_H"] + out["photometric_H"])
    assert torch.allclose(out["b"], out["geometry_b"] + out["photometric_b"])


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_exposes_fixed_spatial_partials(batch: int):
    depth, intrinsics, source_pose, reference_pose = _inputs(batch, translation_z=0.2)

    out = dense_track_reduce(
        source_depth=depth,
        source_intrinsics=intrinsics,
        source_pose_c2w=source_pose,
        reference_depth=depth,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        partial_grid=(2, 2),
    )

    assert out["partial_H"].shape == (batch, 4, 6, 6)
    assert out["partial_b"].shape == (batch, 4, 6)
    assert out["partial_valid"].shape == (batch, 4)
    assert torch.allclose(out["H"], out["partial_H"].sum(1))
    assert torch.allclose(out["b"], out["partial_b"].sum(1))
    solved = deterministic_reduce_solve(out["partial_H"], out["partial_b"], out["partial_valid"])
    assert torch.allclose(solved["H"], out["H"])
    assert torch.allclose(solved["b"], out["b"])


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_validate_false_preserves_validated_result(batch: int):
    depth, intrinsics, source_pose, reference_pose = _inputs(batch, translation_z=0.2)
    kwargs = {
        "source_depth": depth,
        "source_intrinsics": intrinsics,
        "source_pose_c2w": source_pose,
        "reference_depth": depth,
        "reference_intrinsics": intrinsics,
        "reference_pose_c2w": reference_pose,
    }

    eager = dense_track_reduce(**kwargs)
    unchecked = dense_track_reduce(**kwargs, validate=False)

    assert torch.allclose(unchecked["H"], eager["H"])
    assert torch.allclose(unchecked["b"], eager["b"])


@pytest.mark.parametrize("batch", [1, 8])
def test_dense_track_reduce_marks_depth_unavailable_as_zero_evidence(batch: int):
    _, intrinsics, source_pose, reference_pose = _inputs(batch)
    rgb = _rgb(batch)

    out = dense_track_reduce(
        source_depth=None,
        source_intrinsics=intrinsics,
        source_pose_c2w=source_pose,
        reference_depth=None,
        reference_intrinsics=intrinsics,
        reference_pose_c2w=reference_pose,
        source_rgb=rgb,
        reference_rgb=rgb,
        partial_grid=(2, 2),
    )

    assert out["H"].shape == (batch, 6, 6)
    assert out["partial_H"].shape == (batch, 4, 6, 6)
    assert torch.equal(out["H"], torch.zeros_like(out["H"]))
    assert torch.equal(out["b"], torch.zeros_like(out["b"]))
    assert not out["partial_valid"].any()
    assert torch.equal(out["geometry_inlier_count"], torch.zeros(batch, dtype=torch.int64))
    assert torch.equal(out["photometric_inlier_count"], torch.zeros(batch, dtype=torch.int64))


def test_dense_track_reduce_rejects_one_sided_depth():
    depth, intrinsics, source_pose, reference_pose = _inputs(1)
    rgb = _rgb(1)

    with pytest.raises(ValueError, match="source_depth and reference_depth"):
        dense_track_reduce(
            source_depth=depth,
            source_intrinsics=intrinsics,
            source_pose_c2w=source_pose,
            reference_depth=None,
            reference_intrinsics=intrinsics,
            reference_pose_c2w=reference_pose,
            source_rgb=rgb,
            reference_rgb=rgb,
        )
