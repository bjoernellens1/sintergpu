from __future__ import annotations

import pytest
import torch

from vkl.kernels import (
    fit_local_planes,
    point_splat_render,
    voxel_surface_mesh,
    voxelize_points,
    world_voxel_reduce,
)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_voxelize_points_compacts_each_batch_without_cross_batch_merging(batch: int):
    points = torch.tensor(
        [
            [[0.01, 0.01, 0.01], [0.03, 0.01, 0.01], [1.01, 0.0, 0.0], [1.03, 0.0, 0.0]]
        ]
    ).repeat(batch, 1, 1)
    points[:, :, 1] += torch.arange(batch, dtype=points.dtype)[:, None] * 2

    out = voxelize_points(points, 1.0)

    assert out["xyz"].shape == (batch * 2, 3)
    assert torch.equal(torch.bincount(out["batch_index"], minlength=batch), torch.full((batch,), 2))
    assert torch.equal(out["count"], torch.full((batch * 2,), 2, dtype=torch.int64))
    assert torch.allclose(out["xyz"][0], torch.tensor([0.02, 0.01, 0.01]))


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_fit_local_planes_recovers_planar_normal_for_b1_through_b8(batch: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(2.0), torch.arange(2.0), indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.full((4,), 2.0)), -1)[None]
    points = points.repeat(batch, 1, 1)
    neighbors = torch.arange(4, dtype=torch.int64)[None, None].expand(batch, 4, -1)

    out = fit_local_planes(points, neighbors)

    assert out["valid"].all()
    assert torch.allclose(out["normal"].abs()[..., 2], torch.ones(batch, 4), atol=1e-5)
    assert torch.allclose(out["residual"], torch.zeros(batch, 4), atol=1e-6)


def test_voxel_surface_mesh_removes_shared_face_and_is_closed_boundary():
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64)

    out = voxel_surface_mesh(voxels, 1.0)

    assert out["vertices"].shape == (12, 3)
    assert out["faces"].shape == (20, 3)
    edges = torch.cat(
        (
            out["faces"][:, [0, 1]],
            out["faces"][:, [1, 2]],
            out["faces"][:, [2, 0]],
        )
    ).sort(-1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    assert torch.equal(counts, torch.full_like(counts, 2))


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_point_splat_render_depth_buffers_nearest_point_for_b1_through_b8(batch: int):
    xyz = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]]).repeat(batch, 1, 1)
    rgb = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]).repeat(batch, 1, 1)
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]).repeat(batch, 1, 1)

    out = point_splat_render(xyz, rgb, intrinsics, torch.eye(4)[None].repeat(batch, 1, 1), image_hw=(3, 3))

    assert out["valid"].all(dim=(1, 2, 3)).tolist() == [False] * batch
    assert out["valid"][:, 0, 1, 1].all()
    assert torch.allclose(out["depth"][:, 0, 1, 1], torch.ones(batch))
    assert torch.allclose(out["rgb"][:, :, 1, 1], rgb[:, 1])
    assert torch.equal(out["support_count"][:, 0, 1, 1], torch.full((batch,), 2, dtype=torch.int64))


def test_point_splat_render_bounded_radius_fills_a_local_disc():
    xyz = torch.tensor([[[0.0, 0.0, 1.0]]])
    rgb = torch.tensor([[[0.2, 0.4, 0.6]]])
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])

    out = point_splat_render(xyz, rgb, intrinsics, torch.eye(4)[None], image_hw=(3, 3), splat_radius=1)

    assert out["valid"].all()
    assert torch.allclose(out["depth"], torch.ones_like(out["depth"]))
    assert torch.allclose(out["rgb"], rgb[:, 0, :, None, None].expand(-1, -1, 3, 3))



@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_world_voxel_reduce_requires_distinct_sources_for_b1_through_b8(batch: int):
    xyz = torch.tensor(
        [[[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.01, 0.0, 0.0], [1.02, 0.0, 0.0]]]
    ).repeat(batch, 1, 1)
    source_ids = torch.tensor([[7, 7, 8, 9]], dtype=torch.int64).repeat(batch, 1)

    out = world_voxel_reduce(xyz, source_ids, 1.0, min_source_support=2)

    assert torch.equal(out["source_support_count"], torch.tensor([1, 2] * batch, dtype=torch.int64))
    assert out["valid"].tolist() == [False, True] * batch
