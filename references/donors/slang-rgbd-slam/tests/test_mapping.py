import numpy as np
import pytest

from slang_rgbd_slam.mapping import VoxelMap
from slang_rgbd_slam.tracker import Camera


def test_voxel_color_count_pose_and_bound(tmp_path):
    c = Camera(4, 4, 10, 10, 1.5, 1.5)
    m = VoxelMap(voxel_size=1, max_voxels=2)
    pose = np.eye(4)
    pose[:3, 3] = [2, 0, 0]
    depth = np.ones((4, 4))
    rgb = np.full((4, 4, 3), [20, 50, 90], dtype=np.uint8)
    assert m.integrate(depth, c, pose, rgb) == 16
    assert len(m) == 2
    m.integrate(depth, c, pose, rgb)
    assert m.counts.sum() == 16  # two of four spatial voxels retained, seen twice
    assert np.all(m.points[:, 0] > 1)
    np.testing.assert_array_equal(m.rgb, np.tile([20, 50, 90], (2, 1)))
    p = tmp_path / "map.ply"
    m.export_ply(p)
    assert "element vertex 2" in p.read_text()


def test_empty_map():
    m = VoxelMap()
    c = Camera(4, 4, 4, 4, 1.5, 1.5)
    assert m.integrate(np.zeros((4, 4)), c, np.eye(4)) == 0
    assert len(m) == 0


@pytest.mark.parametrize("capacity", [0.5, 1.5, np.nan, True])
def test_map_capacity_requires_positive_integer(capacity):
    with pytest.raises(ValueError, match="map limits"):
        VoxelMap(max_voxels=capacity)
