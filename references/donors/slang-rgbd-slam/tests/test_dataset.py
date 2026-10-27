import numpy as np
import pytest
from PIL import Image

from slang_rgbd_slam.dataset import TUMDataset
from slang_rgbd_slam.tracker import Camera
from slang_rgbd_slam.trajectory import absolute_trajectory_error, read_trajectory, write_trajectory


def test_loader_association(tmp_path):
    c = Camera(4, 4, 4, 4, 1.5, 1.5)
    Image.fromarray(np.full((4, 4), 5000, dtype=np.uint16)).save(tmp_path / "d.png")
    Image.fromarray(np.full((4, 4, 3), 80, dtype=np.uint8)).save(tmp_path / "r.png")
    (tmp_path / "depth.txt").write_text("1.0 d.png\n1.04 d.png\n")
    (tmp_path / "rgb.txt").write_text("1.01 r.png\n")
    ds = TUMDataset(tmp_path, c)
    assert len(ds) == 1
    np.testing.assert_array_equal(ds[0].depth, 1)
    assert ds[0].rgb.shape == (4, 4, 3)
    (tmp_path / "association.txt").write_text("1.01 r.png 1.0 d.png\n")
    assert len(TUMDataset(tmp_path, c)) == 1
    (tmp_path / "association.txt").write_text("1.0 missing.png 1.0 d.png\n")
    with pytest.raises(FileNotFoundError):
        TUMDataset(tmp_path, c)


def test_malformed_and_corrupt(tmp_path):
    c = Camera(4, 4, 4, 4, 1.5, 1.5)
    (tmp_path / "depth.txt").write_text("1 d.png\n")
    (tmp_path / "d.png").write_text("corrupt")
    with pytest.raises(ValueError):
        TUMDataset(tmp_path, c)[0]
    Image.fromarray(np.zeros((2, 4), dtype=np.uint16)).save(tmp_path / "d.png")
    with pytest.raises(ValueError):
        TUMDataset(tmp_path, c)[0]


def test_trajectory_rigid_no_scale(tmp_path):
    est = []
    gt = []
    for i in range(5):
        p = np.eye(4)
        p[:3, 3] = [i, i * i, 0]
        est.append((i, p))
        q = p.copy()
        q[:3, 3] += [2, -3, 1]
        gt.append((i, q))
    path = tmp_path / "poses.txt"
    write_trajectory(path, est)
    actual = read_trajectory(path)
    assert absolute_trajectory_error(actual, gt)["rmse_m"] < 1e-8
    scaled = [(t, p.copy()) for t, p in est]
    for _, p in scaled:
        p[:3, 3] *= 2
    assert absolute_trajectory_error(scaled, gt)["rmse_m"] > 1
    assert absolute_trajectory_error(est, [(20, np.eye(4))])["count"] == 0
