import numpy as np

from slang_rgbd_slam.synthetic import render_depth, synthetic_sequence


def test_independent_ray_render():
    camera, frames = synthetic_sequence(3)
    assert frames[0].depth.shape == (120, 160)
    assert np.isfinite(frames[0].depth).all()
    assert np.ptp(frames[0].depth) > 1
    assert not np.array_equal(frames[0].depth, frames[1].depth)
    np.testing.assert_array_equal(
        frames[2].depth, render_depth(camera, frames[2].world_from_camera)
    )
