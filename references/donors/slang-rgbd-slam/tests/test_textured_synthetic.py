import numpy as np

from slang_rgbd_slam.feature_reference import CPUFeatures
from slang_rgbd_slam.textured_synthetic import render_textured_rgb, textured_sequence, world_texture


def test_static_texture_repeat_and_viewpoint_world_intersections():
    camera, frames = textured_sequence(8)
    _, repeated = textured_sequence(8)
    for a, b in zip(frames, repeated):
        np.testing.assert_array_equal(a.depth, b.depth)
        np.testing.assert_array_equal(a.rgb, b.rgb)
        np.testing.assert_array_equal(
            a.rgb, render_textured_rgb(camera, a.depth, a.world_from_camera)
        )
    sample = frames[3]
    yy, xx = np.indices(sample.depth.shape)
    p = np.stack(
        (
            (xx - camera.cx) * sample.depth / camera.fx,
            (yy - camera.cy) * sample.depth / camera.fy,
            sample.depth,
        ),
        axis=-1,
    )
    world = p @ sample.world_from_camera[:3, :3].T + sample.world_from_camera[:3, 3]
    # Identical physical points have identical appearance regardless of array/view.
    np.testing.assert_array_equal(sample.rgb[10:30, 20:40], world_texture(world[10:30, 20:40]))


def test_closed_path_exact_endpoint_and_sufficient_calibrated_features():
    camera, frames = textured_sequence()
    np.testing.assert_array_equal(frames[0].world_from_camera, np.eye(4))
    np.testing.assert_array_equal(frames[-1].world_from_camera, np.eye(4))
    np.testing.assert_array_equal(frames[0].depth, frames[-1].depth)
    np.testing.assert_array_equal(frames[0].rgb, frames[-1].rgb)
    assert np.linalg.norm(frames[20].world_from_camera[:3, 3]) > 0.05
    frontend = CPUFeatures(camera, max_features=512)
    assert frontend.extract(frames[0].rgb, frames[0].depth)["count"] >= 100
    assert frontend.extract(frames[20].rgb, frames[20].depth)["count"] >= 100
