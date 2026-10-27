"""Independent ray-rendered static room with planes and curved surfaces."""

import numpy as np

from .dataset import Frame
from .tracker import Camera


def render_depth(camera, world_from_camera):
    pose = np.asarray(world_from_camera, dtype=float)
    v, u = np.indices((camera.height, camera.width))
    directions = (
        np.stack(
            ((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)), axis=-1
        )
        @ pose[:3, :3].T
    )
    origin = pose[:3, 3]
    best = np.full(u.shape, np.inf)
    # Back, left, right, floor and ceiling surfaces form a closed visible room.
    for axis, coordinate in ((2, 4.0), (0, -1.8), (0, 1.8), (1, 1.25), (1, -1.3)):
        denom = directions[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (coordinate - origin[axis]) / denom
        best = np.minimum(best, np.where(t > 0, t, np.inf))
    # Spheres provide changing normals; intersections recomputed for each viewpoint.
    for center, radius in (
        ([-0.7, 0.15, 2.8], 0.65),
        ([0.8, -0.3, 3.2], 0.7),
        ([0.2, 0.9, 2.6], 0.35),
    ):
        offset = origin - np.asarray(center)
        a = np.sum(directions**2, axis=-1)
        b = 2 * np.sum(directions * offset, axis=-1)
        c = np.dot(offset, offset) - radius**2
        discriminant = b * b - 4 * a * c
        t = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
        best = np.minimum(best, np.where((discriminant >= 0) & (t > 0), t, np.inf))
    return np.where(np.isfinite(best), best, 0).astype(np.float32)


def synthetic_sequence(frames=8, width=160, height=120):
    if frames < 1:
        raise ValueError("frames must be positive")
    camera = Camera(width, height, width * 0.9, width * 0.9, (width - 1) / 2, (height - 1) / 2)
    result = []
    for i in range(frames):
        angle = i * 0.002
        pose = np.eye(4)
        pose[:3, :3] = [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
        pose[:3, 3] = [0.006 * i, 0.002 * np.sin(i * 0.5), 0.003 * i]
        depth = render_depth(camera, pose)
        rgb = np.stack(
            (
                np.clip(depth * 50, 0, 255),
                np.full_like(depth, 130),
                np.clip(255 - depth * 35, 0, 255),
            ),
            axis=-1,
        ).astype(np.uint8)
        result.append(Frame(i / 30, depth, rgb, pose))
    return camera, result
