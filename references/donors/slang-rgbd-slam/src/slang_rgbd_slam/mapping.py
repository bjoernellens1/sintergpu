"""Bounded CPU sparse voxel map. Host insertion occurs once per keyframe."""

from pathlib import Path

import numpy as np


class VoxelMap:
    def __init__(self, voxel_size=0.03, max_voxels=200000):
        if (
            not np.isfinite(voxel_size)
            or voxel_size <= 0
            or isinstance(max_voxels, (bool, np.bool_))
            or not isinstance(max_voxels, (int, np.integer))
            or max_voxels <= 0
        ):
            raise ValueError("Invalid map limits")
        self.voxel_size = voxel_size
        self.max_voxels = int(max_voxels)
        self.keys = np.empty((0, 3), dtype=np.int64)
        self.sums = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.counts = np.empty(0, dtype=np.int64)
        self.dropped_voxels = 0

    def __len__(self):
        return len(self.keys)

    @property
    def points(self):
        return self.sums / self.counts[:, None]

    @property
    def rgb(self):
        return np.clip(np.rint(self.colors / self.counts[:, None]), 0, 255).astype(np.uint8)

    def integrate(self, depth, camera, world_from_camera, rgb=None):
        depth = np.asarray(depth)
        if depth.shape != (camera.height, camera.width):
            raise ValueError("Depth shape differs from camera")
        if rgb is not None and np.asarray(rgb).shape != (*depth.shape, 3):
            raise ValueError("Invalid RGB shape")
        pose = np.asarray(world_from_camera)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid pose")
        v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
        z = depth[v, u]
        points = np.stack(
            ((u - camera.cx) * z / camera.fx, (v - camera.cy) * z / camera.fy, z), axis=1
        )
        points = points @ pose[:3, :3].T + pose[:3, 3]
        colors = np.full_like(points, 180) if rgb is None else np.asarray(rgb)[v, u].astype(float)
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        all_keys = np.concatenate((self.keys, keys))
        unique, inv = np.unique(all_keys, axis=0, return_inverse=True)
        sums = np.zeros((len(unique), 3))
        cols = np.zeros_like(sums)
        counts = np.zeros(len(unique), dtype=np.int64)
        np.add.at(sums, inv, np.concatenate((self.sums, points)))
        np.add.at(cols, inv, np.concatenate((self.colors, colors)))
        np.add.at(counts, inv, np.concatenate((self.counts, np.ones(len(points), dtype=np.int64))))
        # Keep lexicographically first cells, independent of Python hash/order.
        self.dropped_voxels += max(0, len(unique) - self.max_voxels)
        n = self.max_voxels
        self.keys, self.sums, self.colors, self.counts = unique[:n], sums[:n], cols[:n], counts[:n]
        return len(points)

    def export_ply(self, path):
        header = f"ply\nformat ascii 1.0\nelement vertex {len(self)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        with Path(path).open("w") as f:
            f.write(header)
            for p, c in zip(self.points, self.rgb):
                f.write(" ".join([*(f"{x:.6f}" for x in p), *(str(int(x)) for x in c)]) + "\n")
