"""TUM RGB-D input; callers must supply calibrated camera intrinsics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .trajectory import match_timestamps, read_trajectory


@dataclass
class Frame:
    timestamp: float
    depth: np.ndarray
    rgb: np.ndarray | None = None
    world_from_camera: np.ndarray | None = None


def _index(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Malformed index: {path}")
        t = float(fields[0])
        if not np.isfinite(t):
            raise ValueError("Nonfinite timestamp")
        rows.append((t, fields[1]))
    return sorted(rows)


class TUMDataset:
    def __init__(self, root, camera, *, depth_scale=5000.0, tolerance=0.02):
        self.root = Path(root)
        self.camera = camera
        self.depth_scale = float(depth_scale)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        association = next(
            (
                self.root / n
                for n in ("associations.txt", "association.txt")
                if (self.root / n).exists()
            ),
            None,
        )
        self.records = []
        if association is not None:
            used_rgb, used_depth = set(), set()
            for line in association.read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                f = line.split()
                if len(f) != 4:
                    raise ValueError("Malformed association row")
                rt, dt = float(f[0]), float(f[2])
                if not np.isfinite([rt, dt]).all():
                    raise ValueError("Nonfinite timestamp")
                if abs(rt - dt) > tolerance:
                    continue
                if rt in used_rgb or dt in used_depth:
                    raise ValueError("Association repeats timestamp")
                used_rgb.add(rt)
                used_depth.add(dt)
                self.records.append((dt, f[3], f[1]))
        else:
            depth = _index(self.root / "depth.txt")
            if (self.root / "rgb.txt").exists():
                rgb = _index(self.root / "rgb.txt")
                self.records = [
                    (depth[i][0], depth[i][1], rgb[j][1])
                    for i, j in match_timestamps(depth, rgb, tolerance)
                ]
            else:
                self.records = [(t, p, None) for t, p in depth]
        self.records.sort()
        for _, d, r in self.records:
            for p in (d, r):
                if p is not None and not (self.root / p).is_file():
                    raise FileNotFoundError(self.root / p)
        self.ground_truth = (
            read_trajectory(self.root / "groundtruth.txt")
            if (self.root / "groundtruth.txt").exists()
            else None
        )

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        from PIL import Image

        t, d, r = self.records[i]
        try:
            with Image.open(self.root / d) as im:
                raw = np.array(im)
            rgb = None
            if r is not None:
                with Image.open(self.root / r) as im:
                    rgb = np.array(im.convert("RGB"))
        except (OSError, ValueError) as e:
            raise ValueError(f"Cannot decode RGB-D frame {t}: {e}") from e
        shape = (self.camera.height, self.camera.width)
        if raw.ndim != 2 or raw.shape != shape or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"Depth must be integer PNG of shape {shape}")
        if np.any(raw < 0):
            raise ValueError("Negative depth sample")
        if rgb is not None and rgb.shape != (*shape, 3):
            raise ValueError("RGB shape differs from camera")
        return Frame(t, raw.astype(np.float32) / self.depth_scale, rgb)
