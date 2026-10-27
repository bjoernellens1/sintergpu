"""TUM trajectory serialization and rigid (no scale) positional ATE."""

from pathlib import Path

import numpy as np


def quaternion(rotation):
    # Eigenvector form is stable near 180 degrees; canonical qw >= 0.
    r = np.asarray(rotation, dtype=float)
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[1, 0] + r[0, 1],
                    r[2, 0] + r[0, 2],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[1, 0] + r[0, 1],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[2, 1] + r[1, 2],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[2, 0] + r[0, 2],
                    r[2, 1] + r[1, 2],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, v = np.linalg.eigh(k)
    q = v[:, -1]
    return q if q[3] >= 0 else -q


def write_trajectory(path, trajectory):
    rows = []
    for timestamp, pose in trajectory:
        values = [timestamp, *np.asarray(pose)[:3, 3], *quaternion(np.asarray(pose)[:3, :3])]
        rows.append(" ".join(f"{x:.9f}" for x in values))
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""))


def read_trajectory(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        vals = np.asarray([float(v) for v in line.split()])
        if vals.shape != (8,) or not np.isfinite(vals).all():
            raise ValueError("Malformed TUM trajectory row")
        t, x, y, z, qx, qy, qz, qw = vals
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Zero quaternion")
        qx, qy, qz, qw = q / norm
        p = np.eye(4)
        p[:3, :3] = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
        p[:3, 3] = [x, y, z]
        result.append((t, p))
    return result


def match_timestamps(left, right, tolerance=0.02):
    """Closest-first deterministic one-to-one association."""
    import bisect

    right_times = [x[0] for x in right]
    candidates = []
    for i, (t, _) in enumerate(left):
        lo = bisect.bisect_left(right_times, t - tolerance)
        hi = bisect.bisect_right(right_times, t + tolerance)
        candidates.extend((abs(t - right_times[j]), i, j) for j in range(lo, hi))
    used_l, used_r, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_l and j not in used_r:
            pairs.append((i, j))
            used_l.add(i)
            used_r.add(j)
    return sorted(pairs)


def absolute_trajectory_error(estimated, ground_truth, tolerance=0.02):
    estimated = sorted(estimated, key=lambda x: x[0])
    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    pairs = match_timestamps(estimated, ground_truth, tolerance)
    if not pairs:
        return {"count": 0, "rmse_m": None, "alignment": "rigid_no_scale"}
    a = np.array([estimated[i][1][:3, 3] for i, j in pairs])
    b = np.array([ground_truth[j][1][:3, 3] for i, j in pairs])
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ d @ u.T
    error = ac @ r.T - bc
    return {
        "count": len(pairs),
        "rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "alignment": "rigid_no_scale",
    }
