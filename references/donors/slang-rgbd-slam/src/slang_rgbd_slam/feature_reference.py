"""Independent NumPy feature and calibrated reprojection numerical references.

This module imports no GPU runtime. Reprojection poses transform input landmark
coordinates into the current camera, with left-multiplied SE(3) increments.
"""

from time import perf_counter

import numpy as np

from .cpu import _exp, _pose
from .tracker import Camera, TrackingResult


def optimize_reprojection(
    points,
    pixels,
    camera,
    initial_pose=None,
    *,
    iterations=10,
    huber_delta=3.0,
    damping=1e-4,
    min_correspondences=12,
):
    """Robust calibrated 2D/3D pose optimization from a supplied geometric estimate.

    The default estimate is identity, never a ground-truth pose. Residuals are
    predicted minus observed pixel coordinates; one Huber weight is assigned to
    each observation's Euclidean pixel error. CPU normal equations use float64.
    """
    points, pixels = np.asarray(points, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or pixels.shape != (len(points), 2)
        or not np.isfinite(points).all()
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("points must be finite Nx3 and pixels finite Nx2")
    if not isinstance(camera, Camera):
        raise TypeError("camera must be Camera")
    if (
        not isinstance(iterations, int)
        or iterations < 1
        or not isinstance(min_correspondences, int)
        or min_correspondences < 6
        or not np.isfinite([huber_delta, damping]).all()
        or huber_delta <= 0
        or damping < 0
    ):
        raise ValueError("invalid reprojection solver parameters")
    initial = _pose(np.eye(4) if initial_pose is None else initial_pose)
    pose, failed, count, cost = initial.copy(), False, 0, 0.0
    start = perf_counter()
    for _ in range(iterations):
        if failed:
            count, cost = 0, 0.0
            continue
        transformed = points @ pose[:3, :3].T + pose[:3, 3]
        valid = transformed[:, 2] > 1e-6
        p, target = transformed[valid], pixels[valid]
        count = len(p)
        if count < min_correspondences:
            pose, failed = initial.copy(), True
            continue
        x, y, z = p.T
        inverse_z = 1 / z
        projected = p[:, :2] * inverse_z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        residual = projected - target
        weight = np.sqrt(
            np.minimum(1, huber_delta / np.maximum(np.linalg.norm(residual, axis=1), 1e-20))
        )
        # d projection / d left twist [rotation xyz, translation xyz].
        jacobian = np.zeros((count, 2, 6))
        fx, fy = camera.fx, camera.fy
        jacobian[:, 0] = np.stack(
            (
                -fx * x * y * inverse_z**2,
                fx * (1 + x * x * inverse_z**2),
                -fx * y * inverse_z,
                fx * inverse_z,
                np.zeros(count),
                -fx * x * inverse_z**2,
            ),
            axis=1,
        )
        jacobian[:, 1] = np.stack(
            (
                -fy * (1 + y * y * inverse_z**2),
                fy * x * y * inverse_z**2,
                fy * x * inverse_z,
                np.zeros(count),
                fy * inverse_z,
                -fy * y * inverse_z**2,
            ),
            axis=1,
        )
        weighted_jacobian = (jacobian * weight[:, None, None]).reshape(-1, 6)
        weighted_residual = (residual * weight[:, None]).ravel()
        cost = float(weighted_residual @ weighted_residual)
        try:
            hessian = weighted_jacobian.T @ weighted_jacobian + damping * np.eye(6)
            gradient = weighted_jacobian.T @ weighted_residual
            factor = np.linalg.cholesky(hessian)
            delta = np.linalg.solve(factor.T, np.linalg.solve(factor, -gradient))
            pose = _exp(delta) @ pose
            if not np.isfinite(pose).all():
                pose, failed = initial.copy(), True
        except np.linalg.LinAlgError:
            pose, failed = initial.copy(), True
    return TrackingResult(
        pose,
        not failed,
        count,
        float(np.sqrt(cost / max(count, 1))),
        iterations,
        (perf_counter() - start) * 1000,
    )


_CIRCLE = np.array(
    [
        [0, -3],
        [1, -3],
        [2, -2],
        [3, -1],
        [3, 0],
        [3, 1],
        [2, 2],
        [1, 3],
        [0, 3],
        [-1, 3],
        [-2, 2],
        [-3, 1],
        [-3, 0],
        [-3, -1],
        [-2, -2],
        [-1, -3],
    ]
)
_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], np.uint8)


def grayscale_pyramid(rgb, levels=4):
    """Float32 grayscale and clamped binomial-five factor-two pyramid."""
    image = np.asarray(rgb, np.float32)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or not np.isfinite(image).all()
        or np.any(image < 0)
        or np.any(image > 255)
    ):
        raise ValueError("RGB must be finite HxWx3 in [0,255]")
    if not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be positive")
    gray = (
        image[:, :, 0] * np.float32(0.299)
        + image[:, :, 1] * np.float32(0.587)
        + image[:, :, 2] * np.float32(0.114)
    )
    result = [gray]
    coefficients = [1, 4, 6, 4, 1]
    for _ in range(levels - 1):
        height, width = gray.shape[0] // 2, gray.shape[1] // 2
        if min(height, width) < 1:
            raise ValueError("too many pyramid levels for image")
        down = np.zeros((height, width), np.float32)
        yy, xx = np.arange(height) * 2, np.arange(width) * 2
        for dy, wy in zip(range(-2, 3), coefficients):
            for dx, wx in zip(range(-2, 3), coefficients):
                down += gray[
                    np.clip(yy + dy, 0, len(gray) - 1)[:, None],
                    np.clip(xx + dx, 0, gray.shape[1] - 1)[None, :],
                ] * np.float32(wx * wy)
        gray = down / np.float32(256)
        result.append(gray)
    return result


def fast_harris_candidates(gray, threshold=20):
    """FAST-9 contrast score, deterministic NMS, then 5x5 Harris score."""
    gray = np.asarray(gray, np.float32)
    height, width = gray.shape
    if min(height, width) <= 38:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    center = gray[3:-3, 3:-3]
    ring = np.stack(
        [gray[3 + dy : height - 3 + dy, 3 + dx : width - 3 + dx] - center for dx, dy in _CIRCLE],
        axis=-1,
    )
    ring = np.concatenate((ring, ring[..., :8]), axis=-1)
    windows = np.lib.stride_tricks.sliding_window_view(ring, 9, axis=-1)
    contrast = np.maximum(windows.min(axis=-1), -windows.max(axis=-1)).max(axis=-1)
    score = np.zeros_like(gray)
    score[3:-3, 3:-3] = np.maximum(contrast, 0)
    score[:19] = score[-19:] = 0
    score[:, :19] = score[:, -19:] = 0
    valid = score > threshold
    # A tied FAST score belongs to the lowest row-major pixel.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbor = np.roll(score, (-dy, -dx), axis=(0, 1))
            valid &= score > neighbor if dy < 0 or (dy == 0 and dx < 0) else score >= neighbor
    valid[:19] = False
    valid[-19:] = False
    valid[:, :19] = False
    valid[:, -19:] = False
    yy, xx = np.nonzero(valid)
    if not len(xx):
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    xx_sum, yy_sum, xy_sum = (np.zeros(len(xx), np.float32) for _ in range(3))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            gx = (gray[yy + dy, xx + dx + 1] - gray[yy + dy, xx + dx - 1]) * 0.5
            gy = (gray[yy + dy + 1, xx + dx] - gray[yy + dy - 1, xx + dx]) * 0.5
            xx_sum += gx * gx
            yy_sum += gy * gy
            xy_sum += gx * gy
    harris = xx_sum * yy_sum - xy_sum**2 - 0.04 * (xx_sum + yy_sum) ** 2
    positive = harris > 0
    return np.stack((xx[positive], yy[positive]), axis=1), harris[positive]


def describe_features(gray, coordinates, pattern):
    """Intensity-centroid orientation and rotated 256-pair binary descriptors."""
    coordinates = np.asarray(coordinates, np.int32)
    pattern = np.asarray(pattern, np.int32)
    if coordinates.shape != (len(coordinates), 2) or pattern.shape != (256, 2, 2):
        raise ValueError("coordinates must be Nx2, pattern 256x2x2")
    if not len(coordinates):
        return np.zeros((0, 8), np.uint32)
    offsets = np.array(
        [[x, y] for y in range(-15, 16) for x in range(-15, 16) if x * x + y * y <= 225]
    )
    samples = gray[
        coordinates[:, None, 1] + offsets[None, :, 1], coordinates[:, None, 0] + offsets[None, :, 0]
    ]
    m10, m01 = (
        samples @ offsets[:, 0].astype(np.float32),
        samples @ offsets[:, 1].astype(np.float32),
    )
    angle = np.arctan2(m01, m10)
    cosine, sine = np.cos(angle), np.sin(angle)
    x = (
        pattern[None, :, :, 0] * cosine[:, None, None]
        - pattern[None, :, :, 1] * sine[:, None, None]
    )
    y = (
        pattern[None, :, :, 0] * sine[:, None, None]
        + pattern[None, :, :, 1] * cosine[:, None, None]
    )
    rotated = np.stack((np.floor(x + 0.5), np.floor(y + 0.5)), axis=-1).astype(np.int32)
    pixel = coordinates[:, None, None] + rotated
    intensity = gray[pixel[..., 1], pixel[..., 0]]
    bits = intensity[:, :, 0] < intensity[:, :, 1]
    descriptors = np.zeros((len(coordinates), 8), np.uint32)
    for bit in range(32):
        descriptors |= bits[:, bit::32].astype(np.uint32) << np.uint32(bit)
    return descriptors


def match_features(source, target, *, ratio=0.8, max_hamming=80):
    """Mutual nearest 256-bit Hamming matches with a strict forward ratio test."""
    a = np.asarray(source["descriptors"], np.uint32)
    b = np.asarray(target["descriptors"], np.uint32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (8,) or b.shape[1:] != (8,):
        raise ValueError("descriptors must be Nx8 uint32")
    if not np.isfinite(ratio) or not 0 < ratio <= 1 or not 0 <= max_hamming <= 256:
        raise ValueError("invalid matching thresholds")
    output = np.full(len(a), 0xFFFFFFFF, np.uint32)
    if not len(a) or len(b) < 2:
        return output
    distances = np.empty((len(a), len(b)), np.uint16)
    for begin in range(0, len(a), 64):
        xor = np.bitwise_xor(a[begin : begin + 64, None], b[None])
        distances[begin : begin + 64] = _POPCOUNT[xor.view(np.uint8)].sum(axis=-1)
    best = distances.argmin(axis=1)
    best_distance = distances[np.arange(len(a)), best]
    if len(b) > 1:
        second = np.partition(distances, 1, axis=1)[:, 1]
    else:
        second = np.full(len(a), 257)
    reverse = distances.argmin(axis=0)
    valid = (
        (best_distance <= max_hamming)
        & (best_distance < ratio * second)
        & (reverse[best] == np.arange(len(a)))
    )
    output[valid] = best[valid]
    return output


class CPUFeatures:
    """NumPy oracle for the bounded GPU feature frontend, with compact arrays."""

    def __init__(
        self,
        camera,
        *,
        max_features=1024,
        levels=4,
        fast_threshold=20,
        ratio=0.8,
        max_hamming=80,
        cell_quota=4,
    ):
        from .features import feature_brief_pattern, feature_grids

        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if (
            not isinstance(max_features, int)
            or isinstance(max_features, bool)
            or not 1 <= max_features <= 4096
            or not isinstance(levels, int)
            or not 1 <= levels <= 8
            or not isinstance(cell_quota, int)
            or not 1 <= cell_quota <= 4
            or not np.isfinite([fast_threshold, ratio]).all()
            or not 0 < fast_threshold < 255
            or not 0 < ratio < 1
            or not isinstance(max_hamming, int)
            or not 0 <= max_hamming <= 256
        ):
            raise ValueError("invalid feature configuration")
        if camera.width % 2 ** (levels - 1) or camera.height % 2 ** (levels - 1):
            raise ValueError("camera dimensions must be divisible by pyramid scale")
        self.camera, self.levels, self.max_features = camera, levels, max_features
        self.fast_threshold, self.ratio, self.max_hamming = fast_threshold, ratio, max_hamming
        self.grids = feature_grids(camera, max_features, levels, cell_quota)
        self.pattern = feature_brief_pattern().reshape(256, 2, 2)

    def extract(self, rgb, depth):
        depth = np.asarray(depth, np.float32)
        if (
            depth.shape != (self.camera.height, self.camera.width)
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
        ):
            raise ValueError("depth must be finite nonnegative HxW")
        if np.asarray(rgb).shape != (*depth.shape, 3):
            raise ValueError("RGB shape must match camera")
        pyramid = grayscale_pyramid(rgb, self.levels)
        positions, descriptors = [], []
        for level, (gray, (grid_x, grid_y, quota)) in enumerate(zip(pyramid, self.grids)):
            if not grid_x or not grid_y:
                continue
            coordinates, scores = fast_harris_candidates(gray, self.fast_threshold)
            scale = 2**level
            original = (coordinates.astype(np.float32) + 0.5) * scale - 0.5
            nearest = np.floor(original + 0.5).astype(np.int32)
            valid = depth[nearest[:, 1], nearest[:, 0]] > 0
            coordinates, scores, original, nearest = (
                a[valid] for a in (coordinates, scores, original, nearest)
            )
            selected = []
            height, width = gray.shape
            for gy in range(grid_y):
                for gx in range(grid_x):
                    x0, x1 = gx * width // grid_x, (gx + 1) * width // grid_x
                    y0, y1 = gy * height // grid_y, (gy + 1) * height // grid_y
                    in_cell = (
                        (coordinates[:, 0] >= x0)
                        & (coordinates[:, 0] < x1)
                        & (coordinates[:, 1] >= y0)
                        & (coordinates[:, 1] < y1)
                    )
                    candidate = np.flatnonzero(in_cell)
                    order = np.lexsort(
                        (
                            coordinates[candidate, 1] * width + coordinates[candidate, 0],
                            -scores[candidate],
                        )
                    )
                    selected.extend(candidate[order[:quota]])
            selected = np.asarray(selected, np.int64)
            if not len(selected):
                continue
            pixel = nearest[selected]
            positions.append(
                np.column_stack(
                    (
                        original[selected],
                        depth[pixel[:, 1], pixel[:, 0]],
                        np.full(len(selected), level, np.float32),
                    )
                )
            )
            descriptors.append(describe_features(gray, coordinates[selected], self.pattern))
        position = (
            np.concatenate(positions).astype(np.float32)
            if positions
            else np.empty((0, 4), np.float32)
        )
        descriptor = np.concatenate(descriptors) if descriptors else np.empty((0, 8), np.uint32)
        camera = self.camera
        landmarks = np.column_stack(
            (
                (position[:, 0] - camera.cx) * position[:, 2] / camera.fx,
                (position[:, 1] - camera.cy) * position[:, 2] / camera.fy,
                position[:, 2],
                np.ones(len(position), np.float32),
            )
        ).astype(np.float32)
        return dict(
            count=len(position), positions=position, descriptors=descriptor, landmarks=landmarks
        )

    def match(self, source, target):
        return match_features(source, target, ratio=self.ratio, max_hamming=self.max_hamming)
