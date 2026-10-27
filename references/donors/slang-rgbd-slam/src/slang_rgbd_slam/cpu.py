"""Pure NumPy reference for the same projective ICP and sparse TSDF as Vulkan.

No GPU package is imported here. Geometry and persistent map storage are float32;
normal equations and Cholesky solves use float64 to serve as a numerical reference.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from .tracker import Camera, TrackingResult


def _depth(depth, camera):
    array = np.asarray(depth)
    if array.dtype != np.float32 or array.shape != (camera.height, camera.width):
        raise ValueError("depth must be float32 HxW matching camera")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("depth must contain finite nonnegative metres")
    return np.ascontiguousarray(array).copy()


def _pose(pose):
    array = np.asarray(pose, dtype=np.float64)
    if (
        array.shape != (4, 4)
        or not np.isfinite(array).all()
        or not np.allclose(array[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(array[:3, :3].T @ array[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(array[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("pose must be a finite rigid 4x4 transform")
    return array.copy()


def _geometry(depth, camera, jump):
    y, x = np.indices(depth.shape, dtype=np.float32)
    vertices = np.stack(
        ((x - camera.cx) * depth / camera.fx, (y - camera.cy) * depth / camera.fy, depth), axis=-1
    )
    normals = np.zeros_like(vertices)
    cross = np.cross(
        vertices[1:-1, 2:] - vertices[1:-1, :-2], vertices[2:, 1:-1] - vertices[:-2, 1:-1]
    )
    lengths = np.linalg.norm(cross, axis=-1)
    valid = (
        (depth[1:-1, :-2] > 0)
        & (depth[1:-1, 2:] > 0)
        & (depth[:-2, 1:-1] > 0)
        & (depth[2:, 1:-1] > 0)
        & (abs(depth[1:-1, :-2] - depth[1:-1, 2:]) < jump)
        & (abs(depth[:-2, 1:-1] - depth[2:, 1:-1]) < jump)
        & (lengths > 1e-10)
    )
    normals[1:-1, 1:-1][valid] = cross[valid] / lengths[valid, None]
    normal_valid = np.zeros(depth.shape, bool)
    normal_valid[1:-1, 1:-1] = valid
    return vertices.reshape(-1, 3), normals.reshape(-1, 3), normal_valid.ravel()


def _exp(delta):
    w, v = delta[:3], delta[3:]
    theta2 = w @ w
    theta = np.sqrt(theta2)
    a = 1 - theta2 / 6 if theta2 < 1e-8 else np.sin(theta) / theta
    b = 0.5 - theta2 / 24 if theta2 < 1e-8 else (1 - np.cos(theta)) / theta2
    c = 1 / 6 - theta2 / 120 if theta2 < 1e-8 else (theta - np.sin(theta)) / (theta2 * theta)
    skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    result = np.eye(4)
    result[:3, :3] += a * skew + b * skew @ skew
    result[:3, 3] = v + b * np.cross(w, v) + c * np.cross(w, np.cross(w, v))
    return result


class CPUTracker:
    """Matched CPU implementation of VulkanTracker's fixed-iteration ICP."""

    def __init__(
        self,
        camera,
        levels=3,
        iterations=(8, 5, 4),
        *,
        distance_threshold=0.1,
        normal_threshold=0.7,
        huber_delta=0.02,
        damping=1e-4,
        min_correspondences=20,
        depth_jump=0.1,
    ):
        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if (
            not isinstance(levels, int)
            or levels < 1
            or len(iterations) != levels
            or any(not isinstance(n, int) or n < 1 for n in iterations)
        ):
            raise ValueError("iterations must contain one positive integer per pyramid level")
        scale = 2 ** (levels - 1)
        if camera.width % scale or camera.height % scale:
            raise ValueError("dimensions must be divisible by the pyramid scale")
        if min(camera.width, camera.height) // scale < 4:
            raise ValueError("coarsest camera must have at least four pixels per dimension")
        if (
            not np.isfinite(
                [distance_threshold, normal_threshold, huber_delta, damping, depth_jump]
            ).all()
            or distance_threshold <= 0
            or huber_delta <= 0
            or damping < 0
            or depth_jump <= 0
            or not 0 <= normal_threshold <= 1
            or not isinstance(min_correspondences, int)
            or min_correspondences < 6
        ):
            raise ValueError("invalid ICP filtering, damping, or correspondence parameters")
        self.camera, self.iterations = camera, tuple(iterations)
        self.distance, self.normal_cos, self.huber = (
            distance_threshold,
            normal_threshold,
            huber_delta,
        )
        self.minimum, self.depth_jump, self.damping = min_correspondences, depth_jump, damping
        self.cameras = [
            Camera(
                camera.width // 2**i,
                camera.height // 2**i,
                camera.fx / 2**i,
                camera.fy / 2**i,
                (camera.cx + 0.5) / 2**i - 0.5,
                (camera.cy + 0.5) / 2**i - 0.5,
            )
            for i in range(levels)
        ]
        self.reference = None

    def _maps(self, depth):
        maps = []
        for i, camera in enumerate(self.cameras):
            if i:
                tile = depth.reshape(camera.height, 2, camera.width, 2).transpose(0, 2, 1, 3)
                tile = tile.reshape(camera.height, camera.width, 4)
                valid = tile > 0
                count = valid.sum(axis=-1)
                lo = np.where(valid, tile, np.inf).min(axis=-1)
                hi = tile.max(axis=-1)
                depth = np.where(
                    (count > 0) & (hi - lo <= self.depth_jump),
                    tile.sum(axis=-1) / np.maximum(count, 1),
                    0,
                ).astype(np.float32)
            maps.append(_geometry(depth, camera, self.depth_jump))
        return maps

    def set_reference(self, depth):
        self.reference = self._maps(_depth(depth, self.camera))

    def track(self, depth, initial_pose=None):
        if self.reference is None:
            raise RuntimeError("set_reference must precede track")
        start = perf_counter()
        source = self._maps(_depth(depth, self.camera))
        initial = _pose(np.eye(4) if initial_pose is None else initial_pose)
        pose = initial.copy()
        failed, count, cost = False, 0, 0.0
        for i, iterations in zip(reversed(range(len(self.cameras))), self.iterations):
            camera = self.cameras[i]
            vertices, normals, normal_valid = source[i]
            ref_vertices, ref_normals, ref_normal_valid = self.reference[i]
            for _ in range(iterations):
                if failed:
                    count, cost = 0, 0.0
                    continue
                p = vertices @ pose[:3, :3].T + pose[:3, 3]
                sn = normals @ pose[:3, :3].T
                valid = (vertices[:, 2] > 0) & normal_valid & (p[:, 2] > 0)
                candidates = np.flatnonzero(valid)
                projected = p[candidates]
                pixels = np.rint(
                    projected[:, :2] / projected[:, 2, None] * [camera.fx, camera.fy]
                    + [camera.cx, camera.cy]
                ).astype(np.int64)
                inbounds = (
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < camera.width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < camera.height)
                )
                candidates, pixels = candidates[inbounds], pixels[inbounds]
                ref_indices = pixels[:, 1] * camera.width + pixels[:, 0]
                q, n = ref_vertices[ref_indices], ref_normals[ref_indices]
                p = p[candidates]
                valid = (
                    (q[:, 2] > 0)
                    & ref_normal_valid[ref_indices]
                    & (np.linalg.norm(p - q, axis=1) < self.distance)
                    & (np.sum(sn[candidates] * n, axis=1) > self.normal_cos)
                )
                p, q, n = p[valid], q[valid], n[valid]
                count = len(p)
                residual = np.sum(n * (p - q), axis=1)
                sqrt_weight = np.sqrt(np.minimum(1, self.huber / np.maximum(abs(residual), 1e-20)))
                weighted_normal = n * sqrt_weight[:, None]
                jacobian = np.concatenate((np.cross(p, weighted_normal), weighted_normal), axis=1)
                residual *= sqrt_weight
                cost = float(residual @ residual)
                if count < self.minimum:
                    failed = True
                    pose = initial.copy()
                    continue
                try:
                    hessian = jacobian.T @ jacobian + self.damping * np.eye(6)
                    cholesky = np.linalg.cholesky(hessian)
                    delta = np.linalg.solve(
                        cholesky.T, np.linalg.solve(cholesky, -(jacobian.T @ residual))
                    )
                    pose = _exp(delta) @ pose
                    if not np.isfinite(pose).all():
                        failed = True
                        pose = initial.copy()
                except np.linalg.LinAlgError:
                    failed = True
                    pose = initial.copy()
        return TrackingResult(
            pose,
            not failed,
            count,
            np.sqrt(cost / max(count, 1)),
            sum(self.iterations),
            (perf_counter() - start) * 1000,
        )


_OFFSETS = np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij"), axis=-1)[
    ..., ::-1
].reshape(-1, 3)
_CORNERS = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)])


def _keys(coords):
    shifted = coords.astype(np.int64) + 512
    return shifted[..., 0] | (shifted[..., 1] << 10) | (shifted[..., 2] << 20)


class CPUTSDF:
    """Sparse 8³-block projective TSDF with vectorized allocation and fusion.

    Blocks have a bounded signed ten-bit coordinate domain. Coordinates outside
    it and allocation overflow are counted, never aliased or evicted.
    """

    def __init__(
        self,
        camera,
        *,
        voxel_size=0.03,
        truncation_voxels=4,
        max_blocks=8192,
        max_weight=100,
        near=0.1,
        far=8.0,
        ray_step=None,
    ):
        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if (
            not np.isfinite([voxel_size, truncation_voxels, max_weight, near, far]).all()
            or voxel_size <= 0
            or truncation_voxels <= 0
            or max_weight <= 0
            or near <= 0
            or far <= near
            or not isinstance(max_blocks, int)
            or isinstance(max_blocks, bool)
            or not 1 <= max_blocks <= 32767
        ):
            raise ValueError("invalid TSDF configuration")
        self.camera, self.voxel_size, self.max_blocks = camera, float(voxel_size), max_blocks
        self.mu, self.max_weight = float(voxel_size * truncation_voxels), float(max_weight)
        self.near, self.far = float(near), float(far)
        self.ray_step = float(voxel_size * 0.5 if ray_step is None else ray_step)
        if not np.isfinite(self.ray_step) or self.ray_step <= 0:
            raise ValueError("ray_step must be finite and positive")
        self.coords = np.empty((max_blocks, 3), np.int32)
        self.distances = np.ones((max_blocks, 512), np.float32)
        self.weights = np.zeros((max_blocks, 512), np.float32)
        self.colors = np.zeros((max_blocks, 512, 3), np.float32)
        self.color_weights = np.zeros((max_blocks, 512), np.float32)
        self.count, self.overflow, self.out_of_bounds = 0, 0, 0
        self.active, self.surfaces = 0, 0
        self._sorted_keys = np.empty(0, np.int64)
        self._sorted_indices = np.empty(0, np.int64)
        y, x = np.indices((camera.height, camera.width), dtype=np.float32)
        self.rays = np.stack(
            ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones_like(x)), axis=-1
        ).reshape(-1, 3)

    def stats(self):
        return dict(
            blocks=self.count,
            active=self.active,
            surfaces=self.surfaces,
            domain_errors=self.out_of_bounds,
            overflow=self.overflow,
            probe_exhaustion=0,
        )

    def __len__(self):
        return int(np.count_nonzero(self.weights[: self.count]))

    def _lookup(self, coords):
        keys = _keys(coords)
        indices = np.searchsorted(self._sorted_keys, keys)
        valid = (
            (coords >= -512).all(axis=-1) & (coords <= 511).all(axis=-1) & (indices < self.count)
        )
        result = np.full(keys.shape, -1, np.int64)
        if self.count:
            indices = np.minimum(indices, self.count - 1)
            valid &= self._sorted_keys[indices] == keys
            result[valid] = self._sorted_indices[indices[valid]]
        return result

    def _allocate(self, depth, pose):
        selected = (depth.ravel() >= self.near) & (depth.ravel() <= self.far)
        z = depth.ravel()[selected]
        rays = self.rays[selected]
        start_z = np.maximum(self.near, z - self.mu)
        end_z = z + self.mu
        valid = end_z >= start_z
        rays, start_z, end_z = rays[valid], start_z[valid], end_z[valid]
        start = (rays * start_z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
        end = (rays * end_z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
        block_size = self.voxel_size * 8
        cell = np.floor(start / block_size).astype(np.int32)
        target = np.floor(end / block_size).astype(np.int32)
        direction = end - start
        step = np.sign(direction).astype(np.int32)
        boundary = (cell + (step > 0)) * block_size
        with np.errstate(divide="ignore", invalid="ignore"):
            tmax = np.where(direction != 0, (boundary - start) / direction, np.inf)
            tdelta = np.where(direction != 0, block_size / abs(direction), np.inf)
        # Each iteration advances one axis, including zero-length boundary ties,
        # matching shader DDA. Work is over all rays rather than individual rays.
        visited = []
        active = np.ones(len(cell), bool)
        max_steps = int(np.abs(target - cell).sum(axis=1).max(initial=0)) + 1
        for _ in range(max_steps):
            visited.append(cell[active].copy())
            active &= (cell != target).any(axis=1)
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            axis = np.argmin(tmax[indices], axis=1)
            cell[indices, axis] += step[indices, axis]
            tmax[indices, axis] += tdelta[indices, axis]
        if not visited:
            return np.empty(0, np.int64)
        coords = np.unique(np.concatenate(visited), axis=0)
        valid = (coords >= -512).all(axis=1) & (coords <= 511).all(axis=1)
        self.out_of_bounds += int((~valid).sum())
        active_coords = coords[valid]
        coords = active_coords[self._lookup(active_coords) < 0]
        capacity = self.max_blocks - self.count
        self.overflow += max(0, len(coords) - capacity)
        coords = coords[:capacity]
        self.coords[self.count : self.count + len(coords)] = coords
        self.count += len(coords)
        keys = _keys(self.coords[: self.count])
        self._sorted_indices = np.argsort(keys)
        self._sorted_keys = keys[self._sorted_indices]
        active_indices = self._lookup(active_coords)
        return active_indices[active_indices >= 0]

    def integrate(self, depth, rgb, pose):
        depth, pose = _depth(depth, self.camera), _pose(pose).astype(np.float32)
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.float32)
            if (
                rgb.shape != (*depth.shape, 3)
                or not np.isfinite(rgb).all()
                or np.any(rgb < 0)
                or np.any(rgb > 255)
            ):
                raise ValueError("RGB must have HxWx3 shape and values in [0,255]")
        active = self._allocate(depth, pose)
        self.active = len(active)
        inverse_rotation = pose[:3, :3]
        # Bounded chunks avoid temporaries proportional to maximum map capacity.
        for begin in range(0, len(active), 256):
            blocks = active[begin : begin + 256]
            world = (
                (self.coords[blocks, None, :] * 8 + _OFFSETS[None]).astype(np.float32) + 0.5
            ) * self.voxel_size
            p = (world - pose[:3, 3]) @ inverse_rotation
            positive = p[..., 2] > 0
            u = np.floor(
                self.camera.fx * p[..., 0] / np.maximum(p[..., 2], 1e-20) + self.camera.cx + 0.5
            ).astype(np.int64)
            v = np.floor(
                self.camera.fy * p[..., 1] / np.maximum(p[..., 2], 1e-20) + self.camera.cy + 0.5
            ).astype(np.int64)
            valid = (
                positive & (u >= 0) & (u < self.camera.width) & (v >= 0) & (v < self.camera.height)
            )
            rows, cols = np.nonzero(valid)
            observed = depth[v[rows, cols], u[rows, cols]]
            sdf = observed - p[rows, cols, 2]
            accepted = (observed >= self.near) & (observed <= self.far) & (sdf >= -self.mu)
            rows, cols, observed, sdf = (a[accepted] for a in (rows, cols, observed, sdf))
            confidence = 1 / (1 + observed**2)
            old = self.weights[blocks[rows], cols]
            denominator = old + confidence
            self.distances[blocks[rows], cols] = (
                old * self.distances[blocks[rows], cols] + confidence * np.minimum(1, sdf / self.mu)
            ) / denominator
            self.weights[blocks[rows], cols] = np.minimum(self.max_weight, denominator)
            if len(rows):
                surface = abs(sdf) <= self.mu
                rr, cc = rows[surface], cols[surface]
                color = (
                    rgb[v[rr, cc], u[rr, cc]].astype(np.float32)
                    if rgb is not None
                    else np.full((len(rr), 3), 180, np.float32)
                )
                old_color = self.color_weights[blocks[rr], cc]
                color_denominator = old_color + confidence[surface]
                self.colors[blocks[rr], cc] = (
                    old_color[:, None] * self.colors[blocks[rr], cc]
                    + confidence[surface, None] * color
                ) / color_denominator[:, None]
                self.color_weights[blocks[rr], cc] = np.minimum(self.max_weight, color_denominator)

    def _sample(self, world):
        position = world / self.voxel_size - 0.5
        base = np.floor(position).astype(np.int32)
        fraction = position - base.astype(np.float32)
        voxel = base[:, None, :] + _CORNERS
        block = np.floor_divide(voxel, 8)
        index = self._lookup(block)
        local = np.mod(voxel, 8)
        linear = local[..., 0] + 8 * local[..., 1] + 64 * local[..., 2]
        safe_index = np.maximum(index, 0)
        weight = self.weights[safe_index, linear]
        valid = ((index >= 0) & (weight > 0)).all(axis=1)
        coefficient = np.prod(
            np.where(_CORNERS[None] == 1, fraction[:, None], 1 - fraction[:, None]), axis=2
        )
        distance = np.sum(self.distances[safe_index, linear] * coefficient, axis=1)
        return distance, valid

    def raycast(self, pose):
        pose = _pose(pose).astype(np.float32)
        output = np.zeros(len(self.rays), np.float32)
        if not self.count:
            return output.reshape(self.camera.height, self.camera.width)
        direction = self.rays @ pose[:3, :3].T
        step = self.ray_step / np.linalg.norm(direction, axis=1)
        t = np.full(len(direction), self.near, np.float32)
        previous = np.zeros(len(direction), np.float32)
        previous_t = np.full(len(direction), self.near, np.float32)
        previous_valid = np.zeros(len(direction), bool)
        indices = np.arange(len(direction))
        limit = int(np.ceil((self.far - self.near) / step.min())) + self.max_blocks * 2
        for _ in range(limit):
            indices = indices[t[indices] <= self.far]
            if not len(indices):
                break
            world = direction[indices] * t[indices, None] + pose[:3, 3]
            block = np.floor(world / (self.voxel_size * 8)).astype(np.int32)
            allocated = self._lookup(block) >= 0
            missing = indices[~allocated]
            if len(missing):
                bound = (block[~allocated] + (direction[missing] > 0)) * self.voxel_size * 8
                with np.errstate(divide="ignore", invalid="ignore"):
                    crossing = np.where(
                        abs(direction[missing]) > 1e-20,
                        (bound - pose[:3, 3]) / direction[missing],
                        1e30,
                    )
                t[missing] = np.maximum(t[missing] + 1e-5, crossing.min(axis=1) + 1e-5)
                previous_valid[missing] = False
            found = indices[allocated]
            if len(found):
                distance, valid = self._sample(world[allocated])
                hit = previous_valid[found] & valid & (previous[found] > 0) & (distance <= 0)
                hit_indices = found[hit]
                fraction = previous[hit_indices] / (previous[hit_indices] - distance[hit])
                output[hit_indices] = previous_t[hit_indices] + fraction * (
                    t[hit_indices] - previous_t[hit_indices]
                )
                previous[found], previous_valid[found], previous_t[found] = (
                    distance,
                    valid,
                    t[found],
                )
                t[found] += step[found]
                t[hit_indices] = self.far + 1
        return output.reshape(self.camera.height, self.camera.width)

    def snapshot(self):
        order = self._sorted_indices
        return dict(
            keys=self.coords[: self.count][order].copy(),
            distances=self.distances[: self.count][order].copy(),
            weights=self.weights[: self.count][order].copy(),
            colors=self.colors[: self.count][order].copy(),
            colorWeights=self.color_weights[: self.count][order].copy(),
        )

    def export_ply(self, path):
        # Surface samples on positive local lattice edges; neighbors can lie in
        # another block. One sample per edge, interpolated at its zero crossing.
        points, colors = [], []
        for begin in range(0, self.count, 256):
            end = min(begin + 256, self.count)
            voxels = (self.coords[begin:end, None] * 8 + _OFFSETS).reshape(-1, 3)
            values = self.distances[begin:end].ravel()
            observed = self.weights[begin:end].ravel() > 0
            own_color = self.colors[begin:end].reshape(-1, 3)
            for axis in range(3):
                neighbor = voxels.copy()
                neighbor[:, axis] += 1
                block_index = self._lookup(np.floor_divide(neighbor, 8))
                local = neighbor % 8
                linear = local[:, 0] + 8 * local[:, 1] + 64 * local[:, 2]
                safe_index = np.maximum(block_index, 0)
                other = self.distances[safe_index, linear]
                valid = observed & (block_index >= 0) & (self.weights[safe_index, linear] > 0)
                valid &= (values * other <= 0) & (values != other)
                selected = np.flatnonzero(valid)
                position = voxels[selected].astype(np.float64) + 0.5
                position[:, axis] += values[selected] / (values[selected] - other[selected])
                points.append(position * self.voxel_size)
                alpha = values[selected] / (values[selected] - other[selected])
                colors.append(
                    own_color[selected] * (1 - alpha[:, None])
                    + self.colors[safe_index[selected], linear[selected]] * alpha[:, None]
                )
        points = np.concatenate(points) if points else np.empty((0, 3))
        colors = np.rint(
            np.clip(np.concatenate(colors) if colors else np.empty((0, 3)), 0, 255)
        ).astype(np.uint8)
        self.surfaces = len(points)
        path = Path(path)
        with path.open("w") as stream:
            stream.write(
                f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            )
            for point, color in zip(points, colors):
                stream.write(" ".join(map(str, [*point, *color])) + "\n")
        return len(points)


class CPUSLAM:
    """Matched frame-to-model CPU SLAM, integrating every successfully tracked frame."""

    def __init__(self, camera, *, tracker_options=None, map_options=None):
        self.camera = camera
        self.tracker = CPUTracker(camera, **(tracker_options or {}))
        self.map = CPUTSDF(camera, **(map_options or {}))
        self.world_from_camera = np.eye(4)
        self.initialized = False
        self.trajectory, self.results = [], []
        self.index, self.keyframes, self.last_timestamp = 0, 0, None

    def process(self, depth, rgb=None, *, timestamp=0.0):
        depth = _depth(depth, self.camera)
        if not np.isfinite(timestamp) or (
            self.last_timestamp is not None and timestamp <= self.last_timestamp
        ):
            raise ValueError("timestamps must be finite and increase")
        if rgb is not None and (
            np.asarray(rgb).shape != (*depth.shape, 3) or np.asarray(rgb).dtype != np.uint8
        ):
            raise ValueError("RGB must be HxWx3 uint8")
        start = perf_counter()
        was_initialized = self.initialized
        model = self.map.raycast(self.world_from_camera) if was_initialized else depth
        self.tracker.set_reference(model)
        tracking = self.tracker.track(depth)
        result = dict(
            timestamp=float(timestamp),
            success=bool(tracking.success),
            status="tracked"
            if was_initialized and tracking.success
            else "initialized"
            if tracking.success
            else "lost",
            world_from_camera=None,
            keyframe=False,
            correspondences=tracking.correspondences,
            rmse=float(tracking.rmse),
            tracking_ms=tracking.elapsed_ms,
        )
        if tracking.success:
            self.world_from_camera = self.world_from_camera @ tracking.reference_from_current
            self.map.integrate(depth, rgb, self.world_from_camera)
            self.initialized = True
            self.keyframes += 1
            result["world_from_camera"] = self.world_from_camera.copy()
            result["keyframe"] = True
            self.trajectory.append((float(timestamp), self.world_from_camera.copy()))
        result["mapping"] = self.map.stats()
        result["mapping_success"] = (
            result["mapping"]["overflow"] == 0 and result["mapping"]["domain_errors"] == 0
        )
        result["elapsed_ms"] = (perf_counter() - start) * 1000
        self.results.append(result)
        self.index += 1
        self.last_timestamp = float(timestamp)
        return result
