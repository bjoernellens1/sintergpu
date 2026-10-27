"""NumPy-only dense RGB-D, keyframe BA and sparse geometric loop reference.

No GPU runtime or buffers are instantiated. Geometry storage is float32 and
linear solves use float64. The bounded keyframe map follows the GPU policy.
"""

from time import perf_counter

import numpy as np

from .cpu import _CORNERS, CPUTSDF, CPUTracker, _depth, _exp, _geometry, _pose
from .dense_ba import schur_step
from .feature_reference import CPUFeatures, optimize_reprojection
from .loop_verification import ransac_reference
from .tracker import TrackingResult
from .trajectory import quaternion


def intensity(rgb):
    return (np.asarray(rgb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255


def bilinear(image, uv):
    x, y = np.floor(uv).astype(int).T
    inside = (x >= 0) & (y >= 0) & (x + 1 < image.shape[1]) & (y + 1 < image.shape[0])
    x = np.clip(x, 0, image.shape[1] - 2)
    y = np.clip(y, 0, image.shape[0] - 2)
    a, b, c, d = image[y, x], image[y, x + 1], image[y + 1, x], image[y + 1, x + 1]
    dx, dy = (uv - np.column_stack((x, y))).T
    value = (a * (1 - dx) + b * dx) * (1 - dy) + (c * (1 - dx) + d * dx) * dy
    good = inside & (np.minimum.reduce((a, b, c, d)) >= 0)
    return np.where(good, value, -1), np.column_stack(
        ((b - a) * (1 - dy) + (d - c) * dy, (c - a) * (1 - dx) + (d - b) * dx)
    )


class CPUFusedTracker(CPUTracker):
    def __init__(self, camera, *, photo_weight=0.03, photo_huber=0.1, **options):
        options.pop("group_size", None)
        options.pop("reduction", None)
        super().__init__(camera, **options)
        if (
            not np.isfinite([photo_weight, photo_huber]).all()
            or photo_weight < 0
            or photo_huber <= 0
        ):
            raise ValueError("invalid photometric weights")
        self.photo_weight, self.photo_huber = photo_weight, photo_huber

    def intensity_pyramid(self, gray):
        result = [np.asarray(gray, np.float32).copy()]
        for camera in self.cameras[1:]:
            t = result[-1].reshape(camera.height, 2, camera.width, 2).transpose(0, 2, 1, 3)
            result.append(
                np.where((t < 0).any(axis=(2, 3)), -1, t.mean(axis=(2, 3))).astype(np.float32)
            )
        return result

    def set_reference(self, depth, gray):
        super().set_reference(depth)
        self.reference_gray = self.intensity_pyramid(gray)

    def track(self, depth, rgb, initial_pose=None):
        if self.reference is None:
            raise RuntimeError("set_reference must precede track")
        start = perf_counter()
        source = self._maps(_depth(depth, self.camera))
        gray = self.intensity_pyramid(intensity(rgb))
        initial = _pose(np.eye(4) if initial_pose is None else initial_pose)
        pose = initial.copy()
        failed = False
        count = 0
        cost = 0.0
        for i, iterations in zip(reversed(range(len(self.cameras))), self.iterations):
            camera = self.cameras[i]
            v, sn, nv = source[i]
            qn, nn, rv = self.reference[i]
            for _ in range(iterations):
                if failed:
                    continue
                pp = v @ pose[:3, :3].T + pose[:3, 3]
                rotated = sn @ pose[:3, :3].T
                ids = np.flatnonzero((v[:, 2] > 0) & nv & (pp[:, 2] > 0))
                p = pp[ids]
                uv = p[:, :2] / p[:, 2, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
                pix = np.rint(uv).astype(int)
                keep = (
                    (pix[:, 0] >= 0)
                    & (pix[:, 0] < camera.width)
                    & (pix[:, 1] >= 0)
                    & (pix[:, 1] < camera.height)
                )
                ids, p, uv, pix = ids[keep], p[keep], uv[keep], pix[keep]
                rid = pix[:, 1] * camera.width + pix[:, 0]
                q, n = qn[rid], nn[rid]
                keep = (
                    (q[:, 2] > 0)
                    & rv[rid]
                    & (np.linalg.norm(p - q, axis=1) < self.distance)
                    & ((rotated[ids] * n).sum(axis=1) > self.normal_cos)
                )
                ids, p, q, n, uv = ids[keep], p[keep], q[keep], n[keep], uv[keep]
                count = len(p)
                r = (n * (p - q)).sum(axis=1)
                w = np.sqrt(np.minimum(1, self.huber / np.maximum(abs(r), 1e-20)))
                j = np.concatenate((np.cross(p, n), n), axis=1) * w[:, None]
                r = r * w
                if self.photo_weight:
                    ref = self.reference_gray[i]
                    samples = [
                        bilinear(ref, uv + offset)[0]
                        for offset in ([0, 0], [1, 0], [-1, 0], [0, 1], [0, -1])
                    ]
                    valid = np.minimum.reduce(samples) >= 0
                    valid &= (
                        (uv[:, 0] >= 2)
                        & (uv[:, 0] < camera.width - 3)
                        & (uv[:, 1] >= 2)
                        & (uv[:, 1] < camera.height - 3)
                    )
                    z = p[:, 2]
                    gx = (samples[1] - samples[2]) * 0.5
                    gy = (samples[3] - samples[4]) * 0.5
                    pn = np.column_stack(
                        (
                            gx * camera.fx / z,
                            gy * camera.fy / z,
                            -(gx * camera.fx * p[:, 0] + gy * camera.fy * p[:, 1]) / z**2,
                        )
                    )
                    pr = samples[0] - gray[i].ravel()[ids]
                    pw = self.photo_weight * np.sqrt(
                        np.minimum(1, self.photo_huber / np.maximum(abs(pr), 1e-20))
                    )
                    j = np.vstack(
                        (j, np.concatenate((np.cross(p, pn), pn), axis=1)[valid] * pw[valid, None])
                    )
                    r = np.r_[r, pr[valid] * pw[valid]]
                cost = float(r @ r)
                if count < self.minimum:
                    failed = True
                    continue
                try:
                    delta = np.linalg.solve(j.T @ j + self.damping * np.eye(6), -j.T @ r)
                    pose = _exp(delta) @ pose
                    failed = not np.isfinite(pose).all()
                except np.linalg.LinAlgError:
                    failed = True
        return TrackingResult(
            initial if failed else pose,
            not failed,
            count,
            np.sqrt(cost / max(count, 1)),
            sum(self.iterations),
            (perf_counter() - start) * 1000,
        )


class CPUColorTSDF(CPUTSDF):
    def reset(self):
        self.distances.fill(1)
        self.weights.fill(0)
        self.colors.fill(0)
        self.color_weights.fill(0)
        self.count = self.overflow = self.out_of_bounds = self.active = self.surfaces = 0
        self._sorted_keys = np.empty(0, np.int64)
        self._sorted_indices = np.empty(0, np.int64)

    def raycast_intensity(self, pose):
        depth = self.raycast(pose)
        vertices, _, _ = _geometry(depth, self.camera, 0.1)
        world = vertices @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]
        lattice = world / self.voxel_size - 0.5
        base = np.floor(lattice).astype(int)
        frac = lattice - base
        coords = base[:, None, :] + _CORNERS[None]
        blocks = self._lookup(np.floor_divide(coords, 8).reshape(-1, 3)).reshape(-1, 8)
        local = np.mod(coords, 8)
        offset = local[..., 0] + 8 * local[..., 1] + 64 * local[..., 2]
        safe = np.maximum(blocks, 0)
        valid = (
            (blocks >= 0)
            & (self.weights[safe, offset] > 0)
            & (self.color_weights[safe, offset] > 0)
        )
        w = np.prod(np.where(_CORNERS[None] > 0, frac[:, None, :], 1 - frac[:, None, :]), axis=2)
        color = (self.colors[safe, offset] * w[..., None]).sum(axis=1)
        gray = intensity(color)
        return depth, np.where(valid.all(axis=1) & (depth.ravel() > 0), gray, -1).reshape(
            depth.shape
        ).astype(np.float32)


def optimize_dense_window(
    camera, depths, grays, initial_poses, *, iterations=3, min_factor_ratio=0.8, initial_depths=None
):
    """Coupled anchored world poses/inverse-depth Schur, transactional final cost."""
    depths = np.asarray(depths, np.float32)
    grays = np.asarray(grays, np.float32)
    original = np.asarray(initial_poses, np.float64).copy()
    poses = original.copy()
    frames, height, width = depths.shape
    pixels = height * width
    dofs = 6 * (frames - 1)
    if (
        not 2 <= frames <= 4
        or original.shape != (frames, 4, 4)
        or grays.shape != depths.shape
        or (height, width) != (camera.height, camera.width)
    ):
        raise ValueError("invalid BA camera/window arrays")
    if not isinstance(iterations, int) or iterations < 1 or not 0 <= min_factor_ratio <= 1:
        raise ValueError("invalid BA iterations/support ratio")
    if not np.isfinite(depths).all() or np.any(depths < 0) or not np.isfinite(grays).all():
        raise ValueError("BA images must be finite and depths nonnegative")
    for pose in original:
        _pose(pose)
    y, x = np.indices((height, width))
    ray = np.stack(
        ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones_like(x)), axis=-1
    ).reshape(-1, 3)
    measured = np.divide(
        1,
        depths.reshape(-1),
        out=np.zeros(frames * pixels, np.float32),
        where=depths.reshape(-1) > 0,
    )
    measured = np.where(np.isfinite(measured), measured, 0).astype(np.float64)
    initial_depths = (
        depths.copy() if initial_depths is None else np.asarray(initial_depths, np.float32)
    )
    if (
        initial_depths.shape != depths.shape
        or not np.isfinite(initial_depths).all()
        or np.any(initial_depths < 0)
    ):
        raise ValueError("initial depths must be finite nonnegative and match measured depths")
    initial_rho = np.divide(
        1,
        initial_depths.reshape(-1),
        out=np.zeros(frames * pixels, np.float32),
        where=initial_depths.reshape(-1) > 0,
    )
    initial_rho = np.where(np.isfinite(initial_rho) & (measured > 0), initial_rho, 0).astype(
        np.float64
    )
    rho = initial_rho.copy()
    geometry = [_geometry(d, camera, 0.1) for d in depths]

    def linearize():
        jp = np.zeros((frames * pixels, 2 * (frames - 1), dofs))
        jd = np.zeros(jp.shape[:2])
        residual = np.zeros_like(jd)
        factors = 0
        for source in range(frames):
            sr = rho[source * pixels : (source + 1) * pixels]
            valid = (sr > 0) & np.isfinite(sr)
            ids = np.flatnonzero(valid)
            rray = ray[ids]
            world = (rray / sr[ids, None]) @ poses[source, :3, :3].T + poses[source, :3, 3]
            lane = 0
            for target in range(frames):
                if target == source:
                    continue
                p = (world - poses[target, :3, 3]) @ poses[target, :3, :3]
                z = p[:, 2]
                uv = np.divide(
                    p[:, :2], z[:, None], out=np.zeros((len(p), 2)), where=z[:, None] > 0
                ) * [camera.fx, camera.fy] + [camera.cx, camera.cy]
                pix = np.floor(uv + 0.5).astype(int)
                good = (
                    (z > 1e-6)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < width - 1)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < height - 1)
                )
                pix = np.clip(pix, [0, 0], [width - 1, height - 1])
                rid = pix[:, 1] * width + pix[:, 0]
                verts, norms, nvalid = geometry[target]
                q, n = verts[rid], norms[rid]
                good &= nvalid[rid] & (q[:, 2] > 0) & (np.linalg.norm(p - q, axis=1) < 0.15)
                target_i, grad = bilinear(grays[target], uv)
                good &= target_i >= 0
                selected = np.flatnonzero(good)
                rows = source * pixels + ids[selected]
                pp, ww, nn = p[selected], world[selected], n[selected]
                photo_n = np.column_stack(
                    (
                        grad[selected, 0] * camera.fx / pp[:, 2],
                        grad[selected, 1] * camera.fy / pp[:, 2],
                        -(
                            grad[selected, 0] * camera.fx * pp[:, 0]
                            + grad[selected, 1] * camera.fy * pp[:, 1]
                        )
                        / pp[:, 2] ** 2,
                    )
                )
                raw = np.column_stack(
                    (
                        (nn * (pp - q[selected])).sum(axis=1),
                        target_i[selected] - grays[source].ravel()[ids[selected]],
                    )
                )
                weight = np.sqrt(
                    np.minimum(1, np.array([0.03, 10]) / np.maximum(abs(raw), 1e-20))
                ) * [10, 1 / 255]
                norms2 = np.stack((nn, photo_n), axis=1)
                wn = norms2 @ poses[target, :3, :3].T
                js = np.concatenate((np.cross(ww[:, None, :], wn), wn), axis=2) * weight[..., None]
                dr = (
                    -(rray[selected] / sr[ids[selected], None] ** 2)
                    @ poses[source, :3, :3].T
                    @ poses[target, :3, :3]
                )
                jd[rows, lane : lane + 2] = (norms2 * dr[:, None, :]).sum(axis=2) * weight
                residual[rows, lane : lane + 2] = raw * weight
                if source:
                    jp[rows, lane : lane + 2, 6 * (source - 1) : 6 * source] = js
                if target:
                    jp[rows, lane : lane + 2, 6 * (target - 1) : 6 * target] = -js
                factors += 2 * len(rows)
                lane += 2
        prior = np.where((rho > 0) & (measured > 0), rho - measured, 0)
        cost = float((residual**2).sum() + 100 * np.sum(prior**2))
        return jp, jd, residual, prior, cost, factors

    before = None
    success = True
    try:
        for _ in range(iterations):
            jp, jd, r, prior, cost, count = linearize()
            if before is None:
                before = (cost, count)
            if count < dofs:
                success = False
                break
            dp, dd = schur_step(jp, jd, r, prior)
            for f in range(1, frames):
                poses[f] = _exp(dp[6 * (f - 1) : 6 * f]) @ poses[f]
            updated = rho + dd
            rho = (
                np.where(
                    np.isfinite(updated) & (rho > 0) & (measured > 0),
                    np.clip(updated, measured / 1.1, measured / 0.9),
                    rho,
                )
                .astype(np.float32)
                .astype(np.float64)
            )
        *_, final_cost, final_count = linearize()
        success &= (
            np.isfinite(poses).all()
            and np.isfinite(final_cost)
            and final_count >= max(dofs, min_factor_ratio * before[1])
            and final_cost <= before[0] + abs(before[0]) * 1e-4 + 1e-7
        )
    except (np.linalg.LinAlgError, FloatingPointError):
        success = False
        final_cost = np.inf
        final_count = 0
    return dict(
        success=bool(success),
        original_poses=original,
        poses=poses if success else original,
        inverse_depth=(rho if success else initial_rho).reshape(depths.shape),
        initial_cost=before[0] if before else 0,
        final_cost=final_cost,
        initial_factors=before[1] if before else 0,
        final_factors=final_count,
    )


def optimize_pose_graph(
    initial_poses, edges, measurements, *, iterations=3, damping=1e-4, edge_information=None
):
    """Anchored full six-DoF graph, shortest quaternion chordal residuals."""
    original = np.asarray(initial_poses, np.float64).copy()
    poses = original.copy()
    edges = np.asarray(edges, int)
    information = (
        np.ones((len(edges), 2))
        if edge_information is None
        else np.asarray(edge_information, float)
    )
    measurements = np.asarray(measurements, np.float64)
    dofs = 6 * (len(poses) - 1)

    def residual(value):
        rows = []
        for (a, b), measured, info in zip(edges, measurements, information):
            relative = np.linalg.inv(value[a]) @ value[b]
            error = measured[:3, :3].T @ relative[:3, :3]
            q = quaternion(error)
            rows.append(np.r_[2 * q[:3] * info[0], (relative[:3, 3] - measured[:3, 3]) * info[1]])
        return np.concatenate(rows) if rows else np.empty(0)

    initial_cost = float(residual(original) @ residual(original))
    success = bool(dofs and len(edges))
    for _ in range(iterations):
        if not success:
            break
        r = residual(poses)
        j = np.zeros((len(r), dofs))
        eps = 1e-5
        for k in range(dofs):
            frame = 1 + k // 6
            delta = np.zeros(6)
            delta[k % 6] = eps
            plus = poses.copy()
            minus = poses.copy()
            plus[frame] = _exp(delta) @ poses[frame]
            minus[frame] = _exp(-delta) @ poses[frame]
            j[:, k] = (residual(plus) - residual(minus)) / (2 * eps)
        try:
            step = np.linalg.solve(j.T @ j + damping * np.eye(dofs), -j.T @ r)
            proposed = poses.copy()
            for frame in range(1, len(poses)):
                proposed[frame] = _exp(step[6 * (frame - 1) : 6 * frame]) @ poses[frame]
            next_r = residual(proposed)
            success = np.isfinite(proposed).all() and next_r @ next_r <= (r @ r) * 1.001 + 1e-6
            if success:
                poses = proposed
        except np.linalg.LinAlgError:
            success = False
    return dict(
        success=bool(success),
        poses=poses if success else original,
        cost=float(residual(poses) @ residual(poses)),
        cost_before=initial_cost,
        cost_after=float(residual(poses) @ residual(poses)),
    )


class CPUHybridSLAM:
    """CPU-only matched bounded dense-primary hybrid controller."""

    def __init__(
        self,
        camera,
        *,
        tracker_options=None,
        map_options=None,
        max_keyframes=16,
        keyframe_interval=5,
        ba_every_keyframes=4,
        enable_local_ba=True,
        enable_loops=True,
        max_features=512,
        tuning_profile=None,
        min_loop_frames=30,
        max_loop_edges=128,
        max_frames=65536,
    ):
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (
                    max_keyframes,
                    keyframe_interval,
                    ba_every_keyframes,
                    min_loop_frames,
                    max_loop_edges,
                    max_frames,
                )
            )
            or not 2 <= max_keyframes <= 64
            or min(keyframe_interval, ba_every_keyframes, max_loop_edges, max_frames) < 1
            or min_loop_frames < 1
            or max_loop_edges > 256 - max_keyframes
            or max_frames > 1000000
        ):
            raise ValueError("invalid bounded keyframe/cadence configuration")
        self.camera = camera
        self.tracker = CPUFusedTracker(camera, **dict(tracker_options or {}))
        self.map_options = dict(map_options or {})
        self.map = CPUColorTSDF(camera, **self.map_options)
        self.frontend = CPUFeatures(camera, max_features=max_features)
        self.min_loop_frames = min_loop_frames
        self.max_loop_edges = max_loop_edges
        self.max_frames = max_frames
        self._verification_metrics = {}
        self._last_graph_cost = None
        self._last_graph_metrics = {}
        self._pending_generations = None
        self.capacity = max_keyframes
        self.keyframe_interval = keyframe_interval
        self.ba_every_keyframes = ba_every_keyframes
        self.enable_local_ba = enable_local_ba
        self.enable_loops = enable_loops
        self.world_from_camera = np.eye(4)
        self.initialized = False
        self.index = 0
        self.last_timestamp = None
        self.keyframes = 0
        self.active_keyframes = []
        self.slots = {}
        self.trajectory = []
        self._corrected_history = []
        self.results = []
        self.ba_events = []
        self.loop_events = []
        self._pending_candidate = None
        self._pending_query = None
        self.retained_loops = []
        self.history_anchors = []
        self._recovery = False

    def _capture(self, depth, rgb, features, timestamp):
        if len(self.active_keyframes) == self.capacity:
            slot = self.active_keyframes.pop(0)
            evicted = self.slots[slot]["frame_id"]
            self.retained_loops = [
                (a, b, m) for a, b, m in self.retained_loops if a != evicted and b != evicted
            ]
        else:
            slot = next(i for i in range(self.capacity) if i not in self.active_keyframes)
        low = self.tracker._maps(depth)[-1][0][:, 2].reshape(
            self.tracker.cameras[-1].height, self.tracker.cameras[-1].width
        )
        gray = self.tracker.intensity_pyramid(intensity(rgb))[-1] * 255
        self.slots[slot] = dict(
            raw_depth=depth.copy(),
            measured_low_depth=low.copy(),
            depth=depth.copy(),
            rgb=rgb.copy(),
            low_depth=low.copy(),
            gray=gray.copy(),
            pose=self.world_from_camera.copy(),
            features=features,
            timestamp=timestamp,
            frame_id=self.index + 1,
        )
        self.active_keyframes.append(slot)
        self.keyframes += 1
        return slot

    def _rebuild(self):
        self.map.reset()
        for slot in self.active_keyframes:
            saved = self.slots[slot]
            self.map.integrate(saved["depth"], saved["rgb"], saved["pose"])

    def _ba(self, timestamp):
        slots = self.active_keyframes[-4:]
        saved = [self.slots[s] for s in slots]
        result = optimize_dense_window(
            self.tracker.cameras[-1],
            np.stack([s["measured_low_depth"] for s in saved]),
            np.stack([s["gray"] for s in saved]),
            np.stack([s["pose"] for s in saved]),
            initial_depths=np.stack([s["low_depth"] for s in saved]),
        )
        event = dict(
            timestamp=float(timestamp),
            success=result["success"],
            window=len(slots),
            initial_cost=result["initial_cost"],
            final_cost=result["final_cost"] if np.isfinite(result["final_cost"]) else None,
            initial_factors=result["initial_factors"],
            final_factors=result["final_factors"],
        )
        self.ba_events.append(event)
        if result["success"]:
            for row, s in enumerate(saved):
                z = s["measured_low_depth"]
                rho = result["inverse_depth"][row]
                ratio = np.divide(1, z * rho, out=np.ones_like(z), where=(z > 0) & (rho > 0))
                accepted = (z > 0) & (rho > 0) & np.isfinite(rho)
                y = np.arange(self.camera.height) * z.shape[0] // self.camera.height
                x = np.arange(self.camera.width) * z.shape[1] // self.camera.width
                full_ratio = np.where(accepted, ratio, 1)[y[:, None], x[None, :]]
                s["depth"] = s["raw_depth"] * full_ratio.astype(np.float32)
                s["low_depth"] = np.where(accepted, 1 / np.maximum(rho, 1e-20), z).astype(
                    np.float32
                )
                s["pose"] = result["poses"][row].copy()
                f = s["features"]
                uv = f["positions"][:, :2]
                pix = np.floor(uv + 0.5).astype(int)
                if len(pix):
                    updated = s["depth"][pix[:, 1], pix[:, 0]]
                    f["positions"][:, 2] = updated
                    f["landmarks"][:, :3] = np.column_stack(
                        (
                            (uv[:, 0] - self.camera.cx) * updated / self.camera.fx,
                            (uv[:, 1] - self.camera.cy) * updated / self.camera.fy,
                            updated,
                        )
                    )
                    f["landmarks"][:, 3] = updated > 0
            self.world_from_camera = saved[-1]["pose"].copy()
            self._correct_history(
                {slot: pose for slot, pose in zip(slots, result["original_poses"])}
            )
            self._rebuild()
        return event

    def _retrieve(self, features):
        histogram = np.bincount(features["descriptors"][:, 0] & 255, minlength=256).astype(float)
        norm = np.linalg.norm(histogram)
        best = None
        score = 0.85
        for slot in sorted(self.active_keyframes):
            saved = self.slots[slot]
            if self.index + 1 < saved["frame_id"] + self.min_loop_frames:
                continue
            other = np.bincount(saved["features"]["descriptors"][:, 0] & 255, minlength=256).astype(
                float
            )
            denominator = norm * np.linalg.norm(other)
            similarity = float(histogram @ other / denominator) if denominator else 0
            if similarity > score:
                score = similarity
                best = slot
        return best

    def _correct_history(self, old):
        for i, (slot, frame_id) in enumerate(self.history_anchors):
            if (
                slot in old
                and slot in self.active_keyframes
                and self.slots[slot]["frame_id"] == frame_id
            ):
                correction = self.slots[slot]["pose"] @ np.linalg.inv(old[slot])
                timestamp, pose = self._corrected_history[i]
                self._corrected_history[i] = (timestamp, correction @ pose)

    def _loop_graph(self, verified, candidate, query):
        if (
            query not in self.active_keyframes
            or candidate not in self.active_keyframes
            or len(self.active_keyframes) < 2
        ):
            return False
        old = {slot: self.slots[slot]["pose"].copy() for slot in self.active_keyframes}
        slots = self.active_keyframes
        mapping = {slot: i for i, slot in enumerate(slots)}
        ids = {self.slots[slot]["frame_id"]: slot for slot in slots}
        measured = (
            np.linalg.inv(old[candidate])
            @ verified
            @ np.linalg.inv(self.world_from_camera)
            @ old[query]
        )
        edges = [[i, i + 1] for i in range(len(slots) - 1)]
        measurements = [np.linalg.inv(old[a]) @ old[b] for a, b in zip(slots, slots[1:])]
        information = [[1.0, 1.0] for _ in edges]
        for a, b, value in self.retained_loops:
            if a in ids and b in ids:
                edges.append([mapping[ids[a]], mapping[ids[b]]])
                measurements.append(value)
                information.append([10.0, 10.0])
        edges.append([mapping[candidate], mapping[query]])
        measurements.append(measured)
        information.append([10.0, 10.0])
        result = optimize_pose_graph(
            np.stack([old[slot] for slot in slots]),
            edges,
            measurements,
            edge_information=information,
        )
        self._last_graph_cost = result["cost"]
        self._last_graph_metrics = dict(
            graph_success=result["success"],
            graph_edges=len(edges),
            graph_cost_before=result["cost_before"],
            graph_cost_after=result["cost_after"],
        )
        if not result["success"]:
            return False
        for slot, pose in zip(slots, result["poses"]):
            self.slots[slot]["pose"] = pose
        self.world_from_camera = (
            self.slots[query]["pose"] @ np.linalg.inv(old[query]) @ self.world_from_camera
        )
        self.retained_loops.append(
            (self.slots[candidate]["frame_id"], self.slots[query]["frame_id"], measured.copy())
        )
        self.retained_loops = [
            (a, b, value) for a, b, value in self.retained_loops if a in ids and b in ids
        ][-self.max_loop_edges :]
        self._correct_history(old)
        self._rebuild()
        return True

    def _verify_loop(self, current, slot):
        self._verification_metrics = dict(inliers=0, rmse_px=None)
        saved = self.slots[slot]
        target = saved["features"]
        matches = self.frontend.match(current, target)
        candidate = ransac_reference(current, target, matches)
        self._verification_metrics["inliers"] = candidate["inliers"]
        if not candidate["success"]:
            return None
        ids = np.flatnonzero(matches < len(target["landmarks"]))
        a = current["landmarks"][ids, :3]
        b = target["landmarks"][matches[ids], :3]
        transform = candidate["target_from_source"]
        keep = np.linalg.norm(a @ transform[:3, :3].T + transform[:3, 3] - b, axis=1) < 0.05
        ids = ids[keep]
        b = b[keep]
        points = b @ saved["pose"][:3, :3].T + saved["pose"][:3, 3]
        seed = np.linalg.inv(saved["pose"] @ transform)
        result = optimize_reprojection(
            points,
            current["positions"][ids, :2],
            self.camera,
            initial_pose=seed,
            iterations=8,
            min_correspondences=20,
        )
        if not result.success:
            return None
        p = points @ result.pose[:3, :3].T + result.pose[:3, 3]
        valid = np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-6)
        p, ids = p[valid], ids[valid]
        if len(ids) < 20:
            return None
        uv = p[:, :2] / p[:, 2, None] * [self.camera.fx, self.camera.fy] + [
            self.camera.cx,
            self.camera.cy,
        ]
        squared = np.sum((uv - current["positions"][ids, :2]) ** 2, axis=1)
        cost = np.sum(squared * np.minimum(1, 3 / np.maximum(np.sqrt(squared), 1e-20)))
        if not np.isfinite(cost) or cost > 9 * len(ids):
            return None
        self._verification_metrics.update(
            inliers=len(ids), rmse_px=float(np.sqrt(cost / max(len(ids), 1)))
        )
        return np.linalg.inv(result.pose)

    def corrected_trajectory(self):
        return [(timestamp, pose.copy()) for timestamp, pose in self._corrected_history]

    def process(self, depth, rgb=None, *, timestamp):
        started = perf_counter()
        if self.index >= self.max_frames:
            raise ValueError("bounded trajectory capacity exhausted")
        depth = _depth(depth, self.camera)
        if not np.isfinite(timestamp) or (
            self.last_timestamp is not None and timestamp <= self.last_timestamp
        ):
            raise ValueError("timestamps must be finite and strictly increasing")
        rgb = np.full((*depth.shape, 3), 180, np.uint8) if rgb is None else np.asarray(rgb)
        if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 HxWx3")
        if self.initialized:
            reference, gray = self.map.raycast_intensity(self.world_from_camera)
        else:
            reference, gray = depth, intensity(rgb)
        self.tracker.set_reference(reference, gray)
        tracking = self.tracker.track(depth, rgb)
        capture = not self.initialized or self.index % self.keyframe_interval == 0
        features = (
            self.frontend.extract(rgb, depth)
            if capture or self._recovery or self._pending_candidate is not None
            else None
        )
        loop_pose = None
        loop_attempt = self.enable_loops and self._pending_candidate in self.active_keyframes
        if loop_attempt and self._pending_generations is not None:
            loop_attempt = (
                self._pending_query in self.active_keyframes
                and self._pending_generations
                == (
                    self.slots[self._pending_candidate]["frame_id"],
                    self.slots[self._pending_query]["frame_id"],
                )
            )
        if loop_attempt:
            loop_pose = self._verify_loop(features, self._pending_candidate)
            self.loop_events.append(
                dict(
                    timestamp=float(timestamp),
                    verified=loop_pose is not None,
                    accepted=False,
                    recovered=loop_pose is not None and not tracking.success,
                    candidate_slot=self._pending_candidate,
                    query_slot=self._pending_query,
                    graph_success=False,
                    graph_edges=0,
                    graph_cost_before=None,
                    graph_cost_after=None,
                    **self._verification_metrics,
                )
            )
        success = tracking.success or loop_pose is not None
        result = dict(
            timestamp=float(timestamp),
            success=success,
            status="tracked"
            if success and self.initialized
            else "initialized"
            if success
            else "lost",
            tracking_ms=tracking.elapsed_ms,
            correspondences=tracking.correspondences,
            rmse=tracking.rmse,
            keyframe=False,
            world_from_camera=self.world_from_camera.copy(),
        )
        rebuilt = False
        if success:
            self.world_from_camera = (
                self.world_from_camera @ tracking.pose if tracking.success else loop_pose
            )
            if loop_pose is not None and tracking.success:
                accepted = self._loop_graph(loop_pose, self._pending_candidate, self._pending_query)
                self.loop_events[-1]["accepted"] = accepted
                self.loop_events[-1].update(self._last_graph_metrics)
            if capture:
                self._capture(depth, rgb, features, timestamp)
                result["keyframe"] = True
                if (
                    self.enable_local_ba
                    and not loop_attempt
                    and len(self.active_keyframes) >= 2
                    and self.keyframes % self.ba_every_keyframes == 0
                ):
                    event = self._ba(timestamp)
                    result["local_ba"] = event
                    rebuilt = rebuilt or event["success"]
            if not rebuilt:
                self.map.integrate(depth, rgb, self.world_from_camera)
            self.initialized = True
            self.trajectory.append((float(timestamp), self.world_from_camera.copy()))
            self._corrected_history.append((float(timestamp), self.world_from_camera.copy()))
            anchor = self.active_keyframes[-1]
            self.history_anchors.append((anchor, self.slots[anchor]["frame_id"]))
        if self.enable_loops and (capture or self._recovery):
            self._pending_candidate = self._retrieve(features)
            self._pending_query = self.active_keyframes[-1] if self.active_keyframes else None
            self._pending_generations = (
                (
                    self.slots[self._pending_candidate]["frame_id"],
                    self.slots[self._pending_query]["frame_id"],
                )
                if self._pending_candidate is not None and self._pending_query is not None
                else None
            )
        self._recovery = not success
        result.update(
            world_from_camera=self.world_from_camera.copy(),
            active_keyframes=len(self.active_keyframes),
            mapping=self.map.stats(),
            tracking_backend="NumPy CPU ICP + photometric",
        )
        result["mapping_success"] = all(
            result["mapping"].get(k, 0) == 0
            for k in ("overflow", "domain_errors", "probe_exhaustion")
        )
        result["elapsed_ms"] = (perf_counter() - started) * 1000
        self.results.append(result)
        self.index += 1
        self.last_timestamp = float(timestamp)
        return result
