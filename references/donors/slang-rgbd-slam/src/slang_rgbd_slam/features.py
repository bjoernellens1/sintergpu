"""Bounded GPU FAST-9, Harris selection and clean oriented-BRIEF features.

The deterministic generated pattern is original to this implementation. This is
an ORB-style frontend, not bit-compatible with OpenCV ORB. Pixel geometry and
RGB-D backprojection use the generated SymForce/Caspar/Slang primitives.
"""

from pathlib import Path

import numpy as np


def feature_brief_pattern():
    """Return a deterministic 256-pair Gaussian disk pattern, radius 15 pixels."""
    rng = np.random.default_rng(0)
    points = []
    while len(points) < 512:
        p = np.rint(rng.normal(0, 5, size=2)).astype(np.int32)
        if np.sum(p * p) <= 225:
            points.append(p)
    return np.asarray(points, np.int32).reshape(256, 4)


def feature_grids(camera, max_features, levels, cell_quota):
    """Per-level spatial grid dimensions and quotas bounded by the total budget."""
    weights = np.arange(levels, 0, -1, dtype=np.float64)
    budget = np.floor(weights / weights.sum() * max_features).astype(int)
    budget[: max_features - int(budget.sum())] += 1
    result = []
    for level, n in enumerate(budget):
        if n == 0:
            result.append((0, 0, 0))
            continue
        quota = min(cell_quota, int(n))
        cells = max(1, int(n) // quota)
        aspect = max(1, camera.width // 2**level) / max(1, camera.height // 2**level)
        x = max(1, int(np.sqrt(cells * aspect)))
        x = min(x, cells)
        y = max(1, cells // x)
        result.append((x, y, quota))
    return result


class GPUFeatures:
    def __init__(
        self,
        camera,
        runtime=None,
        max_features=1024,
        levels=4,
        fast_threshold=20,
        ratio=0.8,
        max_hamming=80,
        cell_quota=4,
        matcher_group_size=64,
    ):
        from caspar_slang.primitives import primitives_source
        from caspar_slang.runtime import VulkanRuntime

        from .tracker import Camera

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
            or matcher_group_size not in (32, 64, 128)
        ):
            raise ValueError("invalid feature configuration")
        if camera.width % 2 ** (levels - 1) or camera.height % 2 ** (levels - 1):
            raise ValueError("camera dimensions must be divisible by pyramid scale")
        self.camera, self.max_features, self.levels = camera, max_features, levels
        self.fast_threshold, self.ratio, self.max_hamming = fast_threshold, ratio, max_hamming
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.matcher_group_size = matcher_group_size
        self.grids = feature_grids(camera, max_features, levels, cell_quota)
        self.pyramid = []
        for level in range(levels):
            width, height = max(1, camera.width // 2**level), max(1, camera.height // 2**level)
            self.pyramid.append(
                dict(
                    width=width,
                    height=height,
                    gray=self._buffer(width * height),
                    fast=self._buffer(width * height),
                    harris=self._buffer(width * height),
                )
            )
        self.pattern = self.runtime.buffer(feature_brief_pattern().astype(np.float32).ravel())
        self.forward = self._buffer(max_features)
        self.reverse = self._buffer(max_features)
        source = (
            f"#define MATCH_GROUP_SIZE {matcher_group_size}\n"
            + primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("features.slang").read_text()
        )
        entries = (
            "rgbToGray",
            "downsampleGray",
            "resetFeatures",
            "resetMatchCount",
            "fastDetect",
            "nonmaxHarris",
            "selectCells",
            "describe",
            "matchForward",
            "matchReverse",
            "filterMatches",
        )
        self.kernels = {entry: self.runtime.compile(source, entry=entry) for entry in entries}

    def _buffer(self, count):
        import slangpy as spy

        return self.runtime.device.create_buffer(
            element_count=count,
            struct_size=4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

    def create_set(self):
        """Persistent buffers: compact positions/landmarks float4, eight uint words."""
        return dict(
            positions=self._buffer(self.max_features * 4),
            landmarks=self._buffer(self.max_features * 4),
            descriptors=self._buffer(self.max_features * 8),
            count=self.runtime.buffer(np.zeros(1, np.uint32)),
            match_count=self.runtime.buffer(np.zeros(1, np.uint32)),
        )

    def _params(self, level=0):
        c, p = self.camera, self.pyramid[level]
        x, y, q = self.grids[level]
        return dict(
            width=p["width"],
            height=p["height"],
            fullWidth=c.width,
            fullHeight=c.height,
            fx=c.fx,
            fy=c.fy,
            cx=c.cx,
            cy=c.cy,
            level=level,
            scale=float(2**level),
            gridX=x,
            gridY=y,
            quota=q,
            maxFeatures=self.max_features,
            fastThreshold=self.fast_threshold,
            ratio=self.ratio,
            maxHamming=self.max_hamming,
        )

    def _dispatch(self, entry, threads, buffers, encoder, level=0):
        self.kernels[entry].dispatch(
            thread_count=threads,
            vars=dict(cfg=self._params(level), **buffers),
            command_encoder=encoder,
        )

    def record_extract(self, rgb_buffer, depth_buffer, feature_set, encoder):
        self._dispatch("resetFeatures", [1, 1, 1], dict(featureCount=feature_set["count"]), encoder)
        for level, p in enumerate(self.pyramid):
            threads = [p["width"], p["height"], 1]
            if level == 0:
                self._dispatch(
                    "rgbToGray", threads, dict(rgb=rgb_buffer, grayOutput=p["gray"]), encoder, level
                )
            else:
                self._dispatch(
                    "downsampleGray",
                    threads,
                    dict(grayInput=self.pyramid[level - 1]["gray"], grayOutput=p["gray"]),
                    encoder,
                    level,
                )
            encoder.global_barrier()
            self._dispatch(
                "fastDetect",
                threads,
                dict(grayInput=p["gray"], fastScores=p["fast"]),
                encoder,
                level,
            )
            encoder.global_barrier()
            self._dispatch(
                "nonmaxHarris",
                threads,
                dict(grayInput=p["gray"], fastScores=p["fast"], harrisScores=p["harris"]),
                encoder,
                level,
            )
            encoder.global_barrier()
            gx, gy, quota = self.grids[level]
            if gx and gy:
                self._dispatch(
                    "selectCells",
                    [gx * 64, gy, 1],
                    dict(
                        harrisScores=p["harris"],
                        depth=depth_buffer,
                        featureCount=feature_set["count"],
                        positions=feature_set["positions"],
                        landmarks=feature_set["landmarks"],
                    ),
                    encoder,
                    level,
                )
            encoder.global_barrier()
        for level, p in enumerate(self.pyramid):
            self._dispatch(
                "describe",
                [self.max_features, 1, 1],
                dict(
                    grayInput=p["gray"],
                    pattern=self.pattern,
                    featureCount=feature_set["count"],
                    positions=feature_set["positions"],
                    descriptors=feature_set["descriptors"],
                ),
                encoder,
                level,
            )
        encoder.global_barrier()

    def record_match(self, source_set, target_set, match_buffer, encoder):
        """Source->target indices or UINT_MAX, plus source_set['match_count']."""
        self._dispatch(
            "resetMatchCount", [1, 1, 1], dict(matchCount=source_set["match_count"]), encoder
        )
        buffers = dict(
            sourceDescriptors=source_set["descriptors"],
            targetDescriptors=target_set["descriptors"],
            sourceCount=source_set["count"],
            targetCount=target_set["count"],
        )
        self._dispatch(
            "matchForward",
            [self.max_features, 1, 1],
            dict(**buffers, forwardMatches=self.forward),
            encoder,
        )
        self._dispatch(
            "matchReverse",
            [self.max_features, 1, 1],
            dict(**buffers, reverseMatches=self.reverse),
            encoder,
        )
        encoder.global_barrier()
        self._dispatch(
            "filterMatches",
            [self.max_features, 1, 1],
            dict(
                sourceCount=source_set["count"],
                forwardMatches=self.forward,
                reverseMatches=self.reverse,
                matches=match_buffer,
                matchCount=source_set["match_count"],
            ),
            encoder,
        )
        encoder.global_barrier()

    def snapshot(self, feature_set):
        count = int(feature_set["count"].to_numpy().view(np.uint32).ravel()[0])
        return dict(
            count=count,
            positions=feature_set["positions"]
            .to_numpy()
            .view(np.float32)
            .ravel()[: count * 4]
            .reshape(-1, 4)
            .copy(),
            landmarks=feature_set["landmarks"]
            .to_numpy()
            .view(np.float32)
            .ravel()[: count * 4]
            .reshape(-1, 4)
            .copy(),
            descriptors=feature_set["descriptors"]
            .to_numpy()
            .view(np.uint32)
            .ravel()[: count * 8]
            .reshape(-1, 8)
            .copy(),
        )
