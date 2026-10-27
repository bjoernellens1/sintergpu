"""GPU sparse hashed TSDF mapping, raycasting and surface compaction.

Blocks are 8³ voxel centers. The signed ten-bit block coordinate domain is
[-512, 511] on each axis. Out-of-domain and capacity failures are explicit
counters. Float RGB is stored in the input's 0..255 range. Host readbacks are
restricted to diagnostics and final export; record methods read no counters.
"""

from pathlib import Path

import numpy as np

from .tracker import Camera


class GPUTSDF:
    def __init__(
        self,
        camera,
        runtime=None,
        voxel_size=0.03,
        truncation_voxels=4,
        max_blocks=8192,
        max_weight=100,
        near=0.1,
        far=8,
    ):
        import slangpy as spy
        from caspar_slang.primitives import primitives_source
        from caspar_slang.runtime import VulkanRuntime

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
            or max_blocks <= 0
            or max_blocks > 32767
        ):
            raise ValueError("invalid TSDF configuration")
        self.camera = camera
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.voxel_size, self.mu = voxel_size, voxel_size * truncation_voxels
        self.max_blocks, self.max_weight = max_blocks, max_weight
        self.near, self.far = near, far
        self.hash_capacity = 1 << (4 * max_blocks - 1).bit_length()
        self.surface_capacity = max_blocks * 512 * 3
        self.frame_index = 0
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access

        def allocate(count):
            return self.runtime.device.create_buffer(
                element_count=count, struct_size=4, usage=usage
            )

        self.buffers = {
            "hashKeys": allocate(self.hash_capacity),
            "hashValues": allocate(self.hash_capacity),
            "stamps": allocate(self.hash_capacity),
            "blockKeys": allocate(max_blocks),
            "activeBlocks": allocate(max_blocks),
            "counters": allocate(6),
            "fusionArgs": self.runtime.device.create_buffer(
                element_count=3, struct_size=4, usage=usage | spy.BufferUsage.indirect_argument
            ),
            "distances": allocate(max_blocks * 512),
            "weights": allocate(max_blocks * 512),
            "colorWeights": allocate(max_blocks * 512),
            "colors": allocate(max_blocks * 512 * 3),
        }
        # Surface buffers are allocated lazily; extraction never runs in the frame loop.
        self.depth_buffer = allocate(camera.width * camera.height)
        self.rgb_buffer = allocate(camera.width * camera.height * 3)
        self.pose_buffer = self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
        self.status_buffer = self.runtime.buffer(np.zeros(2, np.uint32))
        self.depth_output = allocate(camera.width * camera.height)
        self.intensity_output = allocate(camera.width * camera.height)
        source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("tsdf.slang").read_text()
        )
        names = (
            "initialize",
            "resetFrame",
            "allocateBlocks",
            "collectBlocks",
            "integrateVoxels",
            "prepareFusion",
            "raycast",
            "resetSurface",
            "extractSurface",
        )
        self.kernels = {name: self.runtime.compile(source, entry=name) for name in names}
        encoder = self.runtime.device.create_command_encoder()
        self.record_reset(encoder)
        self.runtime.device.submit_command_buffer(encoder.finish())

    def record_reset(self, encoder, tracking_status=None):
        """Clear the bounded map on GPU, optionally conditioned on a two-word status."""
        names = (
            "hashKeys",
            "hashValues",
            "stamps",
            "counters",
            "distances",
            "weights",
            "colorWeights",
            "colors",
        )
        self._dispatch(
            "initialize",
            max(self.hash_capacity, self.max_blocks * 512),
            encoder,
            {
                **{name: self.buffers[name] for name in names},
                "trackingStatus": self.status_buffer
                if tracking_status is None
                else tracking_status,
            },
        )
        encoder.global_barrier()

    def _params(self, frame_index=None):
        c = self.camera
        return dict(
            width=c.width,
            height=c.height,
            fx=c.fx,
            fy=c.fy,
            cx=c.cx,
            cy=c.cy,
            voxelSize=self.voxel_size,
            mu=self.mu,
            maxWeight=self.max_weight,
            nearDepth=self.near,
            farDepth=self.far,
            maxBlocks=self.max_blocks,
            hashCapacity=self.hash_capacity,
            frameIndex=self.frame_index if frame_index is None else frame_index,
            surfaceCapacity=self.surface_capacity,
        )

    def _dispatch(self, entry, count, encoder, buffers, frame_index=None):
        self.kernels[entry].dispatch(
            thread_count=count if isinstance(count, list) else [count, 1, 1],
            vars={"cfg": self._params(frame_index), **buffers},
            command_encoder=encoder,
        )

    def record_integrate(
        self, depth_buffer, rgb_buffer, pose_buffer, encoder, tracking_status=None, frame_index=None
    ):
        """Record allocation, unique activation and voxel fusion without readback.

        ``pose_buffer`` is row-major world_from_camera; RGB has three float32
        channels per pixel. A nonzero tracking_status[1] prevents all mutations.
        Frame indices must be unique nonzero uint32 values for this map.
        """
        if frame_index is None:
            frame_index = self.frame_index + 1
        if (
            not isinstance(frame_index, int)
            or not 1 <= frame_index <= 0xFFFFFFFF
            or frame_index <= self.frame_index
        ):
            raise ValueError("frame_index must be a strictly increasing nonzero uint32")
        self.frame_index = max(self.frame_index, frame_index)
        status = self.status_buffer if tracking_status is None else tracking_status
        self._dispatch(
            "resetFrame",
            1,
            encoder,
            dict(counters=self.buffers["counters"], trackingStatus=status),
            frame_index,
        )
        shared = dict(depth=depth_buffer, pose=pose_buffer, trackingStatus=status)
        for entry, names in (
            ("allocateBlocks", ("hashKeys", "hashValues", "blockKeys", "counters")),
            ("collectBlocks", ("hashKeys", "hashValues", "stamps", "activeBlocks", "counters")),
        ):
            self._dispatch(
                entry,
                self.camera.width * self.camera.height,
                encoder,
                {**shared, **{k: self.buffers[k] for k in names}},
                frame_index,
            )
            # A separate dispatch publishes every claimed hash value before any
            # contender collects its block. No shader spins waiting for another lane.
            encoder.global_barrier()
        names = (
            "activeBlocks",
            "blockKeys",
            "counters",
            "distances",
            "weights",
            "colorWeights",
            "colors",
        )
        self._dispatch(
            "prepareFusion",
            1,
            encoder,
            dict(
                counters=self.buffers["counters"],
                trackingStatus=status,
                fusionArgs=self.buffers["fusionArgs"],
            ),
            frame_index,
        )
        encoder.global_barrier()
        self._dispatch_indirect(
            "integrateVoxels",
            encoder,
            {**shared, "rgb": rgb_buffer, **{k: self.buffers[k] for k in names}},
            frame_index,
        )

    def _dispatch_indirect(self, entry, encoder, buffers, frame_index=None):
        import slangpy as spy

        compute = encoder.begin_compute_pass()
        root = compute.bind_pipeline(self.kernels[entry].pipeline)
        cursor = spy.ShaderCursor(root)
        cursor["cfg"] = self._params(frame_index)
        for name, buffer in buffers.items():
            cursor[name] = buffer
        compute.dispatch_compute_indirect(spy.BufferOffsetPair(self.buffers["fusionArgs"], 0))
        compute.end()

    def record_raycast(self, pose_buffer, depth_output, encoder, intensity_output=None):
        names = ("hashKeys", "hashValues", "distances", "weights", "colors", "colorWeights")
        self._dispatch(
            "raycast",
            [self.camera.width, self.camera.height, 1],
            encoder,
            dict(
                pose=pose_buffer,
                rayDepth=depth_output,
                rayIntensity=self.intensity_output
                if intensity_output is None
                else intensity_output,
                **{k: self.buffers[k] for k in names},
            ),
        )

    def _pose(self, pose):
        p = np.asarray(pose, dtype=np.float32)
        if (
            p.shape != (4, 4)
            or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(p[:3, :3]), 1.0, atol=1e-4)
        ):
            raise ValueError("pose must be a finite rigid 4x4 transform")
        return np.ascontiguousarray(p).ravel()

    def integrate(self, depth, rgb, pose):
        d = np.asarray(depth)
        shape = (self.camera.height, self.camera.width)
        if d.dtype != np.float32 or d.shape != shape or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("depth must be finite nonnegative float32 HxW")
        color = (
            np.full((*shape, 3), 180.0, np.float32)
            if rgb is None
            else np.asarray(rgb, dtype=np.float32)
        )
        if (
            color.shape != (*shape, 3)
            or not np.isfinite(color).all()
            or np.any(color < 0)
            or np.any(color > 255)
        ):
            raise ValueError("RGB must have HxWx3 shape and values in [0,255]")
        self.depth_buffer.copy_from_numpy(np.ascontiguousarray(d).ravel())
        self.rgb_buffer.copy_from_numpy(np.ascontiguousarray(color).ravel())
        self.pose_buffer.copy_from_numpy(self._pose(pose))
        encoder = self.runtime.device.create_command_encoder()
        self.record_integrate(self.depth_buffer, self.rgb_buffer, self.pose_buffer, encoder)
        self.runtime.device.submit_command_buffer(encoder.finish())

    def raycast(self, pose):
        self.pose_buffer.copy_from_numpy(self._pose(pose))
        encoder = self.runtime.device.create_command_encoder()
        self.record_raycast(self.pose_buffer, self.depth_output, encoder)
        self.runtime.device.submit_command_buffer(encoder.finish())
        return (
            self.depth_output.to_numpy()
            .view(np.float32)
            .reshape(self.camera.height, self.camera.width)
            .copy()
        )

    def stats(self):
        c = self.buffers["counters"].to_numpy().view(np.uint32).ravel()
        return dict(
            blocks=min(int(c[0]), self.max_blocks),
            overflow=int(c[1]),
            active=int(c[2]),
            surfaces=int(c[3]),
            domain_errors=int(c[4]),
            probe_exhaustion=int(c[5]),
        )

    def snapshot(self):
        """Explicit diagnostic readback, never used by GPU processing."""
        n = self.stats()["blocks"]
        encoded = self.buffers["blockKeys"].to_numpy().view(np.uint32).ravel()[:n].copy()
        keys = (
            np.stack([(encoded >> shift) & 1023 for shift in (0, 10, 20)], axis=1).astype(np.int32)
            - 512
        )
        result = dict(keys=keys)
        for name, dtype, channels in (
            ("distances", np.float32, 1),
            ("weights", np.float32, 1),
            ("colorWeights", np.float32, 1),
            ("colors", np.float32, 3),
        ):
            data = self.buffers[name].to_numpy().view(dtype).ravel()[: n * 512 * channels]
            result[name] = data.reshape((n, 512, channels) if channels > 1 else (n, 512)).copy()
        return result

    def __len__(self):
        return int(np.count_nonzero(self.buffers["weights"].to_numpy().view(np.float32)))

    def export_ply(self, path):
        import slangpy as spy

        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        for name in ("surfacePoints", "surfaceColors"):
            if name not in self.buffers:
                self.buffers[name] = self.runtime.device.create_buffer(
                    element_count=self.surface_capacity * 3, struct_size=4, usage=usage
                )
        encoder = self.runtime.device.create_command_encoder()
        self._dispatch("resetSurface", 1, encoder, dict(counters=self.buffers["counters"]))
        names = (
            "hashKeys",
            "hashValues",
            "blockKeys",
            "counters",
            "distances",
            "weights",
            "colors",
            "surfacePoints",
            "surfaceColors",
        )
        self._dispatch(
            "extractSurface", self.max_blocks * 512, encoder, {k: self.buffers[k] for k in names}
        )
        self.runtime.device.submit_command_buffer(encoder.finish())
        count = self.stats()["surfaces"]
        if count > self.surface_capacity:
            raise RuntimeError("GPU surface export capacity exceeded")
        points = (
            self.buffers["surfacePoints"]
            .to_numpy()
            .view(np.float32)
            .ravel()[: count * 3]
            .reshape(-1, 3)
        )
        color = (
            np.clip(
                np.rint(
                    self.buffers["surfaceColors"].to_numpy().view(np.float32).ravel()[: count * 3]
                ),
                0,
                255,
            )
            .astype(np.uint8)
            .reshape(-1, 3)
        )
        with Path(path).open("w") as stream:
            stream.write(
                f"ply\nformat ascii 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            )
            for p, c in zip(points, color):
                stream.write(" ".join([*(f"{v:.6f}" for v in p), *(str(int(v)) for v in c)]) + "\n")
