"""Persistent GPU frame-to-model RGB-D tracking and sparse TSDF mapping."""

from pathlib import Path
from time import perf_counter

import numpy as np

from .gpu_mapping import GPUTSDF
from .tracker import VulkanTracker


class GPUSLAM:
    """One command submission per frame; only pose/status diagnostics return to host."""

    def __init__(
        self,
        camera,
        *,
        tracker_options=None,
        map_options=None,
        runtime=None,
        tracker_factory=VulkanTracker,
    ):
        from caspar_slang.primitives import primitives_source
        from caspar_slang.runtime import VulkanRuntime

        self.camera = camera
        self.runtime = runtime if runtime is not None else VulkanRuntime()
        self.tracker = tracker_factory(camera, runtime=self.runtime, **(tracker_options or {}))
        self.map = GPUTSDF(camera, runtime=self.runtime, **(map_options or {}))
        self.world_pose = self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
        self.reference_pose = self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
        self.rgb = self.runtime.buffer(np.zeros(camera.width * camera.height * 3, np.float32))
        source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("gpu_pipeline.slang").read_text()
        )
        self.kernels = {
            name: self.runtime.compile(source, entry=name)
            for name in ("resetRelative", "composeWorld")
        }
        self._owned_depth = self.tracker.levels[0]["maps"]["source"]["depth"]
        self._owned_rgb = self.rgb
        self.initialized = False
        self.index = 0
        self.last_timestamp = None
        self.trajectory = []
        self.results = []

    def process(self, depth, rgb=None, *, timestamp=0.0):
        started = perf_counter()
        depth = self.tracker._depth(np.asarray(depth, dtype=np.float32))
        if not np.isfinite(timestamp) or (
            self.last_timestamp is not None and timestamp <= self.last_timestamp
        ):
            raise ValueError("Timestamps must be finite and increase")
        if rgb is None:
            packed = np.full(depth.size * 3, 180, np.float32)
        else:
            rgb = np.asarray(rgb)
            if rgb.shape != (self.camera.height, self.camera.width, 3) or rgb.dtype != np.uint8:
                raise ValueError("RGB must be HxWx3 uint8")
            packed = rgb.astype(np.float32).ravel()
        self._owned_depth.copy_from_numpy(depth)
        self._owned_rgb.copy_from_numpy(packed)
        return self._dispatch_frame(self._owned_depth, self._owned_rgb, timestamp, started)

    def process_device(self, depth_buffer, rgb_buffer, *, timestamp=0.0):
        """Consume GPU float32 depth and RGB(0..255) buffers without image transfers.

        Imported sensor resources must have camera-sized contiguous storage and
        remain valid until this call returns. Pose/diagnostics synchronize the frame.
        This API does not itself implement external-memory sensor DMA import.
        """
        if not np.isfinite(timestamp) or (
            self.last_timestamp is not None and timestamp <= self.last_timestamp
        ):
            raise ValueError("Timestamps must be finite and increase")
        if (
            depth_buffer.size != self.camera.width * self.camera.height * 4
            or rgb_buffer.size != self.camera.width * self.camera.height * 12
        ):
            raise ValueError("GPU input buffer sizes must match float32 camera depth and RGB")
        return self._dispatch_frame(depth_buffer, rgb_buffer, timestamp, perf_counter())

    def _record_after_tracking(self, encoder, source_depth, rgb_buffer):
        """Extension point for GPU keyframe optimization and graph corrections."""

    def _dispatch_frame(self, source_depth, rgb_buffer, timestamp, started):
        self.tracker.levels[0]["maps"]["source"]["depth"] = source_depth
        self.rgb = rgb_buffer
        reference_depth = self.tracker.levels[0]["maps"]["reference"]["depth"]
        image_bytes = self.camera.width * self.camera.height * 4
        encoder = self.runtime.device.create_command_encoder()
        encoder.copy_buffer(self.reference_pose, 0, self.world_pose, 0, 64)
        self.kernels["resetRelative"].dispatch(
            thread_count=[1, 1, 1],
            vars=dict(
                relativePose=self.tracker.pose,
                initialPose=self.tracker.initial_pose,
                trackingStatus=self.tracker.status,
            ),
            command_encoder=encoder,
        )
        encoder.global_barrier()
        fused = hasattr(self.tracker, "record_intensity")
        if fused:
            self.tracker.record_intensity(self.rgb, encoder)
        if self.initialized:
            if fused:
                self.map.record_raycast(
                    self.reference_pose,
                    reference_depth,
                    encoder,
                    intensity_output=self.tracker.levels[0]["maps"]["reference"]["intensity"],
                )
            else:
                self.map.record_raycast(self.reference_pose, reference_depth, encoder)
        else:
            encoder.copy_buffer(reference_depth, 0, source_depth, 0, image_bytes)
            if fused:
                encoder.global_barrier()
                encoder.copy_buffer(
                    self.tracker.levels[0]["maps"]["reference"]["intensity"],
                    0,
                    self.tracker.levels[0]["maps"]["source"]["intensity"],
                    0,
                    image_bytes,
                )
        encoder.global_barrier()
        self.tracker._record_maps("reference", encoder)
        self.tracker._record_track(encoder)
        encoder.global_barrier()
        self.kernels["composeWorld"].dispatch(
            thread_count=[1, 1, 1],
            vars=dict(
                worldPose=self.world_pose,
                referenceWorldPose=self.reference_pose,
                relativePose=self.tracker.pose,
                trackingStatus=self.tracker.status,
            ),
            command_encoder=encoder,
        )
        encoder.global_barrier()
        integration_status = self._record_after_tracking(encoder, source_depth, rgb_buffer)
        encoder.global_barrier()
        self.map.record_integrate(
            source_depth,
            self.rgb,
            self.world_pose,
            encoder,
            tracking_status=self.tracker.status
            if integration_status is None
            else integration_status,
        )
        self.runtime.device.submit_command_buffer(encoder.finish())
        pose = self.world_pose.to_numpy().view(np.float32).reshape(4, 4).astype(np.float64)
        status = self.tracker.status.to_numpy().view(np.uint32).ravel()
        cost = float(self.tracker.diagnostics.to_numpy().view(np.float32).ravel()[0])
        success = int(status[1]) == 0
        mapping_stats = self.map.stats()
        was_initialized = self.initialized
        self.initialized |= success
        self.last_timestamp = float(timestamp)
        result = dict(
            timestamp=float(timestamp),
            success=success,
            status=("tracked" if was_initialized else "initialized") if success else "lost",
            world_from_camera=pose if success else None,
            mapping_success=mapping_stats["overflow"] == 0
            and mapping_stats["domain_errors"] == 0
            and mapping_stats["probe_exhaustion"] == 0,
            mapping=mapping_stats,
            correspondences=int(status[0]),
            rmse=float(np.sqrt(cost / max(1, int(status[0])))),
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        if success:
            self.trajectory.append((float(timestamp), pose.copy()))
        self.results.append(result)
        self.index += 1
        return result
