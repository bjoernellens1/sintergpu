"""Bounded resident RGB-D keyframes and GPU place-retrieval histograms."""

from pathlib import Path

import numpy as np


class GPUKeyframes:
    def __init__(
        self,
        camera,
        low_camera,
        frontend,
        capacity=32,
        min_loop_frames=30,
        retrieval_threshold=0.85,
    ):
        import slangpy as spy
        from caspar_slang.primitives import primitives_source

        if (
            not isinstance(capacity, int)
            or not 2 <= capacity <= 64
            or min_loop_frames < 1
            or not 0 < retrieval_threshold < 1
        ):
            raise ValueError("invalid keyframe budget or retrieval parameters")
        self.camera, self.low_camera, self.frontend = camera, low_camera, frontend
        self.runtime = frontend.runtime
        self.capacity, self.min_loop_frames, self.retrieval_threshold = (
            capacity,
            min_loop_frames,
            retrieval_threshold,
        )

        def allocate(count):
            return self.runtime.device.create_buffer(
                element_count=count,
                struct_size=4,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            )

        self.slots = [
            dict(
                depth=allocate(camera.width * camera.height),
                raw_depth=allocate(camera.width * camera.height),
                rgb=allocate(camera.width * camera.height * 3),
                low_depth=allocate(low_camera.width * low_camera.height),
                measured_low_depth=allocate(low_camera.width * low_camera.height),
                low_rgb=allocate(low_camera.width * low_camera.height * 3),
                pose=self.runtime.buffer(np.eye(4, dtype=np.float32).ravel()),
                features=frontend.create_set(),
            )
            for _ in range(capacity)
        ]
        self.frame_ids = self.runtime.buffer(np.zeros(capacity, np.uint32))
        self.histograms = self.runtime.buffer(np.zeros(capacity * 256, np.float32))
        self.query_histogram = allocate(256)
        self.scores = allocate(capacity)
        self.event = self.runtime.buffer(np.array([0xFFFFFFFF, 0], np.uint32))
        self.rebuild_guard = self.runtime.buffer(np.zeros(2, np.uint32))
        self.tail_guard = self.runtime.buffer(np.zeros(2, np.uint32))
        source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("keyframes.slang").read_text()
        )
        self.kernels = {
            name: self.runtime.compile(source, entry=name)
            for name in (
                "makeTailGuard",
                "captureImages",
                "captureFeatures",
                "makeBAGuard",
                "commitBA",
                "refreshLandmarks",
                "featureHistogram",
                "retrieveKeyframes",
                "selectCandidate",
            )
        }

    def params(self, **updates):
        c, low = self.camera, self.low_camera
        result = dict(
            width=c.width,
            height=c.height,
            lowWidth=low.width,
            lowHeight=low.height,
            maxFeatures=self.frontend.max_features,
            frameRow=0,
            latest=0,
            frameIndex=0,
            slot=0,
            capacity=self.capacity,
            minLoopFrames=self.min_loop_frames,
            fx=c.fx,
            fy=c.fy,
            cx=c.cx,
            cy=c.cy,
            retrievalThreshold=self.retrieval_threshold,
        )
        result.update(updates)
        return result

    def dispatch(self, name, count, encoder, buffers, **params):
        self.kernels[name].dispatch(
            thread_count=[count, 1, 1],
            vars=dict(kf=self.params(**params), **buffers),
            command_encoder=encoder,
        )

    def record_capture(
        self, slot, depth, rgb, world_pose, low_maps, features, status, encoder, frame_index
    ):
        saved = self.slots[slot]
        self.dispatch(
            "captureImages",
            self.camera.width * self.camera.height,
            encoder,
            dict(
                frameDepth=depth,
                frameRGB=rgb,
                worldPose=world_pose,
                lowDepth=low_maps["depth"],
                lowIntensity=low_maps["intensity"],
                trackingStatus=status,
                savedDepth=saved["depth"],
                savedRawDepth=saved["raw_depth"],
                savedMeasuredLowDepth=saved["measured_low_depth"],
                savedRGB=saved["rgb"],
                savedPose=saved["pose"],
                savedLowDepth=saved["low_depth"],
                savedLowRGB=saved["low_rgb"],
                frameIds=self.frame_ids,
            ),
            slot=slot,
            frameIndex=frame_index,
        )
        self.dispatch(
            "captureFeatures",
            self.frontend.max_features,
            encoder,
            dict(
                featurePositions=features["positions"],
                featureLandmarks=features["landmarks"],
                featureDescriptors=features["descriptors"],
                featureCount=features["count"],
                trackingStatus=status,
                savedPositions=saved["features"]["positions"],
                savedLandmarks=saved["features"]["landmarks"],
                savedDescriptors=saved["features"]["descriptors"],
                savedCount=saved["features"]["count"],
            ),
        )
        encoder.global_barrier()
        self.dispatch(
            "featureHistogram",
            256,
            encoder,
            dict(
                featureDescriptors=saved["features"]["descriptors"],
                featureCount=saved["features"]["count"],
                histograms=self.histograms,
            ),
            slot=slot,
        )
        encoder.global_barrier()

    def record_retrieve(self, features, encoder, frame_index):
        self.dispatch(
            "featureHistogram",
            256,
            encoder,
            dict(
                featureDescriptors=features["descriptors"],
                featureCount=features["count"],
                histograms=self.query_histogram,
            ),
            slot=0,
        )
        encoder.global_barrier()
        self.dispatch(
            "retrieveKeyframes",
            self.capacity,
            encoder,
            dict(
                queryHistogram=self.query_histogram,
                archiveHistograms=self.histograms,
                archiveFrameIds=self.frame_ids,
                retrievalScores=self.scores,
            ),
            frameIndex=frame_index,
        )
        encoder.global_barrier()
        self.dispatch(
            "selectCandidate",
            1,
            encoder,
            dict(retrievalScores=self.scores, retrievalEvent=self.event),
        )
        encoder.global_barrier()

    def record_ba_commit(self, ba_problem, slots, current_world_pose, tracking_status, encoder):
        self.dispatch(
            "makeBAGuard",
            1,
            encoder,
            dict(
                trackingStatus=tracking_status,
                baStatus=ba_problem.buffers["status"],
                rebuildGuard=self.rebuild_guard,
            ),
        )
        encoder.global_barrier()
        for row, slot in enumerate(slots):
            saved = self.slots[slot]
            self.dispatch(
                "commitBA",
                self.camera.width * self.camera.height,
                encoder,
                dict(
                    baPoses=ba_problem.poses,
                    baInverseDepth=ba_problem.inverse_depth,
                    rebuildGuard=self.rebuild_guard,
                    savedDepth=saved["depth"],
                    savedRawDepth=saved["raw_depth"],
                    savedMeasuredLowDepth=saved["measured_low_depth"],
                    savedLowDepth=saved["low_depth"],
                    savedPose=saved["pose"],
                    currentWorldPose=current_world_pose,
                ),
                frameRow=row,
                latest=int(row == len(slots) - 1),
            )
            encoder.global_barrier()
            self.dispatch(
                "refreshLandmarks",
                self.frontend.max_features,
                encoder,
                dict(
                    rebuildGuard=self.rebuild_guard,
                    savedDepth=saved["depth"],
                    savedPositions=saved["features"]["positions"],
                    savedLandmarks=saved["features"]["landmarks"],
                    savedCount=saved["features"]["count"],
                ),
            )
        encoder.global_barrier()
