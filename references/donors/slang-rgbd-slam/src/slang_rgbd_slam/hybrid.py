"""Dense GPU tracking with resident keyframes, local dense BA and sparse loop support."""

from pathlib import Path

import numpy as np

from .dense_ba import DenseBA
from .dense_tracker import FusedVulkanTracker
from .features import GPUFeatures
from .gpu_slam import GPUSLAM
from .hardware import load_profile, query_device
from .keyframes import GPUKeyframes
from .loop_verification import RGBDLoopVerifier


class HybridSLAM(GPUSLAM):
    def __init__(
        self,
        camera,
        *,
        runtime=None,
        tracker_options=None,
        map_options=None,
        max_keyframes=16,
        keyframe_interval=5,
        ba_every_keyframes=4,
        enable_local_ba=True,
        enable_loops=True,
        min_loop_frames=30,
        max_loop_edges=128,
        max_frames=65536,
        max_features=512,
        tuning_profile=None,
    ):
        from caspar_slang.primitives import primitives_source
        from caspar_slang.runtime import VulkanRuntime

        if (
            not isinstance(keyframe_interval, int)
            or keyframe_interval < 1
            or not isinstance(ba_every_keyframes, int)
            or ba_every_keyframes < 1
        ):
            raise ValueError("keyframe/BA cadence must be positive integers")
        if (
            not isinstance(max_keyframes, int)
            or isinstance(max_keyframes, bool)
            or not 2 <= max_keyframes <= 64
            or not isinstance(max_frames, int)
            or isinstance(max_frames, bool)
            or not 1 <= max_frames <= 1000000
            or not isinstance(max_loop_edges, int)
            or isinstance(max_loop_edges, bool)
            or not 1 <= max_loop_edges <= 256 - max_keyframes
            or not isinstance(min_loop_frames, int)
            or min_loop_frames < 1
        ):
            raise ValueError("invalid bounded loop/history configuration")
        runtime = runtime if runtime is not None else VulkanRuntime()
        profile_path = (
            Path(tuning_profile)
            if tuning_profile is not None
            else Path.home() / ".cache/slang-rgbd-slam/tuning.json"
        )
        profile = load_profile(runtime.device, profile_path)
        options = dict(tracker_options or {})
        options.setdefault("group_size", profile["selected_group_size"] if profile else 64)
        options.setdefault(
            "reduction",
            profile["selected_reduction"]
            if profile
            else ("subgroup" if query_device(runtime.device)["wave_ops"] is True else "workgroup"),
        )
        super().__init__(
            camera,
            runtime=runtime,
            tracker_factory=FusedVulkanTracker,
            tracker_options=options,
            map_options=map_options,
        )
        self.tuning_profile = profile
        self.frontend = GPUFeatures(
            camera,
            self.runtime,
            max_features=max_features,
            matcher_group_size=profile["selected_matcher_group_size"] if profile else 64,
        )
        self.current_features = self.frontend.create_set()
        self.keyframe_store = GPUKeyframes(
            camera,
            self.tracker.levels[-1]["camera"],
            self.frontend,
            capacity=max_keyframes,
            min_loop_frames=min_loop_frames,
        )
        self.dense_ba = DenseBA(self.runtime, self.tracker.levels[-1]["camera"])
        self.ba_problem = self.dense_ba.prepare()
        self.keyframe_interval, self.ba_every_keyframes = keyframe_interval, ba_every_keyframes
        self.enable_local_ba, self.enable_loops = enable_local_ba, enable_loops
        self.active_keyframes = []
        self.keyframe_timestamps = {}
        self.keyframes = 0
        self._pending_capture = None
        self._ran_ba = False
        self._retrieval_recorded = False
        self._pending_candidate = None
        self._pending_query = None
        self._recovery = False
        self.ba_events = []
        self.loop_events = []

        self.max_frames, self.max_loop_edges = max_frames, max_loop_edges
        self._slot_ids = {}
        self._saved_loops = []
        self._loop_attempt = None
        self._loop_recorded = False
        self._pending_candidate_id = None
        self._pending_query_id = None
        self.history_timestamps = []
        self.history = {
            "historyOldNodes": self.runtime.buffer(np.zeros(max_keyframes * 16, np.float32)),
            "historyNewNodes": self.runtime.buffer(np.zeros(max_keyframes * 16, np.float32)),
            "historyPoses": self.runtime.buffer(np.zeros(max_frames * 16, np.float32)),
            "historyValid": self.runtime.buffer(np.zeros(max_frames, np.uint32)),
            "historyAnchors": self.runtime.buffer(np.zeros(max_frames, np.uint32)),
            "historyAnchorIds": self.runtime.buffer(np.zeros(max_frames, np.uint32)),
            "historyGuard": self.runtime.buffer(np.zeros(2, np.uint32)),
        }
        self.loop_event = self.runtime.buffer(np.zeros(2, np.uint32))
        self.graph_guard = self.runtime.buffer(np.array([0, 1], np.uint32))
        loop_source = (
            primitives_source()
            + "\n"
            + Path(__file__).with_name("shaders").joinpath("loop_pipeline.slang").read_text()
        )
        names = (
            "resetLoopEvent",
            "setGraphInformation",
            "prepareLoopEdge",
            "graphAcceptance",
            "commitGraphNode",
            "correctCurrentPose",
            "retainLoopMeasurement",
            "makeHistoryGuard",
            "correctHistory",
            "storeHistory",
        )
        self.loop_kernels = {name: self.runtime.compile(loop_source, entry=name) for name in names}
        if enable_loops:
            from caspar_slang.pose_graph import PoseGraph

            self.loop_verifier = RGBDLoopVerifier(camera, self.runtime, max_features=max_features)
            self.verification_problem = self.loop_verifier.prepare()
            self.loop_matches = self.frontend._buffer(max_features)
            self.pose_graph = PoseGraph(self.runtime, max_nodes=max_keyframes, max_edges=256)
            self.graph_problem = self.pose_graph.prepare(np.empty((0, 2), np.uint32))
            self.loop_measurements = [
                self.runtime.buffer(np.eye(4, dtype=np.float32).ravel())
                for _ in range(max_loop_edges)
            ]

    def _loop_params(self, **updates):
        params = dict(
            capacity=self.keyframe_store.capacity,
            edgeIndex=0,
            querySlot=0,
            nodeIndex=0,
            historyCount=self.index,
            currentFrame=self.index,
            anchorSlot=0,
            ranBA=0,
            targetSlot=0,
            queryPhysicalSlot=0,
            queryFrameId=0,
            targetFrameId=0,
            odometryEdges=0,
            activeEdges=0,
            proposalEnabled=0,
        )
        params.update(updates)
        return params

    def _loop_dispatch(self, name, count, encoder, buffers, **params):
        self.loop_kernels[name].dispatch(
            thread_count=[count, 1, 1],
            vars={"lp": self._loop_params(**params), **buffers},
            command_encoder=encoder,
        )
        encoder.global_barrier()

    def _prune_loops(self):
        self._saved_loops = [
            edge
            for edge in self._saved_loops
            if self._slot_ids.get(edge["query_slot"]) == edge["query_id"]
            and self._slot_ids.get(edge["candidate_slot"]) == edge["candidate_id"]
        ]

    def _pending_is_valid(self):
        return (
            self._pending_candidate in self.active_keyframes
            and self._pending_query in self.active_keyframes
            and self._slot_ids.get(self._pending_candidate) == self._pending_candidate_id
            and self._slot_ids.get(self._pending_query) == self._pending_query_id
            and (self._pending_candidate != self._pending_query or self._recovery)
        )

    def _dispatch_frame(self, source_depth, rgb_buffer, timestamp, started):
        if self.index >= self.max_frames:
            raise ValueError(
                "GPU trajectory capacity exhausted; increase max_frames before processing"
            )
        self._pending_capture = None
        self._ran_ba = False
        self._retrieval_recorded = False
        self._loop_recorded = False
        self._loop_attempt = None
        self._prune_loops()
        if not self._pending_is_valid():
            self._pending_candidate = None
        frame_index = self.index
        result = super()._dispatch_frame(source_depth, rgb_buffer, timestamp, started)
        self.history_timestamps.append(float(timestamp))
        if self._pending_capture is not None and result["success"]:
            slot = self._pending_capture
            if slot in self.active_keyframes:
                self.active_keyframes.remove(slot)
            self.active_keyframes.append(slot)
            self.keyframe_timestamps[slot] = float(timestamp)
            self._slot_ids[slot] = frame_index + 1
            self.keyframes += 1
        self._prune_loops()
        if self._ran_ba:
            status = int(self.ba_problem.buffers["status"].to_numpy().view(np.uint32).ravel()[0])
            costs = self.ba_problem.buffers["costs"].to_numpy().view(np.float32).ravel()
            event = dict(
                timestamp=float(timestamp),
                success=status == 0 and result["success"],
                window=min(4, len(self.active_keyframes)),
                cost_before=float(costs[0]),
                cost_after=float(costs[1]),
            )
            self.ba_events.append(event)
            result["local_ba"] = event
        if self._loop_recorded:
            flags = self.loop_event.to_numpy().view(np.uint32).ravel()
            stats = self.verification_problem.stats.to_numpy().view(np.uint32).ravel()
            cost = float(self.verification_problem.cost.to_numpy().view(np.float32).ravel()[0])
            graph_stats = self.graph_problem.stats.to_numpy().view(np.uint32).ravel()
            graph_cost = self.graph_problem.cost.to_numpy().view(np.float32).ravel()
            attempt = self._loop_attempt
            event = dict(
                timestamp=float(timestamp),
                candidate_slot=attempt["candidate_slot"],
                query_slot=attempt["query_slot"],
                verified=int(stats[1]) == 0,
                accepted=int(flags[1]) != 0,
                recovered=int(flags[0]) == 2,
                inliers=int(stats[0]),
                rmse_px=float(np.sqrt(cost / max(1, int(stats[0])))),
                graph_success=int(flags[1]) != 0,
                graph_valid_edges=int(graph_stats[0]),
                graph_status=int(graph_stats[1]),
                graph_edges=attempt["graph_edges"],
                graph_cost_before=float(graph_cost[0]),
                graph_cost_after=float(graph_cost[1]),
            )
            if event["accepted"]:
                edge = {
                    key: attempt[key]
                    for key in ("candidate_slot", "query_slot", "candidate_id", "query_id", "pool")
                }
                self._saved_loops = [
                    old for old in self._saved_loops if old["pool"] != edge["pool"]
                ]
                self._saved_loops.append(edge)
                self._prune_loops()
            if event["recovered"]:
                result["status"] = "relocalized"
            self.loop_events.append(event)
            result["loop"] = event
        self._pending_candidate = None
        self._pending_query = None
        if self._retrieval_recorded:
            event = self.keyframe_store.event.to_numpy().view(np.uint32).ravel()
            candidate = int(event[0])
            if candidate != 0xFFFFFFFF and candidate in self.active_keyframes:
                self._pending_candidate = candidate
                self._pending_candidate_id = self._slot_ids[candidate]
                self._pending_query = self.active_keyframes[-1]
                self._pending_query_id = self._slot_ids[self._pending_query]
        self._recovery = not result["success"]
        result["keyframe"] = self._pending_capture is not None and result["success"]
        result["active_keyframes"] = len(self.active_keyframes)
        result["tracking_backend"] = "fused GPU ICP + photometric"
        return result

    def _snapshot_history(self, slots, encoder, name):
        for slot in slots:
            encoder.copy_buffer(
                self.history[name], slot * 64, self.keyframe_store.slots[slot]["pose"], 0, 64
            )
        encoder.global_barrier()

    def _correct_history(self, slots, encoder, ran_ba=False):
        self._snapshot_history(slots, encoder, "historyNewNodes")
        self._loop_dispatch(
            "makeHistoryGuard",
            1,
            encoder,
            dict(
                trackingStatus=self.tracker.status,
                loopEvent=self.loop_event,
                baStatus=self.ba_problem.buffers["status"],
                historyGuard=self.history["historyGuard"],
            ),
            ranBA=int(ran_ba),
        )
        if self.index:
            self._loop_dispatch(
                "correctHistory",
                self.index,
                encoder,
                {
                    **{
                        k: self.history[k]
                        for k in (
                            "historyOldNodes",
                            "historyNewNodes",
                            "historyPoses",
                            "historyValid",
                            "historyAnchors",
                            "historyAnchorIds",
                            "historyGuard",
                        )
                    },
                    "archiveFrameIds": self.keyframe_store.frame_ids,
                },
            )

    def _record_after_tracking(self, encoder, source_depth, rgb_buffer):
        self._loop_dispatch(
            "resetLoopEvent",
            1,
            encoder,
            dict(loopEvent=self.loop_event, graphGuard=self.graph_guard),
        )
        capture = not self.initialized or self.index % self.keyframe_interval == 0
        if capture or self._recovery or self._pending_candidate is not None:
            self.frontend.record_extract(rgb_buffer, source_depth, self.current_features, encoder)
            encoder.global_barrier()
        if self.enable_loops and self._pending_candidate is not None:
            self._record_loop(encoder)
        planned = list(self.active_keyframes)
        if capture:
            if len(planned) < self.keyframe_store.capacity:
                slot = next(i for i in range(self.keyframe_store.capacity) if i not in planned)
            else:
                slot = planned.pop(0)
            planned.append(slot)
            self._pending_capture = slot
            self.keyframe_store.record_capture(
                slot,
                source_depth,
                rgb_buffer,
                self.world_pose,
                self.tracker.levels[-1]["maps"]["source"],
                self.current_features,
                self.tracker.status,
                encoder,
                self.index + 1,
            )
            if (
                self.enable_local_ba
                and not self._loop_recorded
                and len(planned) >= 2
                and (self.keyframes + 1) % self.ba_every_keyframes == 0
            ):
                self._snapshot_history(planned, encoder, "historyOldNodes")
                self._record_local_ba(planned[-4:], encoder)
                self._correct_history(planned, encoder, ran_ba=True)
                self._record_map_rebuild(planned, self.keyframe_store.rebuild_guard, encoder)
                self._ran_ba = True
        if self._loop_recorded:
            replay = [slot for slot in planned if slot != self._pending_capture]
            self._record_map_rebuild(replay, self.graph_guard, encoder)
        if self.enable_loops and (capture or self._recovery):
            self.keyframe_store.record_retrieve(self.current_features, encoder, self.index + 1)
            self._retrieval_recorded = True
        anchor = planned[-1] if planned else 0
        self._loop_dispatch(
            "storeHistory",
            16,
            encoder,
            dict(
                trackingStatus=self.tracker.status,
                currentWorldPose=self.world_pose,
                archiveFrameIds=self.keyframe_store.frame_ids,
                **{
                    k: self.history[k]
                    for k in ("historyPoses", "historyValid", "historyAnchors", "historyAnchorIds")
                },
            ),
            anchorSlot=anchor,
        )
        if self._ran_ba:
            self.keyframe_store.dispatch(
                "makeTailGuard",
                1,
                encoder,
                dict(
                    trackingStatus=self.tracker.status,
                    rebuildGuard=self.keyframe_store.rebuild_guard,
                    tailGuard=self.keyframe_store.tail_guard,
                ),
            )
            encoder.global_barrier()
            return self.keyframe_store.tail_guard
        return None

    def _record_local_ba(self, slots, encoder):
        low = self.tracker.levels[-1]["camera"]
        image_bytes = low.width * low.height * 4
        for row, slot in enumerate(slots):
            saved = self.keyframe_store.slots[slot]
            encoder.copy_buffer(
                self.ba_problem.buffers["depths"],
                row * image_bytes,
                saved["measured_low_depth"],
                0,
                image_bytes,
            )
            encoder.copy_buffer(
                self.ba_problem.buffers["initialDepths"],
                row * image_bytes,
                saved["low_depth"],
                0,
                image_bytes,
            )
            encoder.copy_buffer(
                self.ba_problem.buffers["rgb"],
                row * image_bytes * 3,
                saved["low_rgb"],
                0,
                image_bytes * 3,
            )
            encoder.copy_buffer(self.ba_problem.poses, row * 64, saved["pose"], 0, 64)
        encoder.global_barrier()
        self.dense_ba.record(self.ba_problem, encoder, iterations=3, active_frames=len(slots))
        encoder.global_barrier()
        self.keyframe_store.record_ba_commit(
            self.ba_problem, slots, self.world_pose, self.tracker.status, encoder
        )

    def _record_map_rebuild(self, slots, guard, encoder):
        self.map.record_reset(encoder, tracking_status=guard)
        for slot in slots:
            saved = self.keyframe_store.slots[slot]
            self.map.record_integrate(
                saved["depth"], saved["rgb"], saved["pose"], encoder, tracking_status=guard
            )
        encoder.global_barrier()

    def _record_loop(self, encoder):
        candidate, query = self._pending_candidate, self._pending_query
        slots = list(self.active_keyframes)
        ordered = {slot: index for index, slot in enumerate(slots)}
        candidate_id, query_id = self._slot_ids[candidate], self._slot_ids[query]
        used = {edge["pool"] for edge in self._saved_loops}
        pool = next(
            (i for i in range(self.max_loop_edges) if i not in used),
            self._saved_loops[0]["pool"] if self._saved_loops else 0,
        )
        odometry = max(0, len(slots) - 1)
        existing = list(self._saved_loops)
        edges = [(i, i + 1) for i in range(odometry)]
        edges.extend((ordered[e["candidate_slot"]], ordered[e["query_slot"]]) for e in existing)
        proposal = candidate != query
        proposal_index = len(edges)
        if proposal:
            edges.append((ordered[candidate], ordered[query]))
        self.graph_problem.update_topology(
            np.asarray(edges, np.uint32).reshape(-1, 2), node_count=len(slots)
        )
        for node, slot in enumerate(slots):
            encoder.copy_buffer(
                self.graph_problem.poses, node * 64, self.keyframe_store.slots[slot]["pose"], 0, 64
            )
        encoder.global_barrier()
        self._snapshot_history(slots, encoder, "historyOldNodes")
        self.frontend.record_match(
            self.current_features,
            self.keyframe_store.slots[candidate]["features"],
            self.loop_matches,
            encoder,
        )
        self.loop_verifier.record(
            self.verification_problem,
            self.current_features,
            self.keyframe_store.slots[candidate]["features"],
            self.loop_matches,
            self.keyframe_store.slots[candidate]["pose"],
            encoder,
        )
        self.pose_graph.record_initialize_measurements(
            self.graph_problem, encoder, first_edge=0, edge_count=odometry
        )
        for index, edge in enumerate(existing):
            encoder.copy_buffer(
                self.graph_problem.measurements,
                (odometry + index) * 64,
                self.loop_measurements[edge["pool"]],
                0,
                64,
            )
        encoder.global_barrier()
        self._loop_dispatch(
            "setGraphInformation",
            max(1, len(edges)),
            encoder,
            dict(graphInformation=self.graph_problem.edge_information),
            odometryEdges=odometry,
            activeEdges=len(edges),
            edgeIndex=proposal_index,
            proposalEnabled=int(proposal),
        )
        params = dict(
            edgeIndex=proposal_index,
            querySlot=ordered[query],
            queryPhysicalSlot=query,
            targetSlot=candidate,
            queryFrameId=query_id,
            targetFrameId=candidate_id,
            proposalEnabled=int(proposal),
        )
        self._loop_dispatch(
            "prepareLoopEdge",
            1,
            encoder,
            dict(
                verifiedCameraPose=self.verification_problem.pose,
                verifierStats=self.verification_problem.stats,
                currentWorldPose=self.world_pose,
                trackingStatus=self.tracker.status,
                queryOldWorldPose=self.keyframe_store.slots[query]["pose"],
                targetWorldPose=self.keyframe_store.slots[candidate]["pose"],
                archiveFrameIds=self.keyframe_store.frame_ids,
                graphMeasurements=self.graph_problem.measurements,
                graphInformation=self.graph_problem.edge_information,
                loopEvent=self.loop_event,
            ),
            **params,
        )
        self.pose_graph.record(self.graph_problem, encoder, iterations=3, cg_iterations=20)
        self._loop_dispatch(
            "graphAcceptance",
            1,
            encoder,
            dict(
                loopEvent=self.loop_event,
                graphStats=self.graph_problem.stats,
                trackingStatus=self.tracker.status,
                graphGuard=self.graph_guard,
            ),
        )
        for node, slot in enumerate(slots):
            self._loop_dispatch(
                "commitGraphNode",
                16,
                encoder,
                dict(
                    graphGuard=self.graph_guard,
                    graphPoses=self.graph_problem.poses,
                    archivePose=self.keyframe_store.slots[slot]["pose"],
                ),
                nodeIndex=node,
            )
        self._loop_dispatch(
            "correctCurrentPose",
            1,
            encoder,
            dict(
                graphGuard=self.graph_guard,
                graphBackup=self.graph_problem.buffers["backup"],
                graphPoses=self.graph_problem.poses,
                currentWorldPose=self.world_pose,
            ),
            querySlot=ordered[query],
        )
        if proposal:
            self._loop_dispatch(
                "retainLoopMeasurement",
                16,
                encoder,
                dict(
                    graphGuard=self.graph_guard,
                    graphMeasurements=self.graph_problem.measurements,
                    savedLoopMeasurement=self.loop_measurements[pool],
                ),
                edgeIndex=proposal_index,
            )
        self._correct_history(slots, encoder)
        self._loop_attempt = dict(
            candidate_slot=candidate,
            query_slot=query,
            candidate_id=candidate_id,
            query_id=query_id,
            pool=pool,
            graph_edges=len(edges),
        )
        self._loop_recorded = True

    def corrected_trajectory(self):
        """Final explicit GPU history readback, outside frame timing.

        Frame poses are corrected with their still-resident keyframe generation.
        Once that anchor is evicted, its frames retain their last valid correction.
        """
        count = len(self.history_timestamps)
        valid = self.history["historyValid"].to_numpy().view(np.uint32).ravel()[:count]
        poses = (
            self.history["historyPoses"]
            .to_numpy()
            .view(np.float32)
            .reshape(self.max_frames, 4, 4)[:count]
        )
        return [
            (timestamp, pose.astype(np.float64).copy())
            for timestamp, pose, ok in zip(self.history_timestamps, poses, valid)
            if ok
        ]
