"""Static residency and bounded-work contracts for sparse keyframe support."""

import inspect
from pathlib import Path

from slang_rgbd_slam.gpu_mapping import GPUTSDF
from slang_rgbd_slam.hybrid import HybridSLAM
from slang_rgbd_slam.keyframes import GPUKeyframes


def test_keyframe_storage_is_bounded_and_capture_remains_gpu_resident():
    assert "capacity <= 64" in inspect.getsource(GPUKeyframes.__init__)
    for method in (
        GPUKeyframes.record_capture,
        GPUKeyframes.record_retrieve,
        GPUKeyframes.record_ba_commit,
        HybridSLAM._record_local_ba,
        HybridSLAM._record_map_rebuild,
    ):
        source = inspect.getsource(method)
        for forbidden in (
            "to_numpy(",
            "create_buffer(",
            "submit_command_buffer(",
            "copy_from_numpy(",
            "np.linalg.solve(",
        ):
            assert forbidden not in source


def test_sparse_frontend_has_keyframe_and_recovery_cadence():
    source = inspect.getsource(HybridSLAM._record_after_tracking)
    assert "self.index % self.keyframe_interval" in source
    assert "if capture or self._recovery or self._pending_candidate is not None" in source
    assert "self._record_local_ba(planned[-4:], encoder)" in source


def test_failed_ba_guard_protects_map_reset_and_reintegration():
    shader = Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/tsdf.slang"
    initialize = (
        shader.read_text().split("void initialize(", 1)[1].split('[shader("compute")]', 1)[0]
    )
    assert "if(trackingStatus[1]!=0)return;" in initialize
    reset = inspect.getsource(GPUTSDF.record_reset)
    assert '"trackingStatus"' in reset
    rebuild = inspect.getsource(HybridSLAM._record_map_rebuild)
    assert "tracking_status=guard" in rebuild


def test_ba_commit_updates_low_depth_and_uses_consistent_pixel_rounding():
    source = (Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/keyframes.slang").read_text()
    commit = source.split("void commitBA(", 1)[1].split('[shader("compute")]', 1)[0]
    assert "savedLowDepth" in commit
    assert "floor(p.xy+.5)" in source
    assert "min(featureCount[0],kf.maxFeatures)" in source


def test_graph_and_history_remain_device_resident_until_explicit_export():
    for name in ("_record_loop", "_snapshot_history", "_correct_history"):
        source = inspect.getsource(getattr(HybridSLAM, name))
        for forbidden in ("to_numpy(", "np.linalg", "create_buffer(", "submit_command_buffer("):
            assert forbidden not in source
    frame = inspect.getsource(HybridSLAM._dispatch_frame)
    assert "corrected_trajectory(" not in frame
    assert 'self.history["historyPoses"]' not in frame
    assert 'self.history["historyValid"]' not in frame
    assert "self.graph_problem.poses.to_numpy" not in frame
    assert "self.graph_problem.measurements.to_numpy" not in frame
    export = inspect.getsource(HybridSLAM.corrected_trajectory)
    assert 'self.history["historyValid"].to_numpy()' in export
    assert 'self.history["historyPoses"]' in export and ".to_numpy()" in export
    record = inspect.getsource(HybridSLAM._record_loop)
    assert "self.pose_graph.record(" in record
    assert "iterations=3, cg_iterations=20" in record
    assert "self._saved_loops" in record and "self.max_loop_edges" in record


def test_history_corrections_check_resident_anchor_generation():
    source = (
        Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/loop_pipeline.slang"
    ).read_text()
    history = source.split("void correctHistory(", 1)[1].split('[shader("compute")]', 1)[0]
    assert "index>=lp.historyCount" in history
    assert "slot>=lp.capacity" in history
    assert "historyAnchorIds[index]!=archiveFrameIds[slot]" in history
    assert "graphGuard[1]==0" in source.split("void retainLoopMeasurement(", 1)[1]


def test_ba_observations_are_immutable_and_commit_matches_evaluated_depth():
    source = (Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/keyframes.slang").read_text()
    commit = source.split("void commitBA(", 1)[1].split('[shader("compute")]', 1)[0]
    assert "savedRawDepth[i]/(z*rho)" in commit
    assert "savedLowDepth[i]=1/rho" in commit
    assert "savedDepth[i]*=" not in commit
    assert "ratio>.9" not in commit and "ratio<1.1" not in commit
    assert "savedRawDepth[i]=" not in commit
    assert "savedMeasuredLowDepth[i]=" not in commit
    record = inspect.getsource(HybridSLAM._record_local_ba)
    assert 'saved["measured_low_depth"]' in record
    assert 'self.ba_problem.buffers["initialDepths"]' in record
    shader = (Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/dense_ba.slang").read_text()
    update = shader.split("void updateDenseDepth(", 1)[1].split("// Acceptance", 1)[0]
    assert "clamp(updated,measured/1.1,measured/.9)" in update


def test_ba_depth_prior_is_immutable_and_commit_uses_accepted_state():
    controller = inspect.getsource(HybridSLAM._record_local_ba)
    assert 'saved["measured_low_depth"]' in controller
    assert 'self.ba_problem.buffers["initialDepths"]' in controller
    assert 'saved["low_depth"]' in controller
    shader = (Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/keyframes.slang").read_text()
    commit = shader.split("void commitBA(", 1)[1].split('[shader("compute")]', 1)[0]
    assert "savedRawDepth[i]" in commit
    assert "savedMeasuredLowDepth" in commit
    assert "savedDepth[i]*=" not in commit
    assert "ratio>.9" not in commit and "ratio<1.1" not in commit
    assert "savedRawDepth[i]=" not in commit
    assert "savedMeasuredLowDepth[i]=" not in commit
