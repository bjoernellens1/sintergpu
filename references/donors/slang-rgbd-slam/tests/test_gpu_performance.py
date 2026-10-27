"""Hot-path budgets independent of GPU model and timing noise.

These contracts catch frame-sized readbacks, capacity-sized dispatches, and
unbounded probing before a numerical GPU run can hide their scaling cost.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "slang_rgbd_slam"


def method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
    ]


def shader_function(name):
    source = (ROOT / "shaders" / "tsdf.slang").read_text()
    match = re.search(r"\b" + name + r"\s*\([^)]*\)\s*\{", source)
    assert match, name
    start = match.end()
    depth = 1
    end = start
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start : end - 1]


def test_gpu_recording_has_no_host_synchronization():
    for name in ("record_integrate", "record_raycast"):
        node = method("gpu_mapping.py", "GPUTSDF", name)
        assert not calls(node, "to_numpy"), f"{name} readbacks stall frame recording"
        assert not calls(node, "copy_from_numpy"), f"{name} must consume persistent device inputs"
        assert not calls(node, "submit_command_buffer"), f"{name} must share the frame encoder"
        assert not calls(node, "stats")
        assert not calls(node, "snapshot")


def test_pipeline_has_one_submission_and_only_small_diagnostics():
    process = method("gpu_slam.py", "GPUSLAM", "process")
    device = method("gpu_slam.py", "GPUSLAM", "process_device")
    node = method("gpu_slam.py", "GPUSLAM", "_dispatch_frame")
    assert len(calls(node, "submit_command_buffer")) == 1
    # CPU ingest uploads images; device ingest shares the same frame dispatch.
    assert len(calls(process, "copy_from_numpy")) == 2
    assert not calls(device, "copy_from_numpy")
    assert not calls(device, "to_numpy")
    assert not calls(node, "copy_from_numpy")
    readbacks = calls(node, "to_numpy")
    assert len(readbacks) <= 3
    for call in readbacks:
        receiver = call.func.value
        assert isinstance(receiver, ast.Attribute)
        assert receiver.attr in {"world_pose", "status", "diagnostics"}
    stats_calls = calls(node, "stats")
    assert len(stats_calls) <= 1
    for call in stats_calls:
        assert isinstance(call.func.value, ast.Attribute) and call.func.value.attr == "map"
    stats_node = method("gpu_mapping.py", "GPUTSDF", "stats")
    for call in calls(stats_node, "to_numpy"):
        assert (
            ast.unparse(call.func.value) == 'self.buffers["counters"]'
            or ast.unparse(call.func.value) == "self.buffers['counters']"
        )
    assert not calls(node, "snapshot")


def test_hash_probing_work_is_bounded_independently_of_capacity():
    for name in ("lookup", "allocate"):
        body = shader_function(name)
        assert not re.search(r"for\s*\([^;]*;[^;]*<\s*cfg\.hashCapacity\s*;", body), (
            f"{name} must cap probe work; a full table cannot scan its entire capacity per ray sample"
        )


def test_voxel_fusion_dispatch_uses_active_work_not_map_capacity():
    node = method("gpu_mapping.py", "GPUTSDF", "record_integrate")
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Mult):
            expression = ast.unparse(child)
            assert not ("max_blocks" in expression and "512" in expression), (
                "Fusion must derive launch dimensions on GPU from active block count"
            )
    tree = ast.parse((ROOT / "gpu_mapping.py").read_text())
    assert calls(tree, "dispatch_compute_indirect"), (
        "Active block launch must not read count to CPU"
    )


def test_raycast_work_limit_follows_geometry():
    body = shader_function("raycast")
    limit = re.search(r"uint\s+limit\s*=\s*([^;]+)", body)
    assert limit
    assert "hashCapacity" not in limit[1]
    assert "voxelSize" in limit[1]
    assert "farDepth" in body and "nearDepth" in body


def test_color_fusion_tracks_its_own_observation_confidence():
    source = (ROOT / "shaders" / "tsdf.slang").read_text()
    assert re.search(r"RWStructuredBuffer<float>\s+colorWeights\s*;", source), (
        "TSDF free-space observations must not dilute later RGB surface fusion"
    )
    body = shader_function("integrateVoxels")
    assert "colorWeights[" in body
