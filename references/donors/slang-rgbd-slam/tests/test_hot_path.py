"""Static checks preserve the reviewed recording path and its delegated dependencies."""

import ast
import inspect
import textwrap

from slang_rgbd_slam.mapping import VoxelMap
from slang_rgbd_slam.tracker import VulkanTracker

FORBIDDEN = {
    "to_numpy",
    "copy_from_numpy",
    "compile",
    "buffer",
    "create_buffer",
    "wait_for_idle",
    "readback",
    "prepare",
    "step",
    "dispatch_arrays",
    "submit_command_buffer",
    "load_module_from_source",
    "create_compute_kernel",
    "numpy",
    "zeros",
    "empty",
    "ones",
    "arange",
}


def tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def assert_cached_recording(fn, encoder_name, allowed=()):
    parsed = tree(fn)
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        assert name not in FORBIDDEN - set(allowed), f"{fn.__qualname__}: forbidden {name}"
        if name == "dispatch":
            value = next((k.value for k in node.keywords if k.arg == "command_encoder"), None)
            assert isinstance(value, ast.Name) and value.id == encoder_name
    return parsed


def test_tracking_hot_path_cached_and_batched():
    parsed = assert_cached_recording(VulkanTracker._record_track, "command_encoder")
    assert_cached_recording(VulkanTracker._record_maps, "encoder")
    assert_cached_recording(VulkanTracker._map_params, "encoder")
    # Only pyramid-level and fixed-iteration loops are permitted on the host.
    loops = [n for n in ast.walk(parsed) if isinstance(n, (ast.For, ast.While))]
    assert len(loops) == 2 and all(isinstance(n, ast.For) for n in loops)
    assert isinstance(loops[0].iter, ast.Call) and loops[0].iter.func.id == "zip"
    assert isinstance(loops[1].iter, ast.Call) and loops[1].iter.func.id == "range"
    assert isinstance(loops[1].iter.args[0], ast.Name) and loops[1].iter.args[0].id == "iterations"
    calls = [n for n in ast.walk(parsed) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "_record_maps" for n in calls)


def test_gauss_newton_delegate_uses_prepared_buffers():
    # The optional GPU package is checked when installed; no GPU is constructed.
    import pytest

    gn = pytest.importorskip("caspar_slang.gn")
    parsed = assert_cached_recording(gn.GaussNewton.dispatch, "encoder", ("submit_command_buffer",))
    # Own dispatch may submit/create an encoder only when none was supplied.
    for node in ast.walk(parsed):
        if isinstance(node, ast.If):
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if any(
                isinstance(n.func, ast.Attribute) and n.func.attr == "submit_command_buffer"
                for n in calls
            ):
                assert isinstance(node.test, ast.Compare)
                assert (
                    isinstance(node.test.left, ast.Name) and node.test.left.id == "command_encoder"
                )
                assert (
                    isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value is None
                )


def test_voxel_integration_vectorized():
    assert not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree(VoxelMap.integrate)))
