"""Static contracts keep the resident path separate from convenience readback."""

import ast
from pathlib import Path

import pytest


def test_resident_dispatch_has_no_host_transfer_allocation_or_compilation():
    path = Path(__file__).parents[1] / "src" / "caspar_slang" / "gn.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussNewton"
    )
    dispatch = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch"
    )
    forbidden = {
        "to_numpy",
        "readback",
        "wait",
        "wait_for_idle",
        "wait_for_submit",
        "compile",
        "load_module_from_source",
        "create_compute_kernel",
        "create_buffer",
        "buffer",
        "prepare",
        "pack",
        "unpack",
        "zeros",
        "empty",
    }
    calls = [node for node in ast.walk(dispatch) if isinstance(node, ast.Call)]
    names = [
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        if isinstance(node.func, ast.Name)
        else ""
        for node in calls
    ]
    assert not forbidden.intersection(names)
    assert names.count("create_command_encoder") == 1
    assert names.count("submit_command_buffer") == 1
    # Only the logarithmic reduction chain may be an explicit Python loop.
    loops = [node for node in ast.walk(dispatch) if isinstance(node, (ast.For, ast.While))]
    assert len(loops) == 1
    assert "problem.partials" in ast.unparse(loops[0].iter)
    gpu_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "dispatch"
    ]
    assert len(gpu_calls) == 4
    assert all(
        any(keyword.arg == "command_encoder" for keyword in node.keywords) for node in gpu_calls
    )


@pytest.mark.parametrize(
    "kwargs", [{"size": 0}, {"iterations": 0}, {"size": True}, {"iterations": 1.5}]
)
def test_benchmark_validates_before_loading_gpu(kwargs):
    from caspar_slang.benchmark import run_benchmark

    with pytest.raises(ValueError, match="positive integer"):
        run_benchmark(**kwargs)
