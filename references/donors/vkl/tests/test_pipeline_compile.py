from __future__ import annotations

import torch

from vkl.pipeline import NodeSpec, PipelineExecutor, PipelineSpec


def _spec() -> PipelineSpec:
    return PipelineSpec(
        "compile-status",
        (NodeSpec("rescale", "canonical_rescale", {"rgb": "input.rgb"}),),
    )


def test_executor_reports_compile_fallback_without_claiming_compilation(monkeypatch):
    def fail_compile(_fn, **_kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", fail_compile)
    executor = PipelineExecutor(_spec(), compile_nodes=True)

    assert executor.compile_requested
    assert executor.compile_actual == "eager-fallback"
    assert executor.compile_nodes == {"rescale": "eager-fallback"}


def test_executor_marks_compilation_only_after_successful_execution(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    executor = PipelineExecutor(_spec(), compile_nodes=True)

    assert executor.compile_actual == "pending"
    executor.run({"rgb": torch.rand(1, 3, 8, 8)})

    assert executor.compile_actual == "compiled"
    assert executor.compile_nodes == {"rescale": "compiled"}


def test_executor_falls_back_when_compiled_wrapper_fails_at_first_execution(monkeypatch):
    def deferred_failure(_fn, **_kwargs):
        def fail(**_call_kwargs):
            raise RuntimeError("deferred compiler failure")

        return fail

    monkeypatch.setattr(torch, "compile", deferred_failure)
    executor = PipelineExecutor(_spec(), compile_nodes=True)

    out = executor.run({"rgb": torch.rand(1, 3, 8, 8)})

    assert "rescale" in out
    assert executor.compile_actual == "eager-fallback"
    assert executor.compile_nodes == {"rescale": "eager-fallback"}


def test_executor_reports_unsupported_when_compilation_was_not_requested():
    executor = PipelineExecutor(_spec())

    assert not executor.compile_requested
    assert executor.compile_actual == "unsupported"
