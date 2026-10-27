from __future__ import annotations

import pytest
import torch

from vkl.pipeline import NodeSpec, PipelineExecutor, PipelineSpec


def _spec() -> PipelineSpec:
    return PipelineSpec(
        "execution-contract",
        (NodeSpec("rescale", "canonical_rescale", {"rgb": "input.rgb"}),),
    )


def _parallel_spec() -> PipelineSpec:
    return PipelineSpec(
        "parallel-execution-contract",
        (
            NodeSpec("left", "canonical_rescale", {"rgb": "input.rgb"}),
            NodeSpec("right", "canonical_rescale", {"rgb": "input.rgb"}),
        ),
    )


def _cross_stream_dependency_spec() -> PipelineSpec:
    return PipelineSpec(
        "cross-stream-execution-contract",
        (
            NodeSpec("left", "canonical_rescale", {"rgb": "input.rgb"}),
            NodeSpec("right", "canonical_rescale", {"rgb": "input.rgb"}),
            NodeSpec("join", "canonical_rescale", {"rgb": "right.rgb"}),
        ),
        ("right.rgb",),
    )


def test_stream_request_with_cpu_inputs_executes_serial_without_cuda_streams(monkeypatch):
    def fail_stream(*_args, **_kwargs):
        raise AssertionError("CPU input pipeline must not create CUDA streams")

    monkeypatch.setattr(torch.cuda, "Stream", fail_stream)
    executor = PipelineExecutor(_spec())
    out = executor.run(
        {"rgb": torch.rand(1, 3, 8, 8)},
        mode="streams",
        max_streams=3,
        join_current_stream=False,
    )

    assert "rescale" in out
    assert executor.last_execution_mode == "serial"
    assert executor.last_stream_count == 0
    assert executor.last_outputs_joined
    assert executor.max_parallel_nodes == 1
    assert executor._stream_cache == {}


def test_nonjoining_cpu_transaction_returns_none_completion_event():
    executor = PipelineExecutor(_spec())

    out = executor.run(
        {"rgb": torch.rand(1, 3, 8, 8)},
        mode="streams",
        join_current_stream=False,
    )

    assert out["completion_event"] is None


def test_executor_rejects_malformed_prepared_ready_event():
    executor = PipelineExecutor(_spec())
    with pytest.raises(ValueError, match="ready_event"):
        executor.run({"rgb": torch.rand(1, 3, 8, 8), "ready_event": object()})


def test_single_chain_never_allocates_worker_streams(monkeypatch):
    def fail_stream(*_args, **_kwargs):
        raise AssertionError("parallel width 1 must not allocate worker streams")

    monkeypatch.setattr(torch.cuda, "Stream", fail_stream)
    executor = PipelineExecutor(_spec())

    executor.run({"rgb": torch.rand(1, 3, 8, 8)}, mode="streams", max_streams=8)

    assert executor.max_parallel_nodes == 1
    assert executor.last_execution_mode == "serial"
    assert executor.last_stream_count == 0


def test_camera_conditioning_accepts_yaml_list_parameters_through_executor():
    spec = PipelineSpec(
        "camera-conditioning",
        (
            NodeSpec(
                "rectify",
                "camera_rectify",
                {
                    "image": "input.rgb",
                    "source_intrinsics": "input.intrinsics",
                    "target_intrinsics": "input.intrinsics",
                    "output_hw": "input.output_hw",
                },
                {"validate": False},
            ),
        ),
        ("rectify.image", "rectify.valid"),
    )
    executor = PipelineExecutor(spec)
    rgb = torch.rand(1, 3, 4, 5)
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]])

    out = executor.run({"rgb": rgb, "intrinsics": intrinsics, "output_hw": [4, 5]})

    assert torch.allclose(out["rectify.image"], rgb)
    assert out["rectify.valid"].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_executor_waits_explicit_ready_event_after_caller_switches_stream():
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    consume_stream = torch.cuda.Stream(device=device)
    rgb = torch.zeros(1, 3, 8, 8, device=device)
    with torch.cuda.stream(copy_stream):
        rgb.fill_(0.75)
        ready = torch.cuda.Event()
        ready.record(copy_stream)

    executor = PipelineExecutor(_spec())
    with torch.cuda.stream(consume_stream):
        out = executor.run({"rgb": rgb, "ready_event": ready})
    consume_stream.synchronize()

    assert torch.allclose(out["rescale"]["rgb"], torch.full_like(rgb, 0.75))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_stream_execution_reuses_only_real_parallel_width_on_actual_input_device():
    device = torch.device("cuda")
    executor = PipelineExecutor(_parallel_spec())
    inputs = {"rgb": torch.rand(1, 3, 8, 8, device=device)}

    executor.run(inputs, mode="streams", max_streams=8)
    first = tuple(executor._stream_cache.values())[0]
    executor.run(inputs, mode="streams", max_streams=8)
    second = tuple(executor._stream_cache.values())[0]

    assert executor.max_parallel_nodes == 2
    assert executor.last_execution_mode == "streams"
    assert executor.last_stream_count == 2
    assert executor.last_outputs_joined
    assert first == second
    assert len(first) == 2
    assert all(stream.device == inputs["rgb"].device for stream in first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_stream_execution_can_enqueue_without_serializing_next_frame_on_current_stream():
    device = torch.device("cuda")
    executor = PipelineExecutor(_parallel_spec())
    inputs = {"rgb": torch.rand(1, 3, 8, 8, device=device)}

    executor.run(
        inputs,
        mode="streams",
        max_streams=2,
        join_current_stream=False,
    )

    assert executor.last_execution_mode == "streams"
    assert executor.last_stream_count == 2
    assert not executor.last_outputs_joined

    executor.join_current_stream(device)
    assert executor.last_outputs_joined


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_wait_transaction_waits_only_for_its_completion_event(monkeypatch):
    device = torch.device("cuda")
    executor = PipelineExecutor(_parallel_spec())
    inputs = {"rgb": torch.rand(1, 3, 8, 8, device=device)}
    first = executor.run(inputs, mode="streams", max_streams=2, join_current_stream=False)
    second = executor.run(inputs, mode="streams", max_streams=2, join_current_stream=False)
    waited: list[torch.cuda.Event] = []
    original = torch.cuda.Stream.wait_event

    def wait_event(stream, event):
        waited.append(event)
        return original(stream, event)

    monkeypatch.setattr(torch.cuda.Stream, "wait_event", wait_event)
    executor.wait_transaction(first, stream=torch.cuda.Stream(device=device))

    assert waited == [first["completion_event"]]
    assert second["completion_event"] not in waited


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_nonjoining_stream_execution_records_caller_input_on_each_worker(monkeypatch):
    device = torch.device("cuda")
    executor = PipelineExecutor(_parallel_spec())
    rgb = torch.rand(1, 3, 8, 8, device=device)
    recorded: list[int] = []
    original = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        if tensor is rgb:
            recorded.append(stream.cuda_stream)
        return original(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    executor.run({"rgb": rgb}, mode="streams", max_streams=2, join_current_stream=False)

    workers = next(iter(executor._stream_cache.values()))
    assert set(recorded) == {stream.cuda_stream for stream in workers}
    executor.join_current_stream(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_nonjoining_stream_execution_records_cross_stream_dependency(monkeypatch):
    device = torch.device("cuda")
    executor = PipelineExecutor(_cross_stream_dependency_spec())
    recorded: list[tuple[int, int]] = []
    original = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        recorded.append((tensor.data_ptr(), stream.cuda_stream))
        return original(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    out = executor.run(
        {"rgb": torch.rand(1, 3, 8, 8, device=device)},
        mode="streams",
        max_streams=2,
        join_current_stream=False,
    )

    workers = next(iter(executor._stream_cache.values()))
    assert (out["right.rgb"].data_ptr(), workers[0].cuda_stream) in recorded
    executor.join_current_stream(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_stream_execution_ignores_unreferenced_cpu_metadata():
    executor = PipelineExecutor(_parallel_spec())
    inputs = {
        "rgb": torch.rand(1, 3, 8, 8, device="cuda"),
        "unused_cpu_metadata": torch.ones(1),
    }

    executor.run(inputs, mode="streams")

    assert executor.last_execution_mode == "streams"
    assert executor.last_stream_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_stream_execution_rejects_mixed_devices_when_both_are_referenced():
    spec = PipelineSpec(
        "mixed-device",
        (
            NodeSpec(
                "left",
                "canonical_rescale",
                {"rgb": "input.rgb", "depth": "input.depth"},
            ),
            NodeSpec("right", "canonical_rescale", {"rgb": "input.rgb"}),
        ),
    )
    executor = PipelineExecutor(spec)
    inputs = {
        "rgb": torch.rand(1, 3, 8, 8, device="cuda"),
        "depth": torch.ones(1, 1, 8, 8),
    }

    with pytest.raises(ValueError, match="one CUDA/HIP device"):
        executor.run(inputs, mode="streams")
