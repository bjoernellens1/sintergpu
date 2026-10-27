import json

import pytest
import yaml

from vkl.bench import run_benchmark


def _write_pipeline(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": "metadata_test",
                "nodes": [
                    {
                        "id": "rescale",
                        "op": "canonical_rescale",
                        "inputs": {"rgb": "input.rgb", "depth": "input.depth"},
                        "params": {"target_pixels": 16},
                    }
                ],
            }
        )
    )


def test_benchmark_writes_runtime_and_execution_metadata(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline.yaml"
    benchmark = tmp_path / "benchmark.yaml"
    output = tmp_path / "results"
    _write_pipeline(pipeline)
    benchmark.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "execution": "streams",
                "source": {
                    "batch_size": 1,
                    "height": 4,
                    "width": 4,
                    "ring_size": 1,
                    "seed": 23,
                },
                "warmup": 0,
                "iterations": 1,
            }
        )
    )
    monkeypatch.setenv("VKL_SOURCE_REVISION", "test-source")
    monkeypatch.setenv("VKL_IMAGE_REVISION", "test-image")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx-test")

    result = run_benchmark(pipeline, benchmark, output_dir=output)

    metadata = json.loads((output / "runtime_metadata.json").read_text())
    assert metadata["source_revision"] == "test-source"
    assert metadata["image_revision"] == "test-image"
    assert metadata["rocm_arch"] == "gfx-test"
    assert metadata["pipeline_path"] == str(pipeline)
    assert metadata["benchmark_path"] == str(benchmark)
    assert metadata["device_requested"] == "cpu"
    assert metadata["device_actual"] == "cpu"
    assert metadata["torch_version"]
    assert result["execution"] == "serial"
    assert result["execution_requested"] == "streams"
    assert result["execution_actual"] == "serial"
    assert result["max_streams_requested"] == 4
    assert result["max_parallel_nodes"] == 1
    assert result["stream_pool_size"] == 0
    assert result["timing_mode"] == "serialized_latency"
    assert result["throughput_semantics"] == "derived_from_mean_serialized_invocation_latency"
    assert result["source_seed"] == 23
    assert result["height"] == result["depth_height"] == 4
    assert result["width"] == result["depth_width"] == 4
    assert result["serialized_frames_per_s"] == result["frames_per_s"]
    assert result["serialized_rgb_megapixels_per_s"] == result["native_megapixels_per_s"]
    assert result["serialized_depth_megapixels_per_s"] == result["native_megapixels_per_s"]
    assert result["serialized_input_megapixels_per_s"] == pytest.approx(
        2 * result["native_megapixels_per_s"]
    )


def test_compiled_benchmark_requires_untimed_warmup(tmp_path):
    pipeline = tmp_path / "pipeline.yaml"
    benchmark = tmp_path / "benchmark.yaml"
    _write_pipeline(pipeline)
    benchmark.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "compile_nodes": True,
                "source": {"batch_size": 1, "height": 4, "width": 4, "ring_size": 1},
                "warmup": 0,
                "iterations": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="warmup"):
        run_benchmark(pipeline, benchmark, output_dir=tmp_path / "results")
