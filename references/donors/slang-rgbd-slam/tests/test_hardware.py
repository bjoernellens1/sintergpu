import json
import os
from types import SimpleNamespace

import pytest

from slang_rgbd_slam.hardware import _source_fingerprint, load_profile, query_device


def fake_device(name="device"):
    return SimpleNamespace(
        info=SimpleNamespace(
            adapter_name=name,
            api_name="Vulkan",
            type="vulkan",
            timestamp_frequency=10,
            limits=SimpleNamespace(
                max_compute_threads_per_group=128, min_wave_size=32, max_wave_size=64
            ),
        ),
        features=["Feature.wave_ops", "Feature.timestamp_query"],
    )


def test_actual_capabilities_and_unknown_fields():
    result = query_device(fake_device())
    assert result["wave_ops"] is True
    assert result["limits"]["min_wave_size"] == 32
    assert result["max_compute_shared_memory_bytes"] is None
    assert result["driver_version"] is None
    assert result["limits"]["max_texture_dimension_2d"] is None
    assert query_device(SimpleNamespace())["wave_ops"] is None


def test_profile_cache_requires_device_shader_and_validation(tmp_path):
    d = fake_device()
    profile = dict(
        schema_version=1,
        device=query_device(d),
        source_fingerprint=_source_fingerprint(),
        validated=True,
        selected=dict(matcher_group_size=64, reduction_group_size=64, reduction_mode="wave"),
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert load_profile(d, path) == profile
    assert load_profile(fake_device("different"), path) is None
    profile["validated"] = False
    path.write_text(json.dumps(profile))
    assert load_profile(d, path) is None
    assert load_profile(d, tmp_path / "absent") is None


def test_gpu_timestamp_units_are_device_ticks_not_host_time():
    import pytest

    pytest.importorskip("slangpy")
    from slang_rgbd_slam.hardware import _measure

    class Encoder:
        def write_timestamp(self, pool, index):
            pass

        def finish(self):
            return None

    class Device:
        info = SimpleNamespace(timestamp_frequency=1000)

        def create_command_encoder(self):
            return Encoder()

        def submit_command_buffer(self, buffer):
            pass

        def wait(self):
            pass

        def create_query_pool(self, query_type, count):
            return SimpleNamespace(
                get_results=lambda index, count: [1000, 1010, 2000, 2020, 3000, 3030]
            )

    result = _measure(SimpleNamespace(device=Device()), lambda encoder: None, 3)
    assert result["samples_ms"] == [10.0, 20.0, 30.0]
    assert result["median_ms"] == 20.0


def test_malformed_or_unvalidated_cache_is_rejected(tmp_path):
    d = fake_device()
    path = tmp_path / "profile.json"
    for profile in (
        [1],
        {"device": "garbage"},
        dict(
            schema_version=1,
            device=query_device(d),
            source_fingerprint=_source_fingerprint(),
            validated="yes",
            selected=dict(reduction_group_size=64, reduction_mode="wave", matcher_group_size=64),
        ),
    ):
        path.write_text(json.dumps(profile))
        assert load_profile(d, path) is None


@pytest.mark.skipif(os.environ.get("SLANG_RGBD_GPU_TESTS") != "1", reason="requires Vulkan opt-in")
def test_measured_profile_validates_both_actual_kernels(tmp_path):
    from caspar_slang.runtime import VulkanRuntime

    from slang_rgbd_slam.hardware import tune

    runtime = VulkanRuntime()
    path = tmp_path / "profile.json"
    profile = tune(runtime, path, iterations=3, reduction_count=256, matcher_count=64)
    assert profile["validated"] is True
    assert profile["selected_group_size"] in (32, 64, 128, 256)
    assert profile["selected_matcher_group_size"] in (32, 64, 128)
    assert any(r["validated"] and "timing" in r for r in profile["reduction_variants"])
    assert any(r["validated"] and "timing" in r for r in profile["matching_variants"])
    assert load_profile(runtime.device, path) == profile
