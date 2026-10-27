"""Reported device capabilities and validated, timestamp-measured tuning profiles.

No vendor name determines a setting. Profiles describe measured kernel variants
on an exact runtime/device fingerprint; unavailable fields remain ``None``.
Microbenchmarks are explicit setup work, outside SLAM frame timing.
"""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROFILE_VERSION = 1
_LIMIT_FIELDS = (
    "max_compute_dispatch_thread_groups",
    "max_compute_thread_group_size",
    "max_compute_threads_per_group",
    "min_wave_size",
    "max_wave_size",
    "max_entry_point_uniform_size",
    "max_texture_dimension_2d",
)
_REDUCTION_SOURCE = """
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
#ifndef USE_WAVE
#define USE_WAVE 0
#endif
struct Params {uint count;};
ConstantBuffer<Params> cfg;
StructuredBuffer<float> samples;
RWStructuredBuffer<float> partials;
groupshared float scratch[43][GROUP_SIZE];
[shader("compute")][numthreads(GROUP_SIZE,1,1)]
void reduceNormalTerms(uint3 tid:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
 float r=tid.x<cfg.count?samples[tid.x*7]:0;float j[6];
 for(uint k=0;k<6;k++)j[k]=tid.x<cfg.count?samples[tid.x*7+1+k]:0;
 for(uint t=0;t<43;t++) {
  float value=t<36?j[t/6]*j[t%6]:(t<42?j[t-36]*r:r*r);
#if USE_WAVE
  float sum=WaveActiveSum(value);if(WaveIsFirstLane())scratch[t][lane/WaveGetLaneCount()]=sum;
#else
  scratch[t][lane]=value;
#endif
 }
 GroupMemoryBarrierWithGroupSync();
#if USE_WAVE
 uint waves=(GROUP_SIZE+WaveGetLaneCount()-1)/WaveGetLaneCount();
 for(uint t=lane;t<43;t+=GROUP_SIZE){float sum=0;for(uint w=0;w<waves;w++)sum+=scratch[t][w];partials[group.x*43+t]=sum;}
#else
 for(uint stride=GROUP_SIZE/2;stride>0;stride>>=1) {
  if(lane<stride)for(uint t=0;t<43;t++)scratch[t][lane]+=scratch[t][lane+stride];
  GroupMemoryBarrierWithGroupSync();
 }
 if(lane==0)for(uint t=0;t<43;t++)partials[group.x*43+t]=scratch[t][0];
#endif
}
"""


def _value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        return [int(value.x), int(value.y), int(value.z)]
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    return str(value)


def query_device(device):
    """Read SlangPy's public Device.info/limits/features without dispatching."""
    info = getattr(device, "info", None)
    limits = getattr(info, "limits", None)
    reported_features = getattr(device, "features", None)
    features = (
        None
        if reported_features is None
        else sorted(str(feature).rsplit(".", 1)[-1] for feature in reported_features)
    )
    try:
        version = importlib.metadata.version("slangpy")
    except importlib.metadata.PackageNotFoundError:
        version = None
    result = dict(
        adapter_name=_value(getattr(info, "adapter_name", None)),
        api_name=_value(getattr(info, "api_name", None)),
        device_type=_value(getattr(info, "type", None)),
        driver_version=None,
        timestamp_frequency_hz=_value(getattr(info, "timestamp_frequency", None)),
        limits={name: _value(getattr(limits, name, None)) for name in _LIMIT_FIELDS},
        # Current public DeviceLimits does not expose shared-memory capacity.
        max_compute_shared_memory_bytes=None,
        features=features,
        wave_ops=None if features is None else "wave_ops" in features,
        timestamp_query=None if features is None else "timestamp_query" in features,
        slangpy_version=version,
        os=platform.system(),
        kernel=platform.release(),
    )
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    return result


def _source_fingerprint():
    feature_source = Path(__file__).with_name("shaders").joinpath("features.slang").read_text()
    fused_source = Path(__file__).with_name("shaders").joinpath("dense_tracker.slang").read_text()
    profiler_source = Path(__file__).read_text()
    return hashlib.sha256(
        (_REDUCTION_SOURCE + feature_source + fused_source + profiler_source).encode()
    ).hexdigest()


def load_profile(device, path):
    """Return a compatible validated measured cache, or None when stale/missing."""
    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(profile, dict):
        return None
    if not isinstance(profile.get("device"), dict):
        return None
    selected = profile.get("selected", {})
    valid_selection = (
        isinstance(selected, dict)
        and selected.get("reduction_group_size") in (32, 64, 128, 256)
        and selected.get("reduction_mode") in ("wave", "workgroup")
        and selected.get("matcher_group_size") in (32, 64, 128)
    )
    if not valid_selection:
        return None
    if (
        profile.get("schema_version") != _PROFILE_VERSION
        or profile.get("device", {}).get("fingerprint") != query_device(device)["fingerprint"]
        or profile.get("source_fingerprint") != _source_fingerprint()
        or profile.get("validated") is not True
        or not profile.get("selected")
    ):
        return None
    return profile


def _measure(runtime, record, iterations):
    """GPU timestamp interval only; upload, compile, validation/readback excluded."""
    import slangpy as spy

    warm = runtime.device.create_command_encoder()
    for _ in range(3):
        record(warm)
    runtime.device.submit_command_buffer(warm.finish())
    runtime.device.wait()
    pool = runtime.device.create_query_pool(spy.QueryType.timestamp, iterations * 2)
    encoder = runtime.device.create_command_encoder()
    for i in range(iterations):
        encoder.write_timestamp(pool, i * 2)
        record(encoder)
        encoder.write_timestamp(pool, i * 2 + 1)
    runtime.device.submit_command_buffer(encoder.finish())
    runtime.device.wait()
    ticks = np.asarray(pool.get_results(0, iterations * 2), np.uint64)
    if np.any(ticks[1::2] <= ticks[::2]):
        raise RuntimeError("GPU timestamp ticks must be strictly increasing")
    samples = (ticks[1::2] - ticks[::2]).astype(np.float64) * (
        1000.0 / runtime.device.info.timestamp_frequency
    )
    if not np.isfinite(samples).all() or np.any(samples <= 0):
        raise RuntimeError("GPU timestamp samples must be finite and positive")
    return dict(
        median_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        samples_ms=samples.tolist(),
        iterations=iterations,
    )


def tune(runtime, cache_path=None, *, iterations=15, reduction_count=65536, matcher_count=1024):
    """Validate/time normal-reduction and actual tiled matcher kernel variants.

    Call explicitly when the device is idle. This function submits GPU work.
    Kernel failures remain recorded instead of substituting guessed settings.
    Matching verifies every known descriptor permutation. Reductions compare all
    43 group terms against an independent float64 CPU reference.
    """
    from .features import GPUFeatures
    from .tracker import Camera

    if (
        not isinstance(iterations, int)
        or iterations < 3
        or not isinstance(reduction_count, int)
        or not 64 <= reduction_count <= 2**20
        or not isinstance(matcher_count, int)
        or not 2 <= matcher_count <= 4096
    ):
        raise ValueError("invalid tuning workload sizes")
    device = query_device(runtime.device)
    if device["timestamp_query"] is not True or not device["timestamp_frequency_hz"]:
        raise RuntimeError("device does not report usable GPU timestamp queries")
    rng = np.random.default_rng(23)
    data = rng.normal(0, 0.1, (reduction_count, 7)).astype(np.float32)
    samples_buffer = runtime.buffer(data.ravel())
    reduction_results, matching_results = [], []
    limit = device["limits"]["max_compute_threads_per_group"]
    modes = ["workgroup"] + (["wave"] if device["wave_ops"] is True else [])
    for size in (32, 64, 128, 256):
        if limit is not None and size > limit:
            continue
        for mode in modes:
            entry = dict(group_size=size, mode=mode, validated=False)
            try:
                source = (
                    f"#define GROUP_SIZE {size}\n#define USE_WAVE {int(mode == 'wave')}\n"
                    + _REDUCTION_SOURCE
                )
                kernel = runtime.compile(source, entry="reduceNormalTerms")
                groups = (reduction_count + size - 1) // size
                output = runtime.buffer(np.zeros(groups * 43, np.float32))

                def record(encoder):
                    kernel.dispatch(
                        thread_count=[reduction_count, 1, 1],
                        vars=dict(
                            cfg=dict(count=reduction_count), samples=samples_buffer, partials=output
                        ),
                        command_encoder=encoder,
                    )
                    encoder.global_barrier()

                encoder = runtime.device.create_command_encoder()
                record(encoder)
                runtime.device.submit_command_buffer(encoder.finish())
                actual = output.to_numpy().view(np.float32).reshape(groups, 43)
                padded = np.zeros((groups * size, 7), np.float64)
                padded[:reduction_count] = data
                r, j = padded[:, 0], padded[:, 1:]
                terms = np.concatenate(
                    (
                        (j[:, :, None] * j[:, None, :]).reshape(-1, 36),
                        j * r[:, None],
                        (r * r)[:, None],
                    ),
                    axis=1,
                )
                expected = terms.reshape(groups, size, 43).sum(axis=1)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)
                entry.update(validated=True, timing=_measure(runtime, record, iterations))
            except Exception as error:
                entry["error"] = str(error)
            reduction_results.append(entry)
    words = rng.integers(0, 2**32, (matcher_count, 8), dtype=np.uint32)
    permutation = rng.permutation(matcher_count)
    for size in (32, 64, 128):
        entry = dict(group_size=size, validated=False)
        try:
            frontend = GPUFeatures(
                Camera(64, 64, 60.0, 60.0, 31.5, 31.5),
                runtime,
                max_features=matcher_count,
                levels=1,
                matcher_group_size=size,
            )
            a, b = frontend.create_set(), frontend.create_set()
            for feature_set, array in ((a, words), (b, words[permutation])):
                feature_set["descriptors"].copy_from_numpy(np.ascontiguousarray(array).ravel())
                feature_set["count"].copy_from_numpy(np.array([matcher_count], np.uint32))
            matches = frontend._buffer(matcher_count)

            def record(encoder):
                frontend.record_match(a, b, matches, encoder)

            encoder = runtime.device.create_command_encoder()
            record(encoder)
            runtime.device.submit_command_buffer(encoder.finish())
            actual = matches.to_numpy().view(np.uint32).ravel()
            np.testing.assert_array_equal(actual, np.argsort(permutation).astype(np.uint32))
            if int(a["match_count"].to_numpy().view(np.uint32)[0]) != matcher_count:
                raise RuntimeError("matcher compact count differs from valid permutation")
            entry.update(validated=True, timing=_measure(runtime, record, iterations))
        except Exception as error:
            entry["error"] = str(error)
        matching_results.append(entry)
    valid_reduction = [r for r in reduction_results if r["validated"] and "timing" in r]
    valid_matching = [r for r in matching_results if r["validated"] and "timing" in r]
    if not valid_reduction or not valid_matching:
        raise RuntimeError(
            "no validated measured variants for both workloads: "
            + json.dumps(dict(reduction=reduction_results, matching=matching_results))
        )
    reduction = min(valid_reduction, key=lambda r: r["timing"]["median_ms"])
    matching = min(valid_matching, key=lambda r: r["timing"]["median_ms"])
    profile = dict(
        schema_version=_PROFILE_VERSION,
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        device=device,
        source_fingerprint=_source_fingerprint(),
        validated=True,
        timing_boundary="GPU timestamps of dispatch/control barriers only; upload, compile, readback excluded",
        scope="Microkernel tuning guidance; full SLAM performance must be measured independently",
        cache_limitations=[
            "Userspace driver changes cannot be detected when driver_version is unavailable"
        ],
        workload=dict(reduction_count=reduction_count, matcher_count=matcher_count),
        reduction_variants=reduction_results,
        matching_variants=matching_results,
        selected_group_size=reduction["group_size"],
        selected_reduction="subgroup" if reduction["mode"] == "wave" else "workgroup",
        selected_matcher_group_size=matching["group_size"],
        selected=dict(
            reduction_group_size=reduction["group_size"],
            reduction_mode=reduction["mode"],
            matcher_group_size=matching["group_size"],
        ),
    )
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    return profile
