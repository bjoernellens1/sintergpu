"""Warmed resident Vulkan timing; uploads and numerical readback are excluded."""

import platform
from importlib.metadata import version
from time import perf_counter


def run_benchmark(size=4099, iterations=20, reduction="workgroup"):
    """Time repeated steps on one prepared problem, with one measured submission.

    Timestamp differences cover residual/Jacobian, hierarchical normal-equation
    reductions, barriers, and the 6x6 solve. Host wall time additionally covers
    command recording, submission and waiting, and is never reported as GPU time.
    """
    for name, value in [("size", size), ("iterations", iterations)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    import numpy as np
    import slangpy as spy

    from .gn import GaussNewton, synthetic_problem

    solver = GaussNewton(reduction=reduction)
    problem = solver.prepare(*synthetic_problem(size))
    device = solver.runtime.device
    warmups = 5
    for _ in range(warmups):
        solver.dispatch(problem)
    device.wait_for_idle()

    # QueryPool.get_results returns integer device ticks, not seconds.
    frequency = int(device.info.timestamp_frequency)
    pool = None
    timestamp_unavailable = None
    try:
        if frequency <= 0:
            raise RuntimeError("device reports no timestamp frequency")
        pool = device.create_query_pool(spy.QueryType.timestamp, 2 * iterations)
    except (RuntimeError, ValueError) as error:
        timestamp_unavailable = str(error)

    start = perf_counter()
    encoder = device.create_command_encoder()
    for i in range(iterations):
        if pool is not None:
            encoder.write_timestamp(pool, 2 * i)
        solver.dispatch(problem, command_encoder=encoder)
        if pool is not None:
            encoder.write_timestamp(pool, 2 * i + 1)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    wall_seconds = perf_counter() - start

    gpu_ms = None
    if pool is not None:
        ticks = pool.get_results(0, 2 * iterations)
        differences = np.asarray([ticks[2 * i + 1] - ticks[2 * i] for i in range(iterations)])
        if np.any(differences <= 0):
            raise RuntimeError("GPU timestamp samples must be strictly increasing")
        samples = differences * (1000.0 / frequency)
        gpu_ms = {
            "min": float(np.min(samples)),
            "median": float(np.median(samples)),
            "mean": float(np.mean(samples)),
            "p95": float(np.percentile(samples, 95)),
            "samples": samples.tolist(),
        }
    return {
        "size": size,
        "iterations": iterations,
        "warmups": warmups,
        "reduction": reduction,
        "measured_submissions": 1,
        "device": device.info.adapter_name,
        "api": device.info.api_name,
        "versions": {name: version(name) for name in ("slangpy", "symforce", "numpy")},
        "python": platform.python_version(),
        "os": platform.platform(),
        "timestamp_frequency_hz": frequency,
        "gpu_ms": gpu_ms,
        "timestamp_unavailable": timestamp_unavailable,
        "host_wall_ms_per_step": wall_seconds * 1000.0 / iterations,
        "gpu_boundary": "residual/Jacobian + hierarchical reductions + damped 6x6 solve",
        "host_boundary": "record + single submit + wait; amortized over repeated steps",
        "excluded": [
            "symbolic generation",
            "compilation",
            "buffer allocation",
            "input upload",
            "warmups",
            "query retrieval",
            "numerical readback",
        ],
    }
