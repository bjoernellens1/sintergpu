"""Opt-in VKL projection resident GPU microbenchmark, no speedup claim.

SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/vkl_projection.py --points 307201 --iterations 30 --output /tmp/vkl-projection.json
Compilation, allocation, upload and readback are excluded from GPU timings.
Host timing includes recording, submission and wait, amortized over one submission.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def run(points=307201, iterations=30):
    if os.getenv('SINTERGPU_GPU_TESTS') != '1':
        raise RuntimeError('set SINTERGPU_GPU_TESTS=1 to request GPU execution')
    if points < 1 or iterations < 1:
        raise ValueError('points and iterations must be positive')
    import numpy as np
    import slangpy as spy
    root = Path(__file__).resolve().parents[1]
    source = root/'src/sintergpu/kernels/vkl/project_depth_evidence.slang'
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    module = device.load_module(str(source))
    program = device.link_program([module], [module.entry_point('projectDepthEvidence')])
    kernel = device.create_compute_kernel(program)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    rng = np.random.default_rng(27)
    xyz = rng.uniform(-1,1,(points,3)).astype(np.float32); xyz[:,2] += 2
    arrays = dict(xyz=xyz,visible=np.ones(points,np.uint32),provisional=np.ones(points,np.uint32),depthCandidate=np.ones(points,np.uint32),worldToCamera=np.eye(4,dtype=np.float32),intrinsics=np.array([[100,0,159.5],[0,100,119.5],[0,0,1]],np.float32),sensorDepth=np.full((240,320),2,np.float32))
    variables = {name:device.create_buffer(element_count=data.size,struct_size=4,usage=usage,data=data.reshape(-1)) for name,data in arrays.items()}
    for name in ('eligibleOut','projectedOut','sensorOut','supportOut','conflictOut','residualOut'):
        variables[name] = device.create_buffer(element_count=points,struct_size=4,usage=usage)
    variables['params'] = dict(batches=1,points=points,height=240,width=320,supportThreshold=.1,conflictThreshold=.1,epsilon=float(np.finfo(np.float32).eps))
    for _ in range(5):
        kernel.dispatch(thread_count=[points,1,1],vars=variables)
    device.wait_for_idle()
    frequency = int(device.info.timestamp_frequency)
    if frequency <= 0:
        raise RuntimeError('GPU timestamp frequency unavailable')
    pool = device.create_query_pool(spy.QueryType.timestamp,2*iterations)
    start = time.perf_counter_ns()
    encoder = device.create_command_encoder()
    for index in range(iterations):
        encoder.write_timestamp(pool,2*index)
        kernel.dispatch(thread_count=[points,1,1],vars=variables,command_encoder=encoder)
        encoder.write_timestamp(pool,2*index+1)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    wall_ms = (time.perf_counter_ns()-start)/1e6
    ticks = pool.get_results(0,2*iterations)
    milliseconds = np.array([ticks[2*i+1]-ticks[2*i] for i in range(iterations)])*1000/frequency
    if (milliseconds <= 0).any():
        raise RuntimeError('non-increasing GPU timestamps')
    # Numerical regression tests run separately; this benchmark makes no parity claim.
    slang_libraries = sorted(Path(spy.__file__).parent.glob('libslang-compiler*'))
    compiler = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in slang_libraries if p.is_file()}
    return dict(operation='vkl.project_depth_evidence',status='measurement_only',device=device.info.adapter_name,
        slangpy=importlib.metadata.version('slangpy'),compiler_libraries_sha256=compiler,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),points=points,iterations=iterations,warmups=5,
        gpu_resident_ms=dict(median=float(np.median(milliseconds)),p95=float(np.percentile(milliseconds,95)),samples=milliseconds.tolist()),
        host_record_submit_wait_ms_per_dispatch=wall_ms/iterations,
        measured_submissions=1,excluded=['compile','allocation','upload','readback','transfer-inclusive latency'],
        limits='Single local AMD GPU; no baseline comparison, confidence interval, transfer-inclusive benefit or vendor qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=307201)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = json.dumps(run(args.points,args.iterations),indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result,end='')
