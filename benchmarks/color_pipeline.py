#!/usr/bin/env python3
"""Opt-in resident and transfer-inclusive YUYV decode/remap comparison.

Independent performance review must clear source/guards before GPU execution.
Example: SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/color_pipeline.py
 --width 1280 --height 720 --runs 3 --samples 100 --warmups 20 --output result.json
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from sintergpu.references.color import ColorSpec, validate_request

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'src/sintergpu/kernels/image/camera_color.slang'
ENTRIES=('nv12Decode','nv12Encode','yuyvDecode','remapBilinear','resizeBilinear','fusedYUYVRemap')


def aligned(n): return (n+3)//4*4


def summarize(samples):
    return {'p50_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
            'p99_ms':float(np.percentile(samples,99)),'samples_ms':list(map(float,samples))}


def fusion_gate(separate,fused,*,seed=27,bootstrap=2000):
    """Bootstrap the p50 fractional gain; preserve p95 regression separately."""
    separate=np.asarray(separate,dtype=np.float64); fused=np.asarray(fused,dtype=np.float64)
    if len(separate)!=len(fused) or len(separate)<100 or np.any(separate<=0) or np.any(fused<=0):
        raise ValueError('at least 100 positive paired samples required')
    rng=np.random.default_rng(seed); gains=[]
    for _ in range(bootstrap):
        indices=rng.integers(0,len(separate),len(separate))
        gains.append(1-np.median(fused[indices])/np.median(separate[indices]))
    gain=1-float(np.median(fused)/np.median(separate))
    lower=float(np.percentile(gains,2.5)); upper=float(np.percentile(gains,97.5))
    p95=float(np.percentile(fused,95)/np.percentile(separate,95)-1)
    return {'p50_gain_fraction':gain,'bootstrap_gain_95ci':[lower,upper],'p95_regression_fraction':p95,
            'passed':bool(gain>=.05 and lower>0 and p95<=.05),'bootstrap_samples':bootstrap}


class GPURunner:
    """Benchmark/test-owned runtime. Production kernels borrow every resource."""
    def __init__(self):
        if os.environ.get('SINTERGPU_GPU_TESTS')!='1':
            raise RuntimeError('GPU execution requires SINTERGPU_GPU_TESTS=1 and independent performance clearance')
        import slangpy as spy
        self.spy=spy
        self.device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        module=self.device.load_module(str(SHADER))
        self.kernels={entry:self.device.create_compute_kernel(self.device.link_program([module],[module.entry_point(entry)])) for entry in ENTRIES}
        self.usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access

    def buffer(self,array,*,map_buffer=False):
        array=np.ascontiguousarray(array)
        if array.nbytes%4: raise ValueError('word-addressed allocations must be 4-byte padded')
        return self.device.create_buffer(element_count=array.nbytes//(8 if map_buffer else 4),struct_size=8 if map_buffer else 4,usage=self.usage,data=array)

    def execute(self,entry,source,params,*,uv=None,maps=None,output_seed=None,uv_seed=None):
        p=params
        rows=p['srcHeight'] if entry=='nv12Encode' else p['dstHeight']
        out=np.full(aligned(p['outputPitch']*rows),173,np.uint8) if output_seed is None else output_seed.copy()
        bindings={'sourceBytes':self.buffer(source),'outputBytes':self.buffer(out),'Params':p}
        if entry=='nv12Decode': bindings['uvBytes']=self.buffer(uv)
        if entry in ('remapBilinear','fusedYUYVRemap'): bindings['mapXY']=self.buffer(maps,map_buffer=True)
        if entry=='nv12Encode':
            uvo=np.full(aligned(p['outputUVPitch']*(p['srcHeight']//2)),181,np.uint8) if uv_seed is None else uv_seed.copy()
            bindings['outputUV']=self.buffer(uvo)
        dims=[p['srcWidth']//2,p['srcHeight']//2,1] if entry=='nv12Encode' else [p['dstWidth'],p['dstHeight'],1]
        self.kernels[entry].dispatch(thread_count=dims,vars=bindings)
        self.device.wait_for_idle()
        actual=bindings['outputBytes'].to_numpy().view(np.uint8).reshape(-1)
        if entry=='nv12Encode': return actual,bindings['outputUV'].to_numpy().view(np.uint8).reshape(-1)
        return actual

    def environment(self):
        compiler_paths=list(Path(self.spy.__file__).parent.glob('libslang-compiler*'))
        driver={'status':'unavailable'}
        tool=shutil.which('vulkaninfo')
        if tool:
            result=subprocess.run([tool,'--summary'],capture_output=True,text=True)
            if result.returncode==0: driver={'status':'recorded','vulkaninfo_summary':result.stdout}
        return {'device':self.device.info.adapter_name,'api':self.device.info.api_name,'driver':driver,
                'slangpy':importlib.metadata.version('slangpy'),'compiler_build_tag':str(self.spy.SLANG_BUILD_TAG),
                'compiler_flags':{'floating_point_mode':'precise'},
                'compiler_libraries_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in compiler_paths},
                'shader_sha256':hashlib.sha256(SHADER.read_bytes()).hexdigest(),
                'oracle_sha256':hashlib.sha256((ROOT/'src/sintergpu/references/color.py').read_bytes()).hexdigest()}


def run(width=1280,height=720,runs=3,samples=100,warmups=20):
    if runs<3 or samples<100 or warmups<20: raise ValueError('fusion comparison needs >=3 runs, >=100 samples and >=20 warmups')
    runner=GPURunner(); device=runner.device; spy=runner.spy
    source_pitch=width*2+8; rgb_pitch=width*3+4
    source=np.random.default_rng(27).integers(0,256,aligned(source_pitch*height),dtype=np.uint8)
    xx,yy=np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
    # Synthetic calibration fixture is shared exactly by both paths, not a camera model claim.
    maps=np.ascontiguousarray(np.stack([xx+.23*np.sin(yy/37),yy+.19*np.cos(xx/43)],axis=-1),dtype=np.float32)
    seed=np.full(aligned(rgb_pitch*height),173,np.uint8)
    intermediate_seed=np.full_like(seed,167)
    source_buffer=runner.buffer(source); map_buffer=runner.buffer(maps,map_buffer=True)
    intermediate=runner.buffer(intermediate_seed)
    outputs={kind:runner.buffer(seed) for kind in ('separate','fused')}
    decode=validate_request('yuyv_decode',width=width,height=height,source_pitch=source_pitch,output_pitch=rgb_pitch,
                            source_capacity=source.nbytes,output_capacity=seed.nbytes)
    remap=validate_request('remap',width=width,height=height,source_pitch=rgb_pitch,output_pitch=rgb_pitch,
                           source_capacity=intermediate_seed.nbytes,output_capacity=seed.nbytes,calibration=maps,map_capacity=maps.nbytes)
    fused=validate_request('fused_remap',width=width,height=height,source_pitch=source_pitch,output_pitch=rgb_pitch,
                           source_capacity=source.nbytes,output_capacity=seed.nbytes,calibration=maps,map_capacity=maps.nbytes)
    dimensions=[width,height,1]
    decode_bindings={'Params':decode,'sourceBytes':source_buffer,'outputBytes':intermediate}
    remap_bindings={'Params':remap,'sourceBytes':intermediate,'mapXY':map_buffer,'outputBytes':outputs['separate']}
    fused_bindings={'Params':fused,'sourceBytes':source_buffer,'mapXY':map_buffer,'outputBytes':outputs['fused']}
    def record(encoder,kind):
        if kind=='separate':
            runner.kernels['yuyvDecode'].dispatch(thread_count=dimensions,vars=decode_bindings,command_encoder=encoder)
            encoder.global_barrier()
            runner.kernels['remapBilinear'].dispatch(thread_count=dimensions,vars=remap_bindings,command_encoder=encoder)
        else:
            runner.kernels['fusedYUYVRemap'].dispatch(thread_count=dimensions,vars=fused_bindings,command_encoder=encoder)
    def complete(encoder):
        device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    for _ in range(warmups):
        for kind in ('separate','fused'):
            encoder=device.create_command_encoder(); record(encoder,kind); complete(encoder)
    frequency=int(device.info.timestamp_frequency)
    if frequency<=0: raise RuntimeError('GPU timestamps unavailable')
    query=device.create_query_pool(spy.QueryType.timestamp,2)
    records=[]
    for run_index in range(runs):
        gpu={kind:[] for kind in outputs}; host={kind:[] for kind in outputs}; transfer={kind:[] for kind in outputs}
        # Alternate submission order to reduce simple order/thermal bias.
        for sample in range(samples):
            order=('separate','fused') if (sample+run_index)%2==0 else ('fused','separate')
            for kind in order:
                query.reset()
                start=time.perf_counter_ns(); encoder=device.create_command_encoder(); encoder.write_timestamp(query,0)
                record(encoder,kind); encoder.write_timestamp(query,1); complete(encoder)
                host[kind].append((time.perf_counter_ns()-start)/1e6)
                ticks=query.get_results(0,2); elapsed=(ticks[1]-ticks[0])*1000/frequency
                if elapsed<=0: raise RuntimeError('invalid GPU timestamp interval')
                gpu[kind].append(elapsed)
            for kind in order:
                start=time.perf_counter_ns()
                source_buffer.copy_from_numpy(source); map_buffer.copy_from_numpy(maps)
                encoder=device.create_command_encoder(); record(encoder,kind); complete(encoder)
                outputs[kind].to_numpy()
                transfer[kind].append((time.perf_counter_ns()-start)/1e6)
        actual={kind:output.to_numpy().view(np.uint8).reshape(-1) for kind,output in outputs.items()}
        np.testing.assert_array_equal(actual['separate'],actual['fused'])
        live=np.zeros(seed.size,bool)
        for row in range(height): live[row*rgb_pitch:row*rgb_pitch+width*3]=True
        for kind in outputs: np.testing.assert_array_equal(actual[kind][~live],seed[~live])
        records.append({'run':run_index+1,'parity':'byte-exact including preserved padding',
                        'gpu_pipeline':{kind:summarize(values) for kind,values in gpu.items()},
                        'host_record_submit_complete':{kind:summarize(values) for kind,values in host.items()},
                        'transfer_inclusive_upload_pipeline_readback':{kind:summarize(values) for kind,values in transfer.items()},
                        'resident_gate':fusion_gate(gpu['separate'],gpu['fused']),
                        'transfer_inclusive_gate':fusion_gate(transfer['separate'],transfer['fused'])})
    environment=runner.environment()
    qualified=environment['driver']['status']=='recorded' and bool(environment['compiler_libraries_sha256'])
    passed=qualified and all(r['resident_gate']['passed'] and r['transfer_inclusive_gate']['passed'] for r in records)
    return {'status':'exploratory-fusion-gate-passed' if passed else 'separate-fallback','default_pipeline':'separate',
            'performance_qualification':'exploratory; harness does not establish an exclusive GPU window',
            'promotion':'manual independent qualification and consumer adoption required; numerical gate cannot auto-select fusion',
            'scope':'single-device synthetic calibrated-map fixture; no NVIDIA or real camera qualification',
            'environment':environment,'width':width,'height':height,'source_pitch_bytes':source_pitch,'rgb_pitch_bytes':rgb_pitch,
            'runs':runs,'samples_per_run':samples,'warmups':warmups,'records':records,
            'transfer_bytes_per_frame':{'uploaded':source.nbytes+maps.nbytes,'readback':seed.nbytes},
            'resident_excludes':['compilation','allocation','source/map upload','output readback'],
            'gpu_region':'whole pipeline; separate includes explicit decode-to-remap global barrier',
            'transfer_inclusive_region':'reuse allocations, upload source+map, record/submit/wait, download packed output',
            'nvidia_qualification':'pending'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=1280); p.add_argument('--height',type=int,default=720)
    p.add_argument('--runs',type=int,default=3); p.add_argument('--samples',type=int,default=100); p.add_argument('--warmups',type=int,default=20)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=run(a.width,a.height,a.runs,a.samples,a.warmups)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','default_pipeline','width','height','scope')},indent=2))
