# SPDX-License-Identifier: Apache-2.0
"""Paired camera-color CPU/GPU baselines; standalone benchmark owns resources."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from sintergpu.references import color as reference
from benchmarks.locus_timings import cpu_identity, sample_cpu

# Public catalog identities, independent CPU contract names and input storage.
OPERATIONS=(
    ('image.nv12_to_rgb','nv12_decode',1),
    ('image.rgb_to_nv12','nv12_encode',3),
    ('image.yuyv_to_rgb','yuyv_decode',2),
    ('image.remap_bilinear','remap',3),
    ('image.resize_bilinear','resize',3),
    ('image.yuyv_remap_fused','fused_remap',2),
)


def aligned(n):
    return (n+3)//4*4


def cases(*,small_only=False):
    """Stable pitched inputs, odd remap outputs and explicit color profiles."""
    profiles=[(6,4,7,5,m,r) for m in ('bt601','bt709') for r in ('limited','full')]
    if not small_only:profiles.append((640,480,64,48,'bt709','limited'))
    for width,height,ow,oh,matrix,range_ in profiles:
        for index,(operation,kind,channels) in enumerate(OPERATIONS):
            rng=np.random.default_rng(203+index)
            tags=reference.ColorSpec(matrix,range_)
            output_width,output_height=(ow,oh) if kind in ('remap','resize','fused_remap') else (width,height)
            sp=width*channels+3;up=width+3
            pitch=(width if kind=='nv12_encode' else output_width*3)+3
            arrays={'sourceBytes':rng.integers(0,256,aligned(sp*height),dtype=np.uint8),
                    'outputBytes':np.full(aligned(pitch*(height if kind=='nv12_encode' else output_height)),173,np.uint8)}
            outputs=['outputBytes'];kwargs={}
            if kind=='nv12_decode':
                arrays['uvBytes']=rng.integers(0,256,aligned(up*(height//2)),dtype=np.uint8)
                kwargs.update(uv_pitch=up,uv_capacity=arrays['uvBytes'].nbytes)
            if kind=='nv12_encode':
                arrays['outputUV']=np.full(aligned(up*(height//2)),181,np.uint8);outputs.append('outputUV')
                kwargs.update(output_uv_pitch=up,output_uv_capacity=arrays['outputUV'].nbytes)
            if kind in ('remap','fused_remap'):
                xx,yy=np.meshgrid(np.linspace(-.25,width-.75,output_width,dtype=np.float32),np.linspace(-.25,height-.75,output_height,dtype=np.float32))
                maps=np.ascontiguousarray(np.stack([xx,yy],axis=-1),dtype=np.float32)
                maps[0,0]=[np.nan,0];maps[0,1]=[0,np.inf]
                arrays['mapXY']=maps;kwargs.update(calibration=maps,map_capacity=maps.nbytes)
            p=reference.validate_request(kind,width=width,height=height,output_width=output_width,output_height=output_height,
                source_pitch=sp,output_pitch=pitch,source_capacity=arrays['sourceBytes'].nbytes,
                output_capacity=arrays['outputBytes'].nbytes,tags=tags,for_dispatch=True,**kwargs)
            live={}
            for name in outputs:
                rows=height//2 if name=='outputUV' else height if kind=='nv12_encode' else output_height
                stride=up if name=='outputUV' else pitch
                columns=width if kind=='nv12_encode' else output_width*3
                mask=np.zeros(arrays[name].size,bool)
                for row in range(rows):mask[row*stride:row*stride+columns]=True
                live[name]=mask
            yield {'operation':operation,'kind':kind,'tags':tags,'params':p,'arrays':arrays,'outputs':outputs,'live':live,
                   'dimensions':[width//2,height//2,1] if kind=='nv12_encode' else [output_width,output_height,1]}


def reference_call(case):
    p=case['params'];a=case['arrays'];kind=case['kind']
    common=dict(width=p['srcWidth'],height=p['srcHeight'])
    output=a['outputBytes'].copy()
    if kind=='nv12_decode':
        value=reference.nv12_to_rgb(a['sourceBytes'],a['uvBytes'],**common,y_pitch=p['sourcePitch'],uv_pitch=p['uvPitch'],rgb_pitch=p['outputPitch'],tags=case['tags'],output=output)
    elif kind=='nv12_encode':
        y,uv=reference.rgb_to_nv12(a['sourceBytes'],**common,source_pitch=p['sourcePitch'],y_pitch=p['outputPitch'],uv_pitch=p['outputUVPitch'],tags=case['tags'],y_output=output,uv_output=a['outputUV'].copy())
        return {'outputBytes':y,'outputUV':uv}
    elif kind=='yuyv_decode':
        value=reference.yuyv_to_rgb(a['sourceBytes'],**common,source_pitch=p['sourcePitch'],rgb_pitch=p['outputPitch'],tags=case['tags'],output=output)
    elif kind=='resize':
        value=reference.resize_rgb(a['sourceBytes'],**common,source_pitch=p['sourcePitch'],rgb_pitch=p['outputPitch'],output_width=p['dstWidth'],output_height=p['dstHeight'],output=output)
    elif kind=='remap':
        value=reference.remap_rgb(a['sourceBytes'],a['mapXY'],**common,source_pitch=p['sourcePitch'],rgb_pitch=p['outputPitch'],output=output)
    else:
        value=reference.fused_yuyv_remap(a['sourceBytes'],a['mapXY'],**common,source_pitch=p['sourcePitch'],rgb_pitch=p['outputPitch'],tags=case['tags'],output=output)
    return {'outputBytes':value}


def workload_identity(case):
    p=case['params'];h=hashlib.sha256(json.dumps(p,sort_keys=True).encode())
    for name,a in sorted(case['arrays'].items()):
        h.update(json.dumps([name,a.dtype.str,a.shape]).encode()+b'\0'+a.tobytes()+b'\0')
    return {'id':f"color-pitched/{case['kind']}/{p['srcWidth']}x{p['srcHeight']}-to-{p['dstWidth']}x{p['dstHeight']}/{case['tags'].matrix}-{case['tags'].range}",
            'description':'Seeded padded camera-format input; deterministic synthetic map with nonfinite coordinates, not camera calibration qualification',
            'sha256':h.hexdigest(),'params':p}


def check_parity(case,actual,expected):
    assert set(actual)==set(expected)==set(case['outputs'])
    for name in case['outputs']:
        live=case['live'][name]
        assert actual[name].dtype==np.uint8 and actual[name].shape==expected[name].shape
        assert np.max(np.abs(actual[name][live].astype(int)-expected[name][live].astype(int)))<=1
        np.testing.assert_array_equal(actual[name][~live],case['arrays'][name][~live])
        np.testing.assert_array_equal(expected[name][~live],case['arrays'][name][~live])


def run(*,gpu=False,iterations=50,warmups=5,small_only=False,device_index=0):
    from sintergpu import __version__
    from sintergpu.catalog import list_operations
    from sintergpu.timings import dependency_identities, summarize
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000:
        raise ValueError('bounded positive iteration/warmup counts required')
    if gpu and os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    environment={'cpu':cpu_identity(),'platform':platform.platform(),'python':platform.python_version(),'numpy':importlib.metadata.version('numpy'),
        'logical_cpus':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'library_version':__version__,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'gpu':None,'compiler_sha256':None}
    for name in ('scaling_governor','scaling_cur_freq'):
        p=Path('/sys/devices/system/cpu/cpu0/cpufreq')/name
        environment[name]=p.read_text().strip() if p.is_file() else None
    catalog={op['id']:op for op in list_operations()};dependencies=dependency_identities();device=None;records=[]
    try:
        if gpu:
            import slangpy as spy
            from sintergpu.adapters.slangpy import SlangPyAdapter
            adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
            if type(device_index) is not int or not 0<=device_index<len(adapters):raise ValueError('device index unavailable')
            info=adapters[device_index]
            if info.vendor_id==0x10005:raise ValueError('software Vulkan is not a hardware timing baseline')
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=info.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compilers=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compilers)!=1:raise RuntimeError('identify exactly one bundled compiler binary')
            if int(device.info.timestamp_frequency)<=0:raise RuntimeError('GPU timestamps unavailable')
            environment.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=info.vendor_id,device_id=info.device_id,
                compiler_sha256=hashlib.sha256(compilers[0].read_bytes()).hexdigest(),slangpy=importlib.metadata.version('slangpy'),floating_point_mode='precise',timestamp_frequency=int(device.info.timestamp_frequency))
            try:
                diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
                environment['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):environment['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases(small_only=small_only):
            op=catalog[case['operation']];arrays=case['arrays'];outputs=case['outputs']
            execute=lambda:reference_call(case)
            expected=execute();metrics={'cpu_reference_wall_ms':sample_cpu(execute,iterations=iterations,warmups=warmups)}
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],
                'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'runner_dependencies_sha256':{'benchmarks/locus_timings.py':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest()},
                'workload':workload_identity(case),'environment':dict(environment),'metrics':metrics,'iterations':iterations,'warmups':warmups,
                'scope':'Independent NumPy CPU contract wall time includes validation, output seed copies and computation on prepared inputs. GPU timestamps bracket one dispatch only, excluding compilation/allocation and required output reset. Host submit/complete is separate. Transfer-inclusive uploads every input/map and output seed, dispatches/completes and reads every output including padding. Synthetic workloads on a shared machine are reference baselines, not optimized-CPU comparisons, controlled speedup, camera or fusion qualification.'}
            if gpu:
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={name:device.create_buffer(element_count=a.nbytes//(8 if name=='mapXY' else 4),struct_size=8 if name=='mapXY' else 4,usage=usage,data=a) for name,a in arrays.items()}
                bindings={**buffers,'Params':case['params']};kernel=adapter.kernel(op['id']);query=device.create_query_pool(spy.QueryType.timestamp,2)
                row['adapter_source_identity']=adapter.source_identities[op['id']]
                def reset():
                    for name in outputs:buffers[name].copy_from_numpy(arrays[name])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=case['dimensions'],vars=bindings,**kwargs)
                def read():return {name:buffers[name].to_numpy().view(np.uint8).reshape(-1) for name in outputs}
                for _ in range(warmups):reset();dispatch();device.wait()
                check_parity(case,read(),expected)
                kernel_ms=[];host_ms=[]
                for _ in range(iterations):
                    reset();query.reset();start=time.perf_counter_ns()
                    dispatch(query_pool=query,query_index_before=0,query_index_after=1);device.wait()
                    host_ms.append((time.perf_counter_ns()-start)/1e6)
                    stamps=query.get_timestamp_results(0,2);kernel_ms.append((stamps[1]-stamps[0])*1000)
                check_parity(case,read(),expected)
                def transfer():
                    for name,a in arrays.items():buffers[name].copy_from_numpy(a)
                    dispatch();device.wait();return read()
                for _ in range(warmups):transfer()
                transfer_ms=[]
                for _ in range(iterations):
                    start=time.perf_counter_ns();actual=transfer();transfer_ms.append((time.perf_counter_ns()-start)/1e6)
                check_parity(case,actual,expected)
                metrics.update(gpu_kernel_ms=kernel_ms,host_submit_complete_ms=host_ms,transfer_inclusive_ms=transfer_ms)
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays.values()),readback_bytes_per_iteration=sum(arrays[name].nbytes for name in outputs),
                    correctness='All output bytes within independent float64 oracle 1-LSB contract; every padding byte exact before/after kernel and transfer samples')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return {'schema_version':1,'environment':environment,'measurements':records,'scope':'Paired camera-color reference workloads; no optimization or full-device qualification'}
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5)
    parser.add_argument('--small-only',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());out=args.pop('output');report=run(**args)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(report['measurements']),'output':str(out)}))
