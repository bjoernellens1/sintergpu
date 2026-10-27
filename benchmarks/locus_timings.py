# SPDX-License-Identifier: Apache-2.0
"""Paired scalar-reference/GPU fixture timing; explicitly owns benchmark resources."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import time
from datetime import datetime,timezone


def sample_cpu(function,*,iterations,warmups):
    if type(iterations) is not int or iterations<1 or type(warmups) is not int or warmups<1:raise ValueError('positive iteration/warmup counts required')
    for _ in range(warmups):function()
    samples=[]
    for _ in range(iterations):
        start=time.perf_counter_ns();function();samples.append((time.perf_counter_ns()-start)/1e6)
    return samples


def cpu_identity():
    path=Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith('model name'):return line.split(':',1)[1].strip()
    return platform.processor() or platform.machine()


def run(*,gpu=False,iterations=50,warmups=5,device_index=0):
    from sintergpu.catalog import list_operations
    from sintergpu.fixtures.locus.cases import cases
    from sintergpu.fixtures.locus import reference
    from sintergpu.validation import _f32,_compare
    from sintergpu import __version__
    from sintergpu.timings import summarize,dependency_identities
    if gpu and os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000:raise ValueError('bounded positive iteration/warmup counts required')
    operations={op['fixture']:op for op in list_operations() if op.get('fixture')}
    environment={'cpu':cpu_identity(),'logical_cpus':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'platform':platform.platform(),'python':platform.python_version(),'gpu':None,'compiler_sha256':None,'library_version':__version__,'clock':'perf_counter_ns; GPU timestamp query when enabled'}
    environment['captured_at_utc']=datetime.now(timezone.utc).isoformat()
    for name in ('scaling_governor','scaling_cur_freq'):
        path=Path('/sys/devices/system/cpu/cpu0/cpufreq')/name
        environment[name]=path.read_text().strip() if path.is_file() else None
    device=None;records=[];dependencies=dependency_identities()
    try:
        if gpu:
            import numpy as np
            import slangpy as spy
            from sintergpu.adapters.slangpy import SlangPyAdapter
            adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
            if not 0<=device_index<len(adapters):raise ValueError('device index unavailable')
            if adapters[device_index].vendor_id==0x10005:raise ValueError('software Vulkan adapter is not a hardware GPU timing baseline')
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[device_index].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compilers=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compilers)!=1:raise RuntimeError('identify exactly one bundled compiler binary')
            environment.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=adapters[device_index].vendor_id,device_id=adapters[device_index].device_id,compiler_sha256=hashlib.sha256(compilers[0].read_bytes()).hexdigest(),slangpy=importlib.metadata.version('slangpy'),numpy=importlib.metadata.version('numpy'),floating_point_mode='precise',timestamp_frequency=int(device.info.timestamp_frequency))
            # Driver versions are retained as diagnostic data, not executed commands.
            import subprocess
            diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
            environment['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
        for case in cases():
            op=operations[case['name']]
            params=list(struct.unpack('<32f',struct.pack('<32f',*case['params'])))
            inputs=[_f32(buf) for buf in case['inputs']];initial=_f32(case['output']);state=list(case['state'])
            def execute():return reference.execute(case['name'],params,inputs,initial,state)
            expected,expected_state=execute()
            metrics={'cpu_reference_wall_ms':sample_cpu(execute,iterations=iterations,warmups=warmups)}
            workload_bytes=json.dumps({'params':params,'inputs':inputs,'output_initial':initial,'state_initial':state},sort_keys=True,separators=(',',':'),allow_nan=True).encode()
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'workload':{'id':'locus-fixture/'+case['label'],'sha256':hashlib.sha256(workload_bytes).hexdigest(),'description':'Existing small correctness fixture, not production throughput','width':case['width'],'height':case['height'],'case_source_sha256':hashlib.sha256(Path(reference.__file__).with_name('cases.py').read_bytes()).hexdigest()},'environment':dict(environment),'metrics':metrics,'iterations':iterations,'warmups':warmups,'scope':'CPU scalar reference wall time includes output construction and computation on prepared float32-valued inputs. GPU kernel timestamps exclude allocation, compilation and per-invocation output/state reset; host submit/complete is separate. Transfer-inclusive includes all five buffer uploads (including required output/state initialization), dispatch/completion and complete output/state readbacks. This fixture is not a full production pipeline or optimized-CPU comparison.'}
            if gpu:
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                arrays=[np.asarray(buf,np.float32) for buf in inputs]+[np.asarray(initial,np.float32),np.asarray(state,np.uint32)]
                buffers=[device.create_buffer(element_count=len(a),struct_size=16 if a.ndim==2 else 4,usage=usage,data=a) for a in arrays]
                bindings=dict(zip(('input0','input1','input2','output0','state'),buffers));bindings['Parameters']={'p':[spy.math.float4(*params[i:i+4]) for i in range(0,32,4)]}
                kernel=adapter.kernel(op['id']);row['adapter_source_identity']=adapter.source_identities[op['id']];pool=device.create_query_pool(spy.QueryType.timestamp,2)
                def reset():
                    for index in (3,4):buffers[index].copy_from_numpy(arrays[index])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=[case['width'],case['height'],1],vars=bindings,**kwargs)
                def read():return buffers[3].to_numpy().view(np.float32).reshape(-1),buffers[4].to_numpy().view(np.uint32).reshape(-1)
                for _ in range(warmups):reset();dispatch();device.wait()
                actual,actual_state=read();_compare(case,actual,expected,actual_state,expected_state)
                kernel_ms=[];host_ms=[];transfer_ms=[]
                for _ in range(iterations):
                    reset();pool.reset();start=time.perf_counter_ns()
                    dispatch(query_pool=pool,query_index_before=0,query_index_after=1);device.wait()
                    host_ms.append((time.perf_counter_ns()-start)/1e6)
                    stamps=pool.get_timestamp_results(0,2);kernel_ms.append((stamps[1]-stamps[0])*1000)
                actual,actual_state=read();_compare(case,actual,expected,actual_state,expected_state)
                for _ in range(iterations):
                    start=time.perf_counter_ns()
                    for b,a in zip(buffers,arrays):b.copy_from_numpy(a)
                    dispatch();device.wait();actual,actual_state=read()
                    transfer_ms.append((time.perf_counter_ns()-start)/1e6)
                _compare(case,actual,expected,actual_state,expected_state)
                metrics.update(gpu_kernel_ms=kernel_ms,host_submit_complete_ms=host_ms,transfer_inclusive_ms=transfer_ms)
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays),readback_bytes_per_iteration=arrays[3].nbytes+arrays[4].nbytes,correctness='Frozen scalar reference parity passed for all output/state values')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return {'schema_version':1,'measurements':records,'scope':'Paired small correctness-fixture timing; no optimization or full-device qualification'}
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());out=args.pop('output');record=run(**args);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps({'cases':len(record['measurements']),'output':str(out)}))
