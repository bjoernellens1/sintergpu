# SPDX-License-Identifier: Apache-2.0
"""Paired standalone xsplat math snapshots; benchmark explicitly owns resources."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from benchmarks import xsplat_reference as reference
from benchmarks.locus_timings import sample_cpu,cpu_identity

TAIL=13
BUDGET=8<<20
GAUSSIAN_DTYPE=np.dtype({'names':['mean','logScale','rotation','opacityLogit','sh0'],'formats':[('f4',3),('f4',3),('f4',4),'f4',('f4',3)],'offsets':[0,16,32,48,64],'itemsize':80})
PROJECTED_DTYPE=np.dtype({'names':['meanPx','conic','depth','visible'],'formats':[('f4',2),('f4',3),'f4','u4'],'offsets':[0,16,28,32],'itemsize':48})
CAMERA_DTYPE=np.dtype({'names':['worldToCamera','focal','principal','imageSize','gaussianCount'],'formats':[('f4',(4,4)),('f4',2),('f4',2),('u4',2),'u4'],'offsets':[0,64,72,80,88],'itemsize':96})


def cases():
    for count in (31,32,1025,65539):
        for step in (0,1,37):
            rng=np.random.default_rng(73200+count)
            arrays={name:np.full((count+TAIL)*4,173,np.uint8) for name in ('values','gradients','firstMoment','secondMoment')}
            for name in arrays:
                a=arrays[name].view(np.float32)
                a[:count]=rng.uniform(.01,.2,count) if name=='secondMoment' else rng.uniform(-1,1,count)
            arrays['gradients'].view(np.float32)[:count:7]=0
            yield dict(kind='adam',operation='xsplat_vk.adam.main',count=count,params=dict(count=count,step=step,learningRate=.001,beta1=.9,beta2=.999,epsilon=1e-8),arrays=arrays,outputs=('values','firstMoment','secondMoment'),strides={k:4 for k in arrays},thread_count=[count,1,1],label=f'adam/{count}/step{step}')
    for count in (127,128,1025,65539):
        rng=np.random.default_rng(45100+count)
        g=np.zeros(count+TAIL,GAUSSIAN_DTYPE);g['mean'][:count]=rng.uniform(-1,1,(count,3));g['mean'][:count,2]=rng.uniform(2,8,count)
        g['mean'][:count:11,2]=-2;g['mean'][1:count:13,0]=30
        g['logScale'][:count]=rng.uniform(-3,-2,(count,3));g['rotation'][:count]=rng.normal(size=(count,4))
        g['opacityLogit'][:count]=rng.uniform(-2,2,count);g['sh0'][:count]=rng.uniform(-1,1,(count,3))
        camera=np.zeros(1,CAMERA_DTYPE);angle=.125;co=np.cos(angle);si=np.sin(angle)
        camera['worldToCamera'][0]=[[co,0,si,.125],[0,1,0,-.25],[-si,0,co,.5],[0,0,0,1]]
        camera['focal'][0]=[400,425];camera['principal'][0]=[320,240];camera['imageSize'][0]=[640,480];camera['gaussianCount'][0]=count
        arrays=dict(gaussians=g.view(np.uint8),projected=np.full((count+TAIL)*48,173,np.uint8),camera=camera.view(np.uint8))
        yield dict(kind='project',operation='xsplat_vk.project.projectMain',count=count,params={},arrays=arrays,outputs=('projected',),strides=dict(gaussians=80,projected=48,camera=96),thread_count=[count,1,1],label=f'project/{count}/rotated-mixed')
    yield dict(kind='probe',operation='xsplat_vk.backward_probe.backwardProbeMain',count=1,params={},arrays=dict(output=np.full((1+TAIL)*32,173,np.uint8)),outputs=('output',),strides=dict(output=32),thread_count=[1,1,1],label='backward-probe/fixed-diagnostic')


def reference_call(c):
    a=c['arrays'];n=c['count'];result={k:a[k].copy() for k in c['outputs']}
    if c['kind']=='adam':
        p=dict(c['params']);p.pop('count')
        values=reference.adam(*(a[k].view(np.float32)[:n] for k in ('values','gradients','firstMoment','secondMoment')),**p)
        for k,v in zip(c['outputs'],values):result[k].view(np.float32)[:n]=v
    elif c['kind']=='project':
        g=a['gaussians'].view(GAUSSIAN_DTYPE)[:n];cam=a['camera'].view(CAMERA_DTYPE)[0]
        v=reference.project(*(g[k] for k in ('mean','logScale','rotation')),*(cam[k] for k in ('worldToCamera','focal','principal','imageSize')))
        o=result['projected'].view(PROJECTED_DTYPE);o['meanPx'][:n]=v[:,:2];o['conic'][:n]=v[:,2:5];o['depth'][:n]=v[:,5];o['visible'][:n]=v[:,6].astype(np.uint32)
    else:result['output'].view(np.float32)[:5]=reference.backward_probe()
    return result


def workload_identity(c):
    layout=dict(operation=c['operation'],params=c['params'],count=c['count'],thread_count=c['thread_count'],strides=c['strides'],bindings=tuple(c['arrays']),outputs=c['outputs'],matrix_layout='row-major')
    h=hashlib.sha256(json.dumps(layout,sort_keys=True).encode())
    for name,a in sorted(c['arrays'].items()):h.update(name.encode()+a.dtype.str.encode()+str(a.shape).encode()+b'\0'+a.tobytes()+b'\0')
    return dict(id='xsplat-math/'+c['label'],sha256=h.hexdigest(),description='Seeded finite mathematical fixtures; not a complete renderer, optimizer training loop or optimized CPU comparison',**layout)


def check_parity(c,actual,expected):
    assert set(actual)==set(expected)==set(c['outputs'])
    for k in c['outputs']:
        a=actual[k];e=expected[k];tail=c['count']*c['strides'][k]
        assert a.dtype==e.dtype==np.uint8 and a.shape==e.shape
        np.testing.assert_array_equal(a[tail:],c['arrays'][k][tail:])
        np.testing.assert_array_equal(e[tail:],c['arrays'][k][tail:])
        if c['kind']=='project':
            av=a.view(PROJECTED_DTYPE)[:c['count']];ev=e.view(PROJECTED_DTYPE)[:c['count']]
            np.testing.assert_array_equal(av['visible'],ev['visible'])
            for field in ('meanPx','conic','depth'):np.testing.assert_allclose(av[field],ev[field],rtol=2e-6,atol=2e-6)
        else:
            n=5 if c['kind']=='probe' else c['count']
            np.testing.assert_allclose(a.view(np.float32)[:n],e.view(np.float32)[:n],rtol=2e-6,atol=2e-6)


def validate_layout(kernel,c):
    """Fail closed on actual runtime reflected storage ABI before any dispatch."""
    layout=kernel.program.layout
    if c['kind']=='project':types={'GaussianData':GAUSSIAN_DTYPE,'ProjectedGaussian':PROJECTED_DTYPE,'CameraData':CAMERA_DTYPE}
    elif c['kind']=='probe':types={'BackwardProbe':np.dtype({'names':['dRgb','dAlpha','dInputTransmittance'],'formats':[('f4',3),'f4','f4'],'offsets':[0,12,16],'itemsize':32})}
    else:types={}
    observed={}
    for name,dtype in types.items():
        t=layout.get_type_layout(layout.find_type_by_name(name))
        fields={f.name:int(f.offset) for f in t.fields}
        expected={k:int(dtype.fields[k][1]) for k in dtype.names}
        if fields!=expected or int(t.stride)!=dtype.itemsize:raise RuntimeError(f'unsupported runtime ABI for {name}: {fields}, stride{t.stride}')
        observed[name]=dict(stride=int(t.stride),offsets=fields)
    group=tuple(int(x) for x in layout.entry_points[0].compute_thread_group_size)
    expected={'adam':(32,1,1),'project':(128,1,1),'probe':(1,1,1)}[c['kind']]
    if group!=expected:raise RuntimeError('unexpected compiled thread-group profile')
    return dict(structs=observed,thread_group=group,matrix_layout='explicit compiler row-major')


def run(*,gpu=False,iterations=50,warmups=5,device_index=0):
    from sintergpu import __version__
    from sintergpu.catalog import list_operations
    from sintergpu.timings import dependency_identities,summarize
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000 or type(device_index) is not int or device_index<0:raise ValueError('bounded positive sample/warmup counts and valid device index required')
    if gpu and os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in required after independent measurement clearance')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    env=dict(cpu=cpu_identity(),logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,platform=platform.platform(),python=platform.python_version(),numpy=importlib.metadata.version('numpy'),library_version=__version__,collector_revision=revision.stdout.strip() if revision.returncode==0 else None,captured_at_utc=datetime.now(timezone.utc).isoformat(),gpu=None,compiler_sha256=None)
    for name in ('scaling_governor','scaling_cur_freq'):
        path=Path('/sys/devices/system/cpu/cpu0/cpufreq')/name;env[name]=path.read_text().strip() if path.is_file() else None
    dependencies=dependency_identities();catalog={o['id']:o for o in list_operations()};device=None;records=[]
    try:
        if gpu:
            import slangpy as spy
            from sintergpu.adapters.slangpy import SlangPyAdapter
            adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
            if not device_index<len(adapters):raise ValueError('invalid hardware device index')
            info=adapters[device_index]
            if info.vendor_id==0x10005:raise ValueError('hardware GPU required for timing')
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=info.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise,'matrix_layout':spy.SlangMatrixLayout.row_major})
            limits=device.info.limits
            if limits.max_compute_threads_per_group<128 or limits.max_compute_thread_group_size.x<128 or int(device.info.timestamp_frequency)<=0:raise RuntimeError('128-thread groups and GPU timestamps required')
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compiler=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compiler)!=1:raise RuntimeError('identify exactly one bundled compiler')
            env.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=info.vendor_id,device_id=info.device_id,slangpy=importlib.metadata.version('slangpy'),compiler_sha256=hashlib.sha256(compiler[0].read_bytes()).hexdigest(),floating_point_mode='precise',matrix_layout='row-major',timestamp_frequency=int(device.info.timestamp_frequency),buffer_budget_bytes=BUDGET)
            try:
                diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30);env['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):env['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases():
            op=catalog[case['operation']];arrays=case['arrays'];outputs=case['outputs'];execute=lambda:reference_call(case)
            expected=execute();metrics=dict(cpu_reference_wall_ms=sample_cpu(execute,iterations=iterations,warmups=warmups))
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runner_helper_sha256':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest(),'workload':workload_identity(case),'environment':dict(env),'metrics':metrics,'iterations':iterations,'warmups':warmups,'scope':'Independent float64 NumPy math oracle plus byte-output construction/copies/temporary arrays. Prepared inputs outside both regions. GPU per-entry timestamps exclude compilation, allocation and all output/inout resets. Host submit/complete and complete bound-buffer uploads/full output readbacks are separate. Fixed AD probe is diagnostic only. No full renderer/training loop, optimized CPU, controlled performance comparison, variant selection or full-domain qualification.'}
            if gpu:
                if (case['thread_count'][0]+({'adam':32,'project':128,'probe':1}[case['kind']])-1)//({'adam':32,'project':128,'probe':1}[case['kind']])>int(limits.max_compute_dispatch_thread_groups.x) or any(a.nbytes>BUDGET for a in arrays.values()):raise ValueError('fixture exceeds dispatch or declared per-buffer budget')
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={name:(device.create_buffer(size=a.nbytes,usage=spy.BufferUsage.constant_buffer,data=a) if name=='camera' else device.create_buffer(element_count=a.nbytes//case['strides'][name],struct_size=case['strides'][name],usage=usage,data=a)) for name,a in arrays.items()}
                kernel=adapter.kernel(op['id']);row['reflected_abi']=validate_layout(kernel,case)
                bindings=buffers;entry_uniforms={'p':case['params']} if case['kind']=='adam' else {};pool=device.create_query_pool(spy.QueryType.timestamp,2)
                row['adapter_source_identity']=adapter.source_identities[op['id']]
                def reset():
                    for name in outputs:buffers[name].copy_from_numpy(arrays[name])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=case['thread_count'],vars=bindings,**entry_uniforms,**kwargs)
                def read():return {name:buffers[name].to_numpy().view(arrays[name].dtype).reshape(-1) for name in outputs}
                for _ in range(warmups):reset();dispatch();device.wait()
                check_parity(case,read(),expected);kernel_ms=[];host_ms=[];transfer_ms=[]
                for _ in range(iterations):
                    reset();pool.reset();start=time.perf_counter_ns();dispatch(query_pool=pool,query_index_before=0,query_index_after=1);device.wait()
                    host_ms.append((time.perf_counter_ns()-start)/1e6);stamps=pool.get_timestamp_results(0,2);kernel_ms.append((stamps[1]-stamps[0])*1000)
                check_parity(case,read(),expected)
                def transfer():
                    for name,a in arrays.items():buffers[name].copy_from_numpy(a)
                    dispatch();device.wait();return read()
                for _ in range(warmups):check_parity(case,transfer(),expected)
                for _ in range(iterations):
                    start=time.perf_counter_ns();actual=transfer();transfer_ms.append((time.perf_counter_ns()-start)/1e6)
                check_parity(case,actual,expected)
                metrics.update(gpu_kernel_ms=kernel_ms,host_submit_complete_ms=host_ms,transfer_inclusive_ms=transfer_ms)
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays.values()),readback_bytes_per_iteration=sum(arrays[k].nbytes for k in outputs),correctness='Independent math parity before/after isolated and transfer samples. Defined live floats2e-6absolute/relative, visible flags and complete tail structs exact; within-live-struct padding is unspecified. All Adam inout state reset between samples.')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return dict(schema_version=1,environment=env,measurements=records,scope='Paired xsplat math snapshots; no renderer/training-loop/performance qualification')
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());output=args.pop('output');report=run(**args)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(report['measurements']),output=str(output))))
