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
import tempfile
import numpy as np
from benchmarks import xsplat_reference as reference
from benchmarks import xsplat_projection_bounds as pixel_bounds
from benchmarks import xsplat_spirv as program_precision
from benchmarks.locus_timings import sample_cpu,cpu_identity

ABI_PROFILE="slangpy-0.43.1-vulkan-scalar"
TAIL=13
BUDGET=8<<20
GAUSSIAN_DTYPE=np.dtype({'names':['mean','logScale','rotation','opacityLogit','sh0'],'formats':[('f4',3),('f4',3),('f4',4),'f4',('f4',3)],'offsets':[0,12,24,40,44],'itemsize':56})
PROJECTED_DTYPE=np.dtype({'names':['meanPx','conic','depth','visible'],'formats':[('f4',2),('f4',3),'f4','u4'],'offsets':[0,8,20,24],'itemsize':28})
CAMERA_DTYPE=np.dtype({'names':['worldToCamera','focal','principal','imageSize','gaussianCount'],'formats':[('f4',(4,4)),('f4',2),('f4',2),('u4',2),'u4'],'offsets':[0,64,72,80,88],'itemsize':92})


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
        arrays=dict(gaussians=g.view(np.uint8),projected=np.full((count+TAIL)*PROJECTED_DTYPE.itemsize,173,np.uint8),camera=camera.view(np.uint8))
        yield dict(kind='project',operation='xsplat_vk.project.projectMain',count=count,params={},arrays=arrays,outputs=('projected',),strides=dict(gaussians=GAUSSIAN_DTYPE.itemsize,projected=PROJECTED_DTYPE.itemsize,camera=CAMERA_DTYPE.itemsize),thread_count=[count,1,1],label=f'project/{count}/rotated-mixed')
    yield dict(kind='probe',operation='xsplat_vk.backward_probe.backwardProbeMain',count=1,params={},arrays=dict(output=np.full((1+TAIL)*20,173,np.uint8)),outputs=('output',),strides=dict(output=20),thread_count=[1,1,1],label='backward-probe/fixed-diagnostic')


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
    layout=dict(operation=c['operation'],params=c['params'],count=c['count'],thread_count=c['thread_count'],strides=c['strides'],bindings=tuple(c['arrays']),outputs=c['outputs'],matrix_layout='row-major',abi_profile=ABI_PROFILE,uniform_binding='entry-keywords-and-global-constant-buffer-fields',pixel_validation_profile='fp32-input-enclosure-and-strict-visible' if c['kind']=='project' else None)
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
            bounds=projection_enclosure(c)
            pixel_bounds.validate_pixels(av['meanPx'],av['visible'],bounds)
            visible=ev['visible']!=0
            np.testing.assert_allclose(av['meanPx'][visible],ev['meanPx'][visible],rtol=2e-6,atol=2e-6)
            for field in ('conic','depth'):np.testing.assert_allclose(av[field],ev[field],rtol=2e-6,atol=2e-6)
        else:
            n=5 if c['kind']=='probe' else c['count']
            np.testing.assert_allclose(a.view(np.float32)[:n],e.view(np.float32)[:n],rtol=2e-6,atol=2e-6)



def projection_enclosure(c):
    a=c['arrays'];g=a['gaussians'].view(GAUSSIAN_DTYPE)[:c['count']];cam=a['camera'].view(CAMERA_DTYPE)[0]
    return pixel_bounds.projected_pixel_bounds(g['mean'],*(cam[k] for k in ('worldToCamera','focal','principal','imageSize')))


def pixel_validation_summary(c,actual,expected):
    bounds=projection_enclosure(c);widths=bounds['upper']-bounds['lower']
    a=actual['projected'].view(PROJECTED_DTYPE)[:c['count']];e=expected['projected'].view(PROJECTED_DTYPE)[:c['count']]
    error=np.abs(a['meanPx'].astype(np.float64)-e['meanPx'].astype(np.float64));visible=e['visible']!=0
    return dict(model_sha256=hashlib.sha256(Path(pixel_bounds.__file__).read_bytes()).hexdigest(),assumptions='Input-derived FP32 gamma7 camera-dot enclosure; outward perspective intervals with1ULP add/multiply and2.5ULP division, finite no-overflow inputs; unresolved rejection/viewport masks block qualification',enclosure_width_px=dict(median=float(np.median(widths)),maximum=float(np.max(widths))),observed_deviation_from_float64_oracle_rounded_to_fp32_px=dict(maximum=float(np.max(error)),maximum_visible=float(np.max(error[visible])) if np.any(visible) else None),unresolved_classifications=int(np.count_nonzero(bounds['classification']<0)),visible_accuracy='Existing2e-6 absolute/relative check also retained for visible pixel coordinates; conic/depth/Adam/probe checks unchanged')

def validate_layout(kernel,c):
    """Fail closed on actual runtime reflected storage ABI before any dispatch."""
    layout=kernel.program.layout
    if c['kind']=='project':types={'gaussians':('GaussianData',GAUSSIAN_DTYPE),'projected':('ProjectedGaussian',PROJECTED_DTYPE),'camera':('CameraData',CAMERA_DTYPE)}
    elif c['kind']=='probe':types={'output':('BackwardProbe',np.dtype({'names':['dRgb','dAlpha','dInputTransmittance'],'formats':[('f4',3),'f4','f4'],'offsets':[0,12,16],'itemsize':20}))}
    else:types={}
    observed={};parameters={p.name:p for p in layout.parameters}
    for binding,(name,dtype) in types.items():
        # Reflect the actual bound resource context, not a standalone/default type.
        t=parameters[binding].type_layout.element_type_layout
        fields={f.name:int(f.offset) for f in t.fields}
        expected={k:int(dtype.fields[k][1]) for k in dtype.names}
        if fields!=expected or int(t.stride)!=dtype.itemsize:raise RuntimeError(f'unsupported runtime ABI for {name}: {fields}, stride{t.stride}')
        observed[name]=dict(binding=binding,stride=int(t.stride),offsets=fields)
    group=tuple(int(x) for x in layout.entry_points[0].compute_thread_group_size)
    expected={'adam':(32,1,1),'project':(128,1,1),'probe':(1,1,1)}[c['kind']]
    if group!=expected:raise RuntimeError('unexpected compiled thread-group profile')
    return dict(abi_profile=ABI_PROFILE,structs=observed,thread_group=group,matrix_layout='explicit compiler row-major')


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
    dependencies=dependency_identities();catalog={o['id']:o for o in list_operations()};device=None;records=[];compiler_dump=None;projection_program=None
    try:
        if gpu:
            import slangpy as spy
            from sintergpu.adapters.slangpy import SlangPyAdapter
            adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
            if not device_index<len(adapters):raise ValueError('invalid hardware device index')
            info=adapters[device_index]
            if info.vendor_id==0x10005:raise ValueError('hardware GPU required for timing')
            compiler_dump=tempfile.TemporaryDirectory(prefix='sintergpu-xsplat-compiler-')
            program_proof=json.loads((Path(__file__).resolve().parents[1]/'qualification/evidence/xsplat-project-program-0.2.3.json').read_text())
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=info.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise,'matrix_layout':spy.SlangMatrixLayout.row_major,'dump_intermediates':True,'dump_intermediates_prefix':str(Path(compiler_dump.name)/'actual-')})
            limits=device.info.limits
            if limits.max_compute_threads_per_group<128 or limits.max_compute_thread_group_size.x<128 or int(device.info.timestamp_frequency)<=0:raise RuntimeError('128-thread groups and GPU timestamps required')
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compiler=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compiler)!=1:raise RuntimeError('identify exactly one bundled compiler')
            if hashlib.sha256(compiler[0].read_bytes()).hexdigest()!=program_proof['compiler_sha256']:raise RuntimeError('pixel-model compiler identity requires qualification')
            env.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=info.vendor_id,device_id=info.device_id,slangpy=importlib.metadata.version('slangpy'),compiler_sha256=hashlib.sha256(compiler[0].read_bytes()).hexdigest(),floating_point_mode='precise',matrix_layout='row-major',timestamp_frequency=int(device.info.timestamp_frequency),buffer_budget_bytes=BUDGET,abi_profile=ABI_PROFILE)
            try:
                diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30);env['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):env['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases():
            op=catalog[case['operation']];arrays=case['arrays'];outputs=case['outputs'];execute=lambda:reference_call(case)
            expected=execute();metrics=dict(cpu_reference_wall_ms=sample_cpu(execute,iterations=iterations,warmups=warmups))
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runner_helper_sha256':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest(),'pixel_model_sha256':hashlib.sha256(Path(pixel_bounds.__file__).read_bytes()).hexdigest(),'program_inspector_sha256':hashlib.sha256(Path(program_precision.__file__).read_bytes()).hexdigest(),'workload':workload_identity(case),'environment':dict(env),'metrics':metrics,'iterations':iterations,'warmups':warmups,'scope':'Independent float64 NumPy math oracle plus byte-output construction/copies/temporary arrays. Prepared inputs outside both regions. GPU per-entry timestamps exclude compilation, allocation and all output/inout resets. Host submit/complete and complete storage-buffer uploads/full output readbacks are separate. Uniform fields are serialized by SlangPy on every dispatch and included in host/transfer time. Fixed AD probe is diagnostic only. No full renderer/training loop, optimized CPU, controlled performance comparison, variant selection or full-domain qualification.'}
            if gpu:
                if (case['thread_count'][0]+({'adam':32,'project':128,'probe':1}[case['kind']])-1)//({'adam':32,'project':128,'probe':1}[case['kind']])>int(limits.max_compute_dispatch_thread_groups.x) or any(a.nbytes>BUDGET for a in arrays.values()):raise ValueError('fixture exceeds dispatch or declared per-buffer budget')
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={name:device.create_buffer(element_count=a.nbytes//case['strides'][name],struct_size=case['strides'][name],usage=usage,data=a) for name,a in arrays.items() if name!='camera'}
                kernel=adapter.kernel(op['id']);row['reflected_abi']=validate_layout(kernel,case)
                if case['kind']=='project':
                    if projection_program is None:
                        _=kernel.pipeline  # Force actual code generation before timing/dispatch.
                        projection_program=program_precision.require_reviewed_program(compiler_dump.name,program_proof['binary_sha256'])
                    row['compiled_program']=dict(projection_program,reviewed_proof='qualification/evidence/xsplat-project-program-0.2.3.json')
                bindings=dict(buffers)
                if case['kind']=='project':
                    cam=arrays['camera'].view(CAMERA_DTYPE)[0]
                    # ConstantBuffer is written field-wise by this API, not bound
                    # as a Buffer/ShaderObject (SlangPy0.43.1 cursor_utils.h).
                    bindings['camera']=dict(worldToCamera=spy.float4x4(cam['worldToCamera']),focal=spy.float2(cam['focal'].tolist()),principal=spy.float2(cam['principal'].tolist()),imageSize=spy.uint2([int(v) for v in cam['imageSize']]),gaussianCount=int(cam['gaussianCount']))
                entry_uniforms={'p':case['params']} if case['kind']=='adam' else {};pool=device.create_query_pool(spy.QueryType.timestamp,2)
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
                    for name,buffer in buffers.items():buffer.copy_from_numpy(arrays[name])
                    dispatch();device.wait();return read()
                for _ in range(warmups):check_parity(case,transfer(),expected)
                for _ in range(iterations):
                    start=time.perf_counter_ns();actual=transfer();transfer_ms.append((time.perf_counter_ns()-start)/1e6)
                check_parity(case,actual,expected)
                if case['kind']=='project':row['pixel_validation']=pixel_validation_summary(case,actual,expected)
                metrics.update(gpu_kernel_ms=kernel_ms,host_submit_complete_ms=host_ms,transfer_inclusive_ms=transfer_ms)
                uniform_bytes=92 if case['kind']=='project' else 24 if case['kind']=='adam' else 0
                buffer_bytes=sum(arrays[k].nbytes for k in buffers)
                row.update(uploaded_bytes_per_iteration=buffer_bytes+uniform_bytes,uploaded_buffer_bytes_per_iteration=buffer_bytes,uniform_bytes_per_dispatch=uniform_bytes,uniform_payload_scope="Logical field bytes; not physical bus traffic or driver allocation size",readback_bytes_per_iteration=sum(arrays[k].nbytes for k in outputs),correctness='Independent math parity before/after isolated and transfer samples. Conic/depth/Adam/probe and visible pixels2e-6absolute/relative; all pixels also input-derived FP32 enclosure; ambiguous classifications fail. Visible flags and complete tail structs exact; within-live-struct padding is unspecified. All Adam inout state reset between samples.')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return dict(schema_version=1,environment=env,measurements=records,scope='Paired xsplat math snapshots; no renderer/training-loop/performance qualification')
    finally:
        if device is not None:device.close()
        if compiler_dump is not None:compiler_dump.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());output=args.pop('output');report=run(**args)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(report['measurements']),output=str(output))))
