# SPDX-License-Identifier: Apache-2.0
"""Paired synthetic per-entry VKL baselines; benchmark owns all test resources."""
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
from sintergpu.references.vkl import numpy as core
from sintergpu.references import geometry_depth_numeric as geometry
from sintergpu.references import depth_resampling_numeric as resampling
from sintergpu.references import temporal_best_numeric as temporal
from sintergpu import vkl as setup
from benchmarks.locus_timings import sample_cpu,cpu_identity

EPS=float(np.finfo(np.float32).eps)
LIMIT=65535
BUDGET=8<<20


def _numeric(c):
    a=c['arrays'];p=c['params'];kind=c['kind'];shape=c['shape'];b,_,h,w=shape
    scalar=lambda name:a[name][:b*h*w].reshape(shape)
    if kind=='projection':
        n=p['points'];result=core.project_depth_evidence(a['xyz'].reshape(b,n,3),a['visible'].reshape(b,n)!=0,a['provisional'].reshape(b,n)!=0,a['depthCandidate'].reshape(b,n)!=0,a['worldToCamera'].reshape(b,4,4),a['intrinsics'].reshape(b,3,3),a['sensorDepth'].reshape(shape),support_threshold=p['supportThreshold'],conflict_threshold=p['conflictThreshold'])
        return {dict(eligible='eligibleOut',projected_valid='projectedOut',sensor_valid='sensorOut',support='supportOut',conflict='conflictOut',depth_residual='residualOut')[k]:v for k,v in result.items()}
    if kind=='photometric':
        kwargs=dict(exposure_ev=a['exposureEV'],rgb_gain=a['rgbGain'].reshape(b,3),vignette=a['vignette'].reshape(b,3,3),color_matrix=a['colorMatrix'].reshape(b,3,3),response_gamma=a['responseGamma']) if p['useExposure'] else {}
        return dict(rgbObserved=core.photometric_isp(a['rgbLinear'].reshape(b,3,h,w),**kwargs))
    if kind=='convert':
        result=core.depth_convert(a['values'],a['valid']!=0,a['confidence'],representation=p['representation'],scale=p['scale'],focal_baseline=p['focalBaseline'])
        return {dict(depth='depthOut',inverse_depth='inverseOut',valid='validOut',confidence='confidenceOut')[k]:v for k,v in result.items()}
    if kind=='pyramid':result=core.depth_pyramid_level(scalar('depth'),scalar('valid')!=0,scalar('confidence'))
    elif kind=='geometry':
        return {k+'Out':v for k,v in geometry.depth_geometry(scalar('depth'),a['intrinsics'].reshape(b,3,3),a['pose'].reshape(b,4,4) if p['useWorld'] else None,scalar('confidence') if p['useConfidence'] else None).items()}
    elif kind=='world':return dict(xyz_worldOut=geometry.world_transform(a['xyz'].reshape(b,3,h,w),a['pose'].reshape(b,4,4))['xyz_world'])
    elif kind in ('fusion','temporal_fusion'):
        s=(p['sources'],)+shape;args=(a['depths'].reshape(s),a['valids'].reshape(s)!=0,a['confidences'].reshape(s))
        if kind=='fusion':result=geometry.fuse_depth(*args,relative=bool(p['relative']),consistency_threshold=p['threshold'])
        else:result=geometry.temporal_depth_fuse(*args,a['visibility'].reshape(s)!=0,a['consistency'].reshape(s)!=0,a['trusted'].reshape(s)!=0,consistency_threshold=p['threshold'],min_support=p['minSupport'],min_temporal_support=p['minTemporalSupport'])
        result={k:v for k,v in result.items() if k!='metric_status'}
    elif kind=='best':
        n=p['neighbors'];s=(p['batches'],n);kwargs={}
        if p['useQuality']:kwargs['pairwise_quality']=a['pairwise_quality'].reshape(s)
        if p['useConfidence']:kwargs['pairwise_confidence']=a['pairwise_confidence'].reshape(s)
        result=temporal.temporal_best_observation(a['equivalence'].reshape(s)!=0,a['target_quality'],np.zeros(s,np.float32) if p['useQuality'] else a['neighbor_quality'].reshape(s),**kwargs)
    elif kind=='initialize':
        confidence=scalar('confidence') if p['useConfidence'] else np.ones(shape,np.float32)
        # Identity nearest lift supplies an independently validated sanitation contract.
        lift=resampling.depth_upsample(scalar('depth'),scalar('valid')!=0,confidence,size=(h,w))
        result={k:lift[k] for k in ('depth','valid','confidence')}
    elif kind=='labels':
        valid=scalar('valid')!=0
        result=dict(direct_valid=valid,propagated_valid=np.zeros(shape,bool),provenance=valid.astype(np.int32))
    elif kind=='lift':result=resampling.depth_upsample(scalar('depth'),scalar('valid')!=0,scalar('confidence'),size=(p['outputHeight'],p['outputWidth']))
    elif kind=='densify':result=resampling.depth_densify(scalar('depth'),scalar('valid')!=0,scalar('confidence'),a['rgb'].reshape(b,p['channels'],h,w),direct_valid=scalar('direct_valid')!=0,max_rgb_difference=p['maxRGBDifference'],max_relative_spread=p['maxRelativeSpread'],min_support=p['minSupport'],steps=1)
    else:raise ValueError('unknown benchmark contract '+kind)
    return {k+'Out':v for k,v in result.items()}


def _case(kind,operation,shape,params,inputs,count,*,label,reference):
    c=dict(kind=kind,operation=operation,shape=shape,params=params,arrays={k:np.ascontiguousarray(v).reshape(-1) for k,v in inputs.items()},thread_count=(count,1,1),label=label,reference=reference)
    expected=_numeric(c);c['outputs']=tuple(expected);c['live_counts']={k:v.size for k,v in expected.items()}
    for name,v in expected.items():
        dtype=np.uint32 if v.dtype.kind=='b' else np.int32 if v.dtype.kind in 'iu' else np.float32
        c['arrays'][name]=np.full(v.size+13,73,dtype)
    if kind=='geometry' and not params['useWorld']:
        c['arrays']['xyz_worldOut']=np.full(1,73,np.float32);c['outputs']+=('xyz_worldOut',);c['live_counts']['xyz_worldOut']=0
    for a in c['arrays'].values():
        if a.dtype not in (np.dtype('float32'),np.dtype('uint32'),np.dtype('int32')) or a.nbytes>BUDGET:raise ValueError('fixture exceeds typed buffer budget')
    if (count+255)//256>LIMIT:raise ValueError('fixture exceeds declared dispatch budget')
    return c


def cases():
    """Two irregular multi-group shapes, profiles, invalid inputs and equal ties."""
    for variant,(b,h,w) in enumerate(((2,31,35),(1,47,61))):
        rng=np.random.default_rng(26801+variant);shape=(b,1,h,w);count=b*h*w;label=f'{b}x{h}x{w}'
        depth=(1+rng.random(shape,dtype=np.float32)*np.float32(.1)).astype(np.float32);valid=rng.random(shape)>.2;confidence=rng.random(shape,dtype=np.float32)
        depth.reshape(-1)[::37]=np.nan;depth.reshape(-1)[1::41]=0;valid.reshape(-1)[::53]=False
        validity=valid.astype(np.uint32)*7
        intr=np.tile(np.eye(3,dtype=np.float32),(b,1,1));intr[:,0,0]=np.float32(30);intr[:,1,1]=np.float32(35);intr[:,0,2]=w/2;intr[:,1,2]=h/2
        pose=np.tile(np.eye(4,dtype=np.float32),(b,1,1));pose[:,:3,3]=np.array([.2,-.3,.4],np.float32)
        n=259 if not variant else 1031
        xyz=rng.normal(size=(b,n,3)).astype(np.float32);xyz[...,2]=rng.uniform(.25,2,(b,n));xyz[:,0]=np.nan
        flags=[(rng.random((b,n))>.25).astype(np.uint32)*7 for _ in range(3)]
        p=dict(batches=b,points=n,height=h,width=w,supportThreshold=.25,conflictThreshold=.25,epsilon=EPS)
        yield _case('projection','vkl.project_depth_evidence.projectDepthEvidence',shape,p,dict(xyz=xyz,visible=flags[0],provisional=flags[1],depthCandidate=flags[2],worldToCamera=pose,intrinsics=intr,sensorDepth=depth),b*n,label=label,reference=core)
        kwargs=dict(rgbLinear=rng.random((b,3,h,w),dtype=np.float32),exposureEV=rng.uniform(-1,1,b).astype(np.float32),rgbGain=rng.uniform(.2,2,(b,3)).astype(np.float32),vignette=rng.uniform(0,.1,(b,3,3)).astype(np.float32),colorMatrix=rng.uniform(-.5,1,(b,3,3)).astype(np.float32),responseGamma=rng.uniform(.3,1.5,b).astype(np.float32))
        kwargs['rgbLinear'].reshape(-1)[::31]=0
        for enabled in (False,True):
            p=dict(batches=b,height=h,width=w,epsilon=EPS,**{k:int(enabled) for k in ('useExposure','useGain','useVignette','useColor','useGamma')})
            yield _case('photometric','vkl.photometric_isp.photometricISP',shape,p,kwargs,count,label=label+f'/options-{int(enabled)}',reference=core)
        values=depth.reshape(-1).copy();values[2::43]=np.inf;values[3::47]=-1
        for representation in (0,1,2):
            yield _case('convert','vkl.depth_convert.depthConvert',shape,dict(count=count,representation=representation,scale=2.,focalBaseline=4.),dict(values=values,valid=validity,confidence=confidence),count,label=label+f'/representation-{representation}',reference=core)
        yield _case('pyramid','vkl.depth_pyramid.depthPyramidLevel',shape,dict(batches=b,height=h,width=w,epsilon=EPS),dict(depth=depth,valid=validity,confidence=confidence),b*((h+1)//2)*((w+1)//2),label=label,reference=core)
        for enabled in (False,True):
            plan=setup.geometry_layout(b,h,w,max_workgroups_x=LIMIT,world=enabled,confidence=enabled)
            yield _case('geometry','vkl.geometry.depthGeometry',shape,plan['params'],dict(depth=depth,intrinsics=intr,pose=pose if enabled else np.zeros(1,np.float32),confidence=confidence if enabled else np.zeros(1,np.float32)),count,label=label+f'/options-{int(enabled)}',reference=geometry)
        points=rng.normal(size=(b,3,h,w)).astype(np.float32);points[:,:,0,0]=0;points[0,0,0,1]=np.nan
        plan=setup.geometry_layout(b,h,w,max_workgroups_x=LIMIT,operation='world_transform')
        yield _case('world','vkl.geometry.worldTransform',shape,plan['params'],dict(xyz=points,pose=pose),count,label=label,reference=geometry)
        sources=3 if not variant else 4;s=(sources,)+shape
        depths=np.tile(depth,(sources,1,1,1,1));depths+=rng.uniform(-.03,.03,s).astype(np.float32);depths[-1,...,::7]=3
        valids=(rng.random(s)>.2).astype(np.uint32)*7;conf=rng.random(s,dtype=np.float32);conf[...,::29]=0
        for relative in (False,True):
            plan=setup.fusion_layout(b,h,w,sources,max_workgroups_x=LIMIT,relative=relative,threshold=.25)
            yield _case('fusion','vkl.depth_fusion.fuseDepth',shape,plan['params'],dict(depths=depths,valids=valids,confidences=conf),count,label=label+f'/relative-{int(relative)}',reference=geometry)
        plan=setup.fusion_layout(b,h,w,sources,max_workgroups_x=LIMIT,temporal=True,threshold=.25,min_support=2,min_temporal_support=1)
        yield _case('temporal_fusion','vkl.depth_fusion.temporalDepthFuse',shape,plan['params'],dict(depths=depths,valids=valids,confidences=conf,visibility=(rng.random(s)>.1).astype(np.uint32),consistency=(rng.random(s)>.1).astype(np.uint32),trusted=(rng.random(s)>.8).astype(np.uint32)),count,label=label,reference=geometry)
        neighbors=7 if not variant else 33;eq=(rng.random((n,neighbors))>.25).astype(np.uint32)*7;t=rng.random(n,dtype=np.float32);q=rng.random((n,neighbors),dtype=np.float32);pc=rng.random((n,neighbors),dtype=np.float32)
        q[0]=.75;t[0]=.25;eq[0]=7
        for enabled in (False,True):
            plan=setup.temporal_best_layout(n,neighbors,max_workgroups_x=LIMIT,quality=enabled,confidence=enabled)
            yield _case('best','vkl.temporal_best.temporalBest',(n,1,1,1),plan['params'],dict(equivalence=eq,target_quality=t,neighbor_quality=np.zeros(1,np.float32) if enabled else q,pairwise_quality=q if enabled else np.zeros(1,np.float32),pairwise_confidence=pc if enabled else np.zeros(1,np.float32)),n,label=label+f'/options-{int(enabled)}',reference=temporal)
        plan=setup.depth_pyramid_plan(b,h,w,max_workgroups_x=LIMIT,confidence=bool(variant))[0]
        yield _case('initialize','vkl.depth_resampling.initializeDepth',shape,plan['params'],dict(depth=depth,valid=validity,confidence=confidence if variant else np.zeros(1,np.float32)),count,label=label+f'/confidence-{variant}',reference=resampling)
        plan=setup.depth_downsample_plan(b,h,w,scale=2,max_workgroups_x=LIMIT)['stages'][-1]
        # Label stage consumes an already established mask, not raw depths.
        oh,ow=plan['shape'][-2:];labelvalid=(rng.random((b,1,oh,ow))>.2).astype(np.uint32)*7
        yield _case('labels','vkl.depth_resampling.labelDirectDepth',plan['shape'],plan['params'],dict(valid=labelvalid),b*oh*ow,label=label,reference=resampling)
        plan=setup.depth_lift_layout(b,h,w,h+4,w+6,max_workgroups_x=LIMIT)
        yield _case('lift','vkl.depth_resampling.depthUpsample',shape,plan['params'],dict(depth=depth,valid=validity,confidence=confidence),plan['thread_count'][0],label=label,reference=resampling)
        channels=1 if not variant else 3;rgb=rng.uniform(0,.1,(b,channels,h,w)).astype(np.float32)
        plan=setup.depth_densify_plan(b,h,w,max_workgroups_x=LIMIT,channels=channels,direct=True,max_rgb_difference=.15,max_relative_spread=.1,min_support=2)[0]
        yield _case('densify','vkl.depth_resampling.densifyDepth',shape,plan['params'],dict(depth=depth,valid=validity,confidence=confidence,rgb=rgb,direct_valid=validity),count,label=label+f'/channels-{channels}',reference=resampling)


def reference_call(c):
    result={k:c['arrays'][k].copy() for k in c['outputs']}
    for name,value in _numeric(c).items():result[name][:value.size]=value.reshape(-1)
    return result


def workload_identity(c):
    stage=dict(operation=c['operation'],params=c['params'],thread_count=c['thread_count'],bindings=tuple(c['arrays']),outputs=c['outputs'],live_counts=c['live_counts'],shape=c['shape'])
    h=hashlib.sha256(json.dumps(stage,sort_keys=True).encode())
    for name,a in sorted(c['arrays'].items()):h.update(json.dumps([name,a.dtype.str,a.shape]).encode()+b'\0'+a.tobytes()+b'\0')
    return dict(id='vkl-synthetic/'+c['kind']+'/'+c['label'],sha256=h.hexdigest(),description='Seeded irregular synthetic correctness workload; reference loops/allocations are not optimized CPU or production throughput',**stage)


def check_parity(c,actual,expected):
    assert set(actual)==set(expected)==set(c['outputs'])
    for name in c['outputs']:
        a=actual[name];e=expected[name];count=c['live_counts'][name]
        assert a.shape==e.shape and a.dtype==e.dtype
        np.testing.assert_array_equal(a[count:],c['arrays'][name][count:])
        np.testing.assert_array_equal(e[count:],c['arrays'][name][count:])
        if a.dtype.kind in 'iu':np.testing.assert_array_equal(a[:count],e[:count],err_msg=name)
        elif c['kind']=='convert':np.testing.assert_allclose(a[:count],e[:count],atol=0,rtol=2e-6,equal_nan=True,err_msg=name)
        elif c['kind']=='projection':np.testing.assert_allclose(a[:count],e[:count],atol=3e-7,rtol=2e-6,equal_nan=True,err_msg=name)
        elif c['kind']=='photometric':np.testing.assert_allclose(a[:count],e[:count],atol=3e-7,rtol=2e-5,equal_nan=True,err_msg=name)
        elif c['kind']=='pyramid':np.testing.assert_allclose(a[:count],e[:count],atol=1e-7,rtol=2e-6,equal_nan=True,err_msg=name)
        else:np.testing.assert_allclose(a[:count],e[:count],atol=5e-7,rtol=8e-6,equal_nan=True,err_msg=name)


def run(*,gpu=False,iterations=50,warmups=5,device_index=0):
    from sintergpu import __version__
    from sintergpu.catalog import list_operations
    from sintergpu.timings import dependency_identities,summarize
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000:raise ValueError('bounded positive sample/warmup counts required')
    if gpu and os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in required after independent measurement clearance')
    env=dict(cpu=cpu_identity(),logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,platform=platform.platform(),python=platform.python_version(),numpy=importlib.metadata.version('numpy'),library_version=__version__,captured_at_utc=datetime.now(timezone.utc).isoformat(),gpu=None,compiler_sha256=None)
    for name in ('scaling_governor','scaling_cur_freq'):
        path=Path('/sys/devices/system/cpu/cpu0/cpufreq')/name;env[name]=path.read_text().strip() if path.is_file() else None
    dependencies=dependency_identities();catalog={o['id']:o for o in list_operations()};device=None;records=[]
    try:
        if gpu:
            import slangpy as spy
            from sintergpu.adapters.slangpy import SlangPyAdapter
            adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
            if type(device_index) is not int or not 0<=device_index<len(adapters):raise ValueError('invalid hardware device index')
            info=adapters[device_index]
            if info.vendor_id==0x10005:raise ValueError('hardware GPU required for timing')
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=info.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
            limits=device.info.limits
            if limits.max_compute_threads_per_group<256 or limits.max_compute_thread_group_size.x<256 or int(device.info.timestamp_frequency)<=0:raise RuntimeError('256-thread groups and GPU timestamps required')
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compiler=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compiler)!=1:raise RuntimeError('identify exactly one bundled compiler')
            env.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=info.vendor_id,device_id=info.device_id,slangpy=importlib.metadata.version('slangpy'),compiler_sha256=hashlib.sha256(compiler[0].read_bytes()).hexdigest(),floating_point_mode='precise',timestamp_frequency=int(device.info.timestamp_frequency),buffer_budget_bytes=BUDGET)
            try:
                diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
                env['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):env['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases():
            op=catalog[case['operation']];arrays=case['arrays'];outputs=case['outputs'];execute=lambda:reference_call(case)
            expected=execute();metrics=dict(cpu_reference_wall_ms=sample_cpu(execute,iterations=iterations,warmups=warmups))
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(case['reference'].__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'runner_helper_sha256':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest(),'setup_source_sha256':hashlib.sha256(Path(setup.__file__).read_bytes()).hexdigest(),
                'workload':workload_identity(case),'environment':dict(env),'metrics':metrics,'iterations':iterations,'warmups':warmups,
                'scope':'CPU independent NumPy oracle includes output construction/copies, scalar loops and temporary allocations on prepared float32-valued inputs. GPU per-entry timestamps exclude compile, allocation and complete output reset. Host submit/complete and full bound-buffer upload/complete output readback costs are separate. Synthetic irregular fixtures, not optimized CPU, full pipeline, trajectory, production throughput or optimized-variant selection.'}
            if gpu:
                count=case['thread_count'][0]
                if (count+255)//256>int(limits.max_compute_dispatch_thread_groups.x) or any(a.nbytes>BUDGET for a in arrays.values()):raise ValueError('fixture exceeds device dispatch or declared buffer budget')
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={name:device.create_buffer(element_count=a.size,struct_size=a.dtype.itemsize,usage=usage,data=a) for name,a in arrays.items()}
                kernel=adapter.kernel(op['id']);bindings={**buffers,'params':case['params']};pool=device.create_query_pool(spy.QueryType.timestamp,2)
                row['adapter_source_identity']=adapter.source_identities[op['id']]
                def reset():
                    for name in outputs:buffers[name].copy_from_numpy(arrays[name])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=case['thread_count'],vars=bindings,**kwargs)
                def read():return {name:buffers[name].to_numpy().view(arrays[name].dtype).reshape(-1) for name in outputs}
                for _ in range(warmups):reset();dispatch();device.wait()
                check_parity(case,read(),expected);kernel_ms=[];host_ms=[];transfer_ms=[]
                for _ in range(iterations):
                    reset();pool.reset();start=time.perf_counter_ns()
                    dispatch(query_pool=pool,query_index_before=0,query_index_after=1);device.wait()
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
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays.values()),readback_bytes_per_iteration=sum(arrays[k].nbytes for k in outputs),correctness='Full output live values match independent reference under existing operation tolerances; integer/mask outputs and all output tails exact before/after isolated and transfer regions')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return dict(schema_version=1,environment=env,measurements=records,scope='Paired synthetic per-entry VKL CPU/GPU baselines; no full application or performance qualification')
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());output=args.pop('output');report=run(**args)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(report['measurements']),output=str(output))))
