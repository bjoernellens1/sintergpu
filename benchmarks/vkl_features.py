# SPDX-License-Identifier: Apache-2.0
"""Standalone binary-feature qualification/timing support; owns test resources."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import numpy as np
from sintergpu.features import record_feature_plan


class FeatureRunner:
    def __init__(self,device_index=0):
        if os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in and independent source clearance required')
        import slangpy as spy
        from sintergpu.adapters.slangpy import SlangPyAdapter
        self.spy=spy
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if type(device_index) is not int or not 0<=device_index<len(adapters):raise ValueError('invalid device index')
        a=adapters[device_index]
        if a.vendor_id==0x10005:raise ValueError('hardware GPU required for qualification')
        self.device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=a.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if not self.device.has_feature(spy.Feature.int64):raise RuntimeError('shaderInt64 required')
            if self.device.info.limits.max_compute_threads_per_group<256 or self.device.info.limits.max_compute_thread_group_size.x<256:raise RuntimeError('256-thread groups required')
            self.adapter=SlangPyAdapter(self.device,allow_experimental=True)
            # A conservative 8-MiB test budget, not a reported hardware maximum.
            self.limits=dict(max_workgroups_x=int(self.device.info.limits.max_compute_dispatch_thread_groups.x),max_storage_buffer_range=8<<20)
            files=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(files)!=1:raise RuntimeError('identify exactly one bundled compiler')
            self.environment=dict(gpu=self.device.info.adapter_name,api=self.device.info.api_name,vendor_id=a.vendor_id,device_id=a.device_id,slangpy=importlib.metadata.version('slangpy'),compiler_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),floating_point_mode='precise',shader_int64=True,buffer_budget_bytes=8<<20)
        except BaseException:
            self.device.close();raise

    def close(self):self.device.close()

    def prepare(self,plan,inputs):
        arrays={};buffers={};usage=self.spy.BufferUsage.shader_resource|self.spy.BufferUsage.unordered_access
        for name,count in plan['elements'].items():
            dtype=np.dtype(plan['dtypes'][name]);fill=-77 if dtype==np.int64 else np.nan if dtype==np.float32 else 0xdeadbeef
            a=np.ascontiguousarray(inputs[name],dtype=dtype).reshape(-1) if name in inputs else np.full(max(count,1),fill,dtype)
            if a.size<max(count,1):raise ValueError('insufficient benchmark input: '+name)
            arrays[name]=a.copy();buffers[name]=self.device.create_buffer(element_count=a.size,struct_size=dtype.itemsize,usage=usage,data=a)
        kernels={op:self.adapter.kernel(op) for op in plan['operations']}
        return dict(plan=plan,arrays=arrays,buffers=buffers,kernels=kernels)

    def execute(self,prepared):
        encoder=self.device.create_command_encoder()
        record_feature_plan(prepared['kernels'],prepared['buffers'],prepared['plan'],command_encoder=encoder)
        self.device.submit_command_buffer(encoder.finish());self.device.wait()

    def read(self,prepared,names):
        return {name:prepared['buffers'][name].to_numpy().view(prepared['arrays'][name].dtype).reshape(-1) for name in names}


OUTPUTS={
    'briefDescribe':('descriptorOut','briefValidOut'),
    'packBinary':('landmarkPacked','candidatePacked'),
    'initializeMatch':('bestDistance','bestIndex'),
    'matchPairs':('pairDistance',),
    'rowMinimum':('bestDistance','bestIndex'),
    'reverseMinimum':('reverseIndex',),
    'finalizeMatch':('matchedIndexOut','distanceOut','matchValidOut'),
}


def stage_reference(stage,arrays):
    """Independent NumPy stage oracle, including untouched output tail/state."""
    from sintergpu.references import features_numeric as ref
    p=stage['params'];entry=stage['entrypoint'];a=arrays
    result={name:a[name].copy() for name in OUTPUTS[entry]}
    b=p['batches'];d=p['bits']
    if entry=='briefDescribe':
        n=p['points'];h=p['height'];w=p['width'];c=p['channels']
        value=ref.brief_describe(a['rgb'][:b*c*h*w].reshape(b,c,h,w),a['pixels'][:b*n*2].reshape(b,n,2),
            pixels_valid=a['pixelsValid'][:b*n].reshape(b,n).astype(bool) if p['usePixelsValid'] else None,
            rgb_valid=a['rgbValid'][:b*h*w].reshape(b,1,h,w).astype(bool) if p['useRgbValid'] else None,
            pairs=None if p['defaultPairs'] else a['pairs'][:d*4].reshape(d,2,2),num_bits=d)
        result['descriptorOut'][:b*n*d]=value['descriptor'].reshape(-1)
        result['briefValidOut'][:b*n]=value['valid'].reshape(-1)
        return result
    m=p['landmarks'];n=p['candidates'];words=p['words'];start=p['chunkStart'];count=p['chunkCount']
    if entry=='packBinary':
        for prefix,points in [('landmark',m),('candidate',n)]:
            descriptor=a[prefix+'Descriptor'][:b*points*d].reshape(b,points,d)!=0
            packed=np.zeros((b,points,words),np.uint32)
            for bit in range(d):packed[:,:,bit//32]|=descriptor[:,:,bit].astype(np.uint32)<<np.uint32(bit%32)
            result[prefix+'Packed'][:b*points*words]=packed.reshape(-1)
    elif entry=='initializeMatch':
        result['bestDistance'][:b*m]=d+1;result['bestIndex'][:b*m]=0
    elif entry=='matchPairs':
        xy=a['projected'][:b*m*2].reshape(b,m,2);cp=a['candidateXY'][:b*n*2].reshape(b,n,2)[:,start:start+count].astype(np.float32)
        with np.errstate(over='ignore',invalid='ignore'):
            delta=xy[:,:,None]-cp[:,None]
            local=np.sum(delta*delta,axis=-1,dtype=np.float32)<=np.float32(p['radiusSquared'])
        local&=np.isfinite(xy).all(-1)[:,:,None]
        if p['useLandmarkValid']:local&=a['landmarkValid'][:b*m].reshape(b,m,1)!=0
        if p['useCandidateValid']:local&=a['candidateValid'][:b*n].reshape(b,1,n)[:,:,start:start+count]!=0
        first=a['landmarkPacked'][:b*m*words].reshape(b,m,words)
        second=a['candidatePacked'][:b*n*words].reshape(b,n,words)[:,start:start+count]
        xor=np.ascontiguousarray(first[:,:,None]^second[:,None])
        lut=np.array([i.bit_count() for i in range(256)],np.uint8)
        distance=lut[xor.view(np.uint8)].sum(-1,dtype=np.uint32)
        result['pairDistance'][:b*m*count]=np.where(local,distance,d+1).reshape(-1)
    elif entry=='rowMinimum':
        distances=a['pairDistance'][:b*m*count].reshape(b*m,count)
        index=distances.argmin(-1);value=distances[np.arange(b*m),index]
        improve=value<a['bestDistance'][:b*m]
        result['bestDistance'][:b*m]=np.where(improve,value,a['bestDistance'][:b*m])
        result['bestIndex'][:b*m]=np.where(improve,index+start,a['bestIndex'][:b*m])
    elif entry=='reverseMinimum':
        distances=a['pairDistance'][:b*m*count].reshape(b,m,count)
        result['reverseIndex'][:b*n].reshape(b,n)[:,start:start+count]=distances.argmin(1)
    elif entry=='finalizeMatch':
        index=a['bestIndex'][:b*m].reshape(b,m);distance=a['bestDistance'][:b*m].reshape(b,m)
        valid=np.isfinite(a['projected'][:b*m*2].reshape(b,m,2)).all(-1)&(index<n)&(distance<=p['maxDistance'])
        if p['useLandmarkValid']:valid&=a['landmarkValid'][:b*m].reshape(b,m)!=0
        if p['mutual']:valid&=np.take_along_axis(a['reverseIndex'][:b*n].reshape(b,n),np.minimum(index,n-1).astype(np.int64),axis=1)==np.arange(m)[None]
        for name,value in [('matchedIndexOut',np.where(valid,index.astype(np.int64),-1)),('distanceOut',np.where(valid,distance,0)),('matchValidOut',valid)]:result[name][:b*m]=value.reshape(-1)
    else:raise ValueError('unknown stage '+entry)
    return result


def _seed_arrays(plan,inputs):
    arrays={}
    for name,count in plan['elements'].items():
        dtype=np.dtype(plan['dtypes'][name]);fill=-77 if dtype==np.int64 else np.nan if dtype==np.float32 else 0xdeadbeef
        # Include poisoned output/input tails to detect over-dispatch writes/reads.
        a=np.full(max(count,1)+3,fill,dtype)
        if name in inputs:a[:np.asarray(inputs[name]).size]=np.asarray(inputs[name],dtype=dtype).reshape(-1)
        arrays[name]=a
    return arrays


def cases():
    """Seeded successful/invalid features, cross-256 reductions and last chunk."""
    from sintergpu.features import brief_layout,binary_match_plan
    limits=dict(max_workgroups_x=65535,max_storage_buffer_range=8<<20)
    rng=np.random.default_rng(25719)
    b,c,h,w,n,d=1,3,43,47,259,259
    plan=brief_layout(b,c,h,w,n,d,**limits,use_pixels_valid=True,use_rgb_valid=True)
    inputs=dict(rgb=rng.normal(size=(b,c,h,w)).astype(np.float32),pixels=np.stack((rng.integers(15,w-15,size=(b,n)),rng.integers(15,h-15,size=(b,n))),axis=-1),pixelsValid=np.ones(b*n,np.uint32),rgbValid=np.ones(b*h*w,np.uint32))
    inputs['pixelsValid'][::19]=0
    a=_seed_arrays(plan,inputs);stage=plan['stages'][0]
    yield dict(stage=stage,arrays={k:a[k].copy() for k in stage['bindings']},outputs=OUTPUTS[stage['entrypoint']],expected=stage_reference(stage,a),original=inputs)
    b,m,n,d=2,257,259,259
    plan=binary_match_plan(b,m,n,d,**limits,radius_px=5.,max_hamming_distance=160,chunk_size=129,mutual=True,use_landmark_valid=True,use_candidate_valid=True)
    descriptor=rng.integers(0,2,(b,n,d),dtype=np.uint32)
    xy=rng.integers(0,40,(b,n,2),dtype=np.int64)
    inputs=dict(landmarkDescriptor=descriptor[:,:m].copy(),candidateDescriptor=descriptor,projected=xy[:,:m].astype(np.float32),candidateXY=xy,
        landmarkValid=np.ones(b*m,np.uint32),candidateValid=np.ones(b*n,np.uint32))
    inputs['projected'][0,0]=np.nan;inputs['landmarkValid'][::23]=0;inputs['candidateValid'][::31]=0
    a=_seed_arrays(plan,inputs);seen=set()
    for stage in plan['stages']:
        entry=stage['entrypoint'];p=stage['params'];expected=stage_reference(stage,a)
        if entry not in seen or (entry in ('matchPairs','rowMinimum','reverseMinimum') and p['chunkCount']==1):
            yield dict(stage=stage,arrays={k:a[k].copy() for k in stage['bindings']},outputs=OUTPUTS[entry],expected=expected,original=inputs)
            seen.add(entry)
        a.update(expected)


def workload_identity(case):
    import json
    stage=case['stage'];p=stage['params'];h=hashlib.sha256(json.dumps(stage,sort_keys=True).encode())
    for name,a in sorted(case['arrays'].items()):h.update(json.dumps([name,a.dtype.str,a.shape]).encode()+b'\0'+a.tobytes()+b'\0')
    return dict(id=f"vkl-features/{stage['entrypoint']}/chunk-{p.get('chunkStart',0)}-{p.get('chunkCount',0)}",sha256=h.hexdigest(),
        description='Deterministic synthetic feature-stage fixture with full output/state tails; not trajectory or production pipeline qualification',params=p,thread_count=stage['thread_count'],bindings=stage['bindings'])


def run(*,gpu=False,iterations=50,warmups=5,device_index=0):
    from datetime import datetime,timezone
    import platform
    import subprocess
    import time
    from benchmarks.locus_timings import cpu_identity,sample_cpu
    from sintergpu import __version__
    import sintergpu.features as feature_setup
    from sintergpu.catalog import list_operations
    from sintergpu.timings import dependency_identities,summarize
    from sintergpu.references import features_numeric as reference
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000:raise ValueError('bounded positive sample/warmup counts required')
    env=dict(cpu=cpu_identity(),platform=platform.platform(),python=platform.python_version(),numpy=importlib.metadata.version('numpy'),logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,library_version=__version__,captured_at_utc=datetime.now(timezone.utc).isoformat(),gpu=None,compiler_sha256=None)
    for name in ('scaling_governor','scaling_cur_freq'):
        path=Path('/sys/devices/system/cpu/cpu0/cpufreq')/name;env[name]=path.read_text().strip() if path.is_file() else None
    runner=None;records=[];catalog={o['id']:o for o in list_operations()};dependencies=dependency_identities()
    try:
        if gpu:
            runner=FeatureRunner(device_index);env.update(runner.environment)
            if int(runner.device.info.timestamp_frequency)<=0:raise RuntimeError('GPU timestamps unavailable')
            env['timestamp_frequency']=int(runner.device.info.timestamp_frequency)
            try:
                result=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
                env['vulkaninfo_summary']=result.stdout if result.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):env['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases():
            stage=case['stage'];arrays=case['arrays'];outputs=case['outputs'];operation=stage['operation'];op=catalog[operation]
            execute=lambda:stage_reference(stage,arrays)
            expected=execute();metrics=dict(cpu_reference_wall_ms=sample_cpu(execute,iterations=iterations,warmups=warmups))
            row={**dependencies,'operation':operation,'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'setup_source_sha256':hashlib.sha256(Path(feature_setup.__file__).read_bytes()).hexdigest(),'runner_helper_sha256':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest(),
                'workload':workload_identity(case),'environment':dict(env),'metrics':metrics,'iterations':iterations,'warmups':warmups,
                'scope':'CPU NumPy stage oracle includes output copies, computation and temporary allocations on prepared inputs. GPU per-entry timestamps exclude compile, allocation and required state reset. Host submit/complete and all-binding upload plus complete output/state readback measured separately. Streaming stages use frozen intermediate states; per-entry times are not whole-match pipeline times or optimized CPU comparisons.'}
            if gpu:
                device=runner.device;spy=runner.spy;usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={k:device.create_buffer(element_count=a.size,struct_size=a.dtype.itemsize,usage=usage,data=a) for k,a in arrays.items()}
                kernel=runner.adapter.kernel(operation);bindings={**buffers,'Params':stage['params']};pool=device.create_query_pool(spy.QueryType.timestamp,2)
                row['adapter_source_identity']=runner.adapter.source_identities[operation]
                def reset():
                    for name in outputs:buffers[name].copy_from_numpy(arrays[name])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=stage['thread_count'],vars=bindings,**kwargs)
                def read():return {name:buffers[name].to_numpy().view(arrays[name].dtype).reshape(-1) for name in outputs}
                def check(actual):
                    for name in outputs:np.testing.assert_array_equal(actual[name],expected[name],err_msg=operation+' '+name)
                for _ in range(warmups):reset();dispatch();device.wait()
                check(read());kernel_ms=[];host_ms=[];transfer_ms=[]
                for _ in range(iterations):
                    reset();pool.reset();start=time.perf_counter_ns()
                    dispatch(query_pool=pool,query_index_before=0,query_index_after=1);device.wait()
                    host_ms.append((time.perf_counter_ns()-start)/1e6);stamps=pool.get_timestamp_results(0,2);kernel_ms.append((stamps[1]-stamps[0])*1000)
                check(read())
                def transfer():
                    for name,a in arrays.items():buffers[name].copy_from_numpy(a)
                    dispatch();device.wait();return read()
                for _ in range(warmups):check(transfer())
                for _ in range(iterations):
                    start=time.perf_counter_ns();actual=transfer();transfer_ms.append((time.perf_counter_ns()-start)/1e6)
                check(actual)
                metrics.update(gpu_kernel_ms=kernel_ms,host_submit_complete_ms=host_ms,transfer_inclusive_ms=transfer_ms)
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays.values()),readback_bytes_per_iteration=sum(arrays[k].nbytes for k in outputs),correctness='Exact independent CPU stage parity, including entire outputs/state and poisoned tails before and after isolated and transfer samples')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return dict(schema_version=1,environment=env,measurements=records,scope='Paired synthetic per-entry CPU/GPU feature timings; no full-pipeline performance selection')
    finally:
        if runner is not None:runner.close()


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());output=args.pop('output');record=run(**args)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(record['measurements']),output=str(output))))
