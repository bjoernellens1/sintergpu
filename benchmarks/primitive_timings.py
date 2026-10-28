# SPDX-License-Identifier: Apache-2.0
"""Independent per-entry scan/compaction snapshots; benchmark owns all resources."""
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
from sintergpu.references import scan as reference
from benchmarks.locus_timings import sample_cpu,cpu_identity

BUDGET=8<<20
TAIL=13


def padded(values,dtype=None):
    data=np.asarray(values,dtype=dtype).reshape(-1)
    seed=-73 if data.dtype==np.float32 else 0xdeadbeef
    return np.concatenate((data,np.full(TAIL,seed,data.dtype)))


def _block_prefix(values,ids,product):
    output=np.empty_like(values)
    with np.errstate(over='ignore',invalid='ignore'):
        for first in range(0,len(values),256):
            output[first:first+256]=reference.segmented_scan(values[first:first+256],ids[first:first+256],product=product)
    ends=np.minimum(np.arange(256,len(values)+256,256),len(values))-1
    return output,output[ends],ids[ends]


def make_scan_case(entry,dtype,count,product=False):
    dtype=np.dtype(dtype);rng=np.random.default_rng(28400+count)
    ids=(np.arange(count,dtype=np.uint32)//1000)*2
    if count==1025:
        # Explicit integer increments: NumPy boolean addition would collapse resets.
        ids=(np.arange(count)>=17).astype(np.uint32)*3+(np.arange(count)>=1000).astype(np.uint32)*5
    elif count==70001:ids[:]=0
    else:ids+=(np.arange(count)>=17)+(np.arange(count)>=256)
    if dtype==np.uint32:
        values=rng.choice(np.array([1,3,5,7],np.uint32),count) if product else rng.integers(0,2**32,count,dtype=np.uint32)
    elif product:values=rng.uniform(.999,1.001,count).astype(dtype)
    else:values=np.ones(count,dtype) if count==70001 else rng.uniform(-1,1,count).astype(dtype)
    if product and count!=70001:values[127::385]=0
    blocks=(count+255)//256
    seed=np.full(count,-73 if dtype==np.float32 else 0xdeadbeef,dtype)
    arrays=dict(segmentIds=padded(ids))
    if entry=='scanBlocks':
        arrays.update(values=padded(values),output=padded(seed),blockValues=padded(np.full(blocks,seed[0],dtype)),blockIds=padded(np.full(blocks,0xdeadbeef,np.uint32)))
        outputs=('output','blockValues','blockIds');live=dict(output=count,blockValues=blocks,blockIds=blocks)
    elif entry=='scanCarry':
        local,totals,blockids=_block_prefix(values,ids,product)
        with np.errstate(over='ignore',invalid='ignore'):scanned=reference.segmented_scan(totals,blockids,product=product)
        arrays.update(output=padded(local),blockIds=padded(blockids),scannedBlocks=padded(scanned));outputs=('output',);live=dict(output=count)
    elif entry=='scanExclusive':
        with np.errstate(over='ignore',invalid='ignore'):inclusive=reference.segmented_scan(values,ids,product=product)
        arrays.update(inclusiveValues=padded(inclusive),output=padded(seed));outputs=('output',);live=dict(output=count)
    else:raise ValueError('unknown scan stage')
    family='scan_uint' if dtype==np.uint32 else 'scan'
    return dict(kind=entry,operation='primitives.'+family+'.'+entry,params=dict(count=count,product=int(product)),arrays=arrays,outputs=outputs,live_counts=live,thread_count=[((count+255)//256)*256,1,1],label=f'{family}/{entry}/{count}/'+('product' if product else 'sum'))


def make_compact_case(entry,count,mode):
    i=np.arange(count,dtype=np.uint32)
    predicate=np.zeros(count,np.uint32) if mode=='none' else np.full(count,7,np.uint32) if mode=='all' else ((i%7)==0).astype(np.uint32)*3
    arrays=dict(predicate=padded(predicate));k=int(np.count_nonzero(predicate))
    if entry=='compactPrepare':
        arrays.update(flags=padded(np.full(count,0xdeadbeef,np.uint32)),segmentIds=padded(np.full(count,0xdeadbeef,np.uint32)),selectedCount=padded(np.array([0xdeadbeef],np.uint32)))
        outputs=('flags','segmentIds','selectedCount');live=dict(flags=count,segmentIds=count,selectedCount=1)
    elif entry=='compactScatter':
        prefix=reference.segmented_scan((predicate!=0).astype(np.uint32),np.zeros(count,np.uint32))
        arrays.update(payload=padded((count-i)%13),prefix=padded(prefix),compacted=padded(np.full(count,0xdeadbeef,np.uint32)),selectedCount=padded(np.array([0xdeadbeef],np.uint32)))
        outputs=('compacted','selectedCount');live=dict(compacted=k,selectedCount=1)
    else:raise ValueError('unknown compaction stage')
    return dict(kind=entry,operation='primitives.compact.'+entry,params=dict(count=count),arrays=arrays,outputs=outputs,live_counts=live,thread_count=[max(256,((count+255)//256)*256),1,1],label=f'compact/{entry}/{count}/{mode}')


def cases():
    for count in (255,256,257,70001,1025):
        for dtype in (np.float32,np.uint32):
            for product in (False,True):
                for entry in ('scanBlocks','scanCarry','scanExclusive'):yield make_scan_case(entry,dtype,count,product)
    for count in (257,70001):
        for mode in ('none','all','sparse'):
            for entry in ('compactPrepare','compactScatter'):yield make_compact_case(entry,count,mode)
    yield make_compact_case('compactPrepare',0,'none')


def reference_call(c):
    a=c['arrays'];p=c['params'];n=p['count'];kind=c['kind']
    result={k:a[k].copy() for k in c['outputs']}
    with np.errstate(over='ignore',invalid='ignore'):
        if kind=='scanBlocks':
            output,totals,ids=_block_prefix(a['values'][:n],a['segmentIds'][:n],bool(p['product']))
            result['output'][:n]=output;result['blockValues'][:len(totals)]=totals;result['blockIds'][:len(ids)]=ids
        elif kind=='scanCarry':
            index=np.arange(n);block=index//256
            keep=(block>0)
            keep[keep]&=a['segmentIds'][:n][keep]==a['blockIds'][block[keep]-1]
            prior=a['scannedBlocks'][block[keep]-1]
            result['output'][:n][keep]=prior*a['output'][:n][keep] if p['product'] else prior+a['output'][:n][keep]
        elif kind=='scanExclusive':
            result['output'][:n]=1 if p['product'] else 0
            keep=a['segmentIds'][1:n]==a['segmentIds'][:n-1]
            result['output'][1:n][keep]=a['inclusiveValues'][:n-1][keep]
        elif kind=='compactPrepare':
            result['flags'][:n]=(a['predicate'][:n]!=0).astype(np.uint32);result['segmentIds'][:n]=0;result['selectedCount'][0]=0
        elif kind=='compactScatter':
            compacted=reference.stable_compact(a['payload'][:n],a['predicate'][:n]);result['compacted'][:len(compacted)]=compacted
            result['selectedCount'][0]=len(compacted)
        else:raise ValueError('unknown primitive stage')
    return result


def workload_identity(c):
    layout=dict(operation=c['operation'],params=c['params'],thread_count=c['thread_count'],bindings=tuple(c['arrays']),outputs=c['outputs'],live_counts=c['live_counts'])
    h=hashlib.sha256(json.dumps(layout,sort_keys=True).encode())
    for name,a in sorted(c['arrays'].items()):h.update(json.dumps([name,a.dtype.str,a.shape]).encode()+b'\0'+a.tobytes()+b'\0')
    return dict(id='primitive-stage/'+c['label'],sha256=h.hexdigest(),description='Seeded stage inputs; stage CPU work includes oracle/output construction, not prerequisite GPU stages or optimized CPU throughput',**layout)


def check_parity(c,actual,expected):
    assert set(actual)==set(expected)==set(c['outputs'])
    for name in c['outputs']:
        a=actual[name];e=expected[name];n=c['live_counts'][name]
        assert a.shape==e.shape and a.dtype==e.dtype
        np.testing.assert_array_equal(a[n:],c['arrays'][name][n:]);np.testing.assert_array_equal(e[n:],c['arrays'][name][n:])
        if a.dtype==np.float32:np.testing.assert_allclose(a[:n],e[:n],rtol=2e-4,atol=2e-4,equal_nan=True)
        else:np.testing.assert_array_equal(a[:n],e[:n])
    if c['kind']=='scanCarry':
        n=c['params']['count'];idx=np.arange(n);block=idx//256;unchanged=block==0
        later=block>0;unchanged[later]=c['arrays']['segmentIds'][:n][later]!=c['arrays']['blockIds'][block[later]-1]
        np.testing.assert_array_equal(actual['output'][:n][unchanged],c['arrays']['output'][:n][unchanged])


def run(*,gpu=False,iterations=50,warmups=5,device_index=0,fixture_set="all"):
    from sintergpu import __version__
    from sintergpu.catalog import list_operations
    from sintergpu.timings import dependency_identities,summarize
    if type(iterations) is not int or not 1<=iterations<=100000 or type(warmups) is not int or not 1<=warmups<=10000 or type(device_index) is not int or device_index<0:raise ValueError('bounded positive sample/warmup counts and valid device index required')
    if fixture_set not in ("all","legacy","irregular"):raise ValueError("unknown fixture set")
    if gpu and os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in required after independent measurement clearance')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    env=dict(fixture_set=fixture_set,cpu=cpu_identity(),logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,platform=platform.platform(),python=platform.python_version(),numpy=importlib.metadata.version('numpy'),library_version=__version__,collector_revision=revision.stdout.strip() if revision.returncode==0 else None,captured_at_utc=datetime.now(timezone.utc).isoformat(),gpu=None,compiler_sha256=None)
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
            device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=info.luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
            limits=device.info.limits
            if limits.max_compute_threads_per_group<256 or limits.max_compute_thread_group_size.x<256 or int(device.info.timestamp_frequency)<=0:raise RuntimeError('256-thread groups and GPU timestamps required')
            adapter=SlangPyAdapter(device,allow_experimental=True)
            compiler=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
            if len(compiler)!=1:raise RuntimeError('identify exactly one bundled compiler')
            env.update(gpu=device.info.adapter_name,api=device.info.api_name,vendor_id=info.vendor_id,device_id=info.device_id,slangpy=importlib.metadata.version('slangpy'),compiler_sha256=hashlib.sha256(compiler[0].read_bytes()).hexdigest(),floating_point_mode='precise',timestamp_frequency=int(device.info.timestamp_frequency),buffer_budget_bytes=BUDGET)
            try:
                diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30);env['vulkaninfo_summary']=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
            except (OSError,subprocess.TimeoutExpired):env['vulkaninfo_summary']='unavailable: optional driver diagnostic missing or timed out'
        for case in cases():
            irregular=case["params"]["count"]==1025
            if fixture_set=="legacy" and irregular or fixture_set=="irregular" and not irregular:continue
            op=catalog[case['operation']];arrays=case['arrays'];outputs=case['outputs'];execute=lambda:reference_call(case)
            expected=execute();metrics=dict(cpu_reference_wall_ms=sample_cpu(execute,iterations=iterations,warmups=warmups))
            row={**dependencies,'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runner_helper_sha256':hashlib.sha256(Path(__file__).with_name('locus_timings.py').read_bytes()).hexdigest(),'workload':workload_identity(case),'environment':dict(env),'metrics':metrics,'iterations':iterations,'warmups':warmups,'scope':'Independent NumPy/serial per-stage reference work includes output construction, copies and temporary arrays. Upstream stage input preparation is outside both CPU/GPU timings. GPU per-entry timestamps exclude compilation, allocation and full output/inout reset; host submit/complete and complete bound-buffer uploads/full output readbacks are separate. Not a complete scan/compaction hierarchy, optimized CPU alternative, controlled performance comparison or optimized-variant selection.'}
            if gpu:
                if (case['thread_count'][0]+255)//256>int(limits.max_compute_dispatch_thread_groups.x) or any(a.nbytes>BUDGET for a in arrays.values()):raise ValueError('fixture exceeds dispatch or declared per-buffer budget')
                usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
                buffers={name:device.create_buffer(element_count=a.size,struct_size=a.dtype.itemsize,usage=usage,data=a) for name,a in arrays.items()}
                kernel=adapter.kernel(op['id']);bindings={**buffers,**case['params']};pool=device.create_query_pool(spy.QueryType.timestamp,2)
                row['adapter_source_identity']=adapter.source_identities[op['id']]
                def reset():
                    for name in outputs:buffers[name].copy_from_numpy(arrays[name])
                    device.wait()
                def dispatch(**kwargs):kernel.dispatch(thread_count=case['thread_count'],vars=bindings,**kwargs)
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
                row.update(uploaded_bytes_per_iteration=sum(a.nbytes for a in arrays.values()),readback_bytes_per_iteration=sum(arrays[k].nbytes for k in outputs),correctness='Per-stage independent oracle parity before/after isolated and transfer samples. Uint/modulo results, undefined-capacity seeds, block IDs, output tails and untouched carry regions exact; float values retain existing2e-4 tolerances.')
            for samples in metrics.values():summarize(samples)
            records.append(row)
        return dict(schema_version=1,environment=env,measurements=records,scope='Paired per-entry primitive stage fixtures; no full hierarchy/application/performance qualification')
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-set',choices=('all','legacy','irregular'),default='all')
    args=vars(parser.parse_args());output=args.pop('output');report=run(**args)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(report['measurements']),output=str(output))))
