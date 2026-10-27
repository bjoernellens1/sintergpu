# SPDX-License-Identifier: Apache-2.0
"""Explicit single-device measurements, not optimization/adoption qualification.

SINTERGPU_GPU_TESTS=1 PYTHONPATH=src python benchmarks/vkl_geometry_depth.py \
    --height 480 --width 640 --sources 4 --trust mixed --iterations 30 --output results.json

Preallocate and warm up. Retain GPU-resident and full upload/submit/readback samples
separately; this standalone harness owns test resources. Production owns none here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
import time


def measure(device,record,uploads,readbacks,iterations,warmups):
    import numpy as np
    import slangpy as spy
    for _ in range(warmups):
        encoder=device.create_command_encoder();record(encoder)
        device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    pool=device.create_query_pool(spy.QueryType.timestamp,2*iterations)
    start=time.perf_counter_ns();encoder=device.create_command_encoder()
    for n in range(iterations):
        encoder.write_timestamp(pool,2*n);record(encoder);encoder.write_timestamp(pool,2*n+1)
    device.submit_command_buffer(encoder.finish());device.wait_for_idle()
    host_ms=(time.perf_counter_ns()-start)/1e6/iterations
    ticks=pool.get_results(0,2*iterations)
    resident=[(int(ticks[2*n+1])-int(ticks[2*n]))*1000/int(device.info.timestamp_frequency) for n in range(iterations)]
    if not all(t>0 for t in resident):raise RuntimeError('non-increasing GPU timestamps')
    transfers=[]
    for _ in range(iterations):
        start=time.perf_counter_ns()
        for buffer,data in uploads:buffer.copy_from_numpy(data)
        encoder=device.create_command_encoder();record(encoder)
        device.submit_command_buffer(encoder.finish());device.wait_for_idle()
        for buffer in readbacks:buffer.to_numpy()
        transfers.append((time.perf_counter_ns()-start)/1e6)
    def stats(samples):return dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples)
    return dict(resident=stats(resident),transfer_inclusive=stats(transfers),
                host_record_submit_wait_ms_per_iteration=host_ms,
                uploaded_bytes_per_iteration=sum(a.nbytes for _,a in uploads),
                readback_bytes_per_iteration=sum(b.size for b in readbacks))


def run(height=480,width=640,sources=4,trust='none',distribution='consistent',iterations=30,warmups=5,world=False):
    if os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit SINTERGPU_GPU_TESTS=1 required')
    if type(iterations) is not int or iterations<1 or warmups<1:raise ValueError('positive iteration/warmup counts required')
    if trust not in ('none','all','mixed') or distribution not in ('consistent','conflict','invalid'):raise ValueError('unknown distribution')
    import numpy as np
    import slangpy as spy
    from sintergpu.adapters.slangpy import SlangPyAdapter
    from sintergpu.vkl import geometry_layout,fusion_layout,validate_capacities
    device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    limit=int(device.info.limits.max_compute_dispatch_thread_groups[0])
    layouts={
        'geometry':geometry_layout(1,height,width,world=world,confidence=True,max_workgroups_x=limit),
        'world_transform':geometry_layout(1,height,width,operation='world_transform',max_workgroups_x=limit),
        'fuse_depth':fusion_layout(1,height,width,sources,max_workgroups_x=limit),
        'temporal':fusion_layout(1,height,width,sources,temporal=True,max_workgroups_x=limit),
    }
    ids={'geometry':'vkl.geometry.depthGeometry','world_transform':'vkl.geometry.worldTransform',
         'fuse_depth':'vkl.depth_fusion.fuseDepth','temporal':'vkl.depth_fusion.temporalDepthFuse'}
    adapter=SlangPyAdapter(device,allow_experimental=True)
    usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
    rng=np.random.default_rng(914);states={};pixels=height*width
    for name,layout in layouts.items():
        adapter.prepare(ids[name]);buffers={};uploads=[];readbacks=[]
        for binding,size in layout['elements'].items():
            if binding.endswith('Out'):data=np.zeros(size,np.uint32 if binding=='validOut' else np.int32 if 'source' in binding or 'count' in binding or 'contributor' in binding else np.float32)
            elif binding in ('valids','visibility','consistency','trusted'):
                data=np.ones(size,np.uint32)
                if binding=='valids' and distribution=='invalid':data[:]=0
                if binding=='trusted':
                    if trust=='none':data[:]=0
                    elif trust=='mixed':data[:]=(np.arange(size)%7==0)
            elif binding=='intrinsics':data=np.array([420,0,width/2,0,415,height/2,0,0,1],np.float32)
            elif binding=='pose':data=np.eye(4,dtype=np.float32).reshape(-1) if size==16 else np.zeros(size,np.float32)
            elif binding=='xyz':data=rng.normal(size=size).astype(np.float32)
            elif binding in ('confidence','confidences'):data=rng.uniform(.5,1,size).astype(np.float32)
            elif binding=='depths':
                data=rng.uniform(.99,1.01,size).astype(np.float32)
                if distribution=='conflict':data*=np.repeat(np.arange(1,sources+1,dtype=np.float32),pixels)
            else:data=np.ones(size,np.float32)
            buffers[binding]=device.create_buffer(element_count=size,struct_size=4,usage=usage,data=data)
            if binding.endswith('Out'):readbacks.append(buffers[binding])
            else:uploads.append((buffers[binding],data))
        validate_capacities(layout,{k:b.size//4 for k,b in buffers.items()})
        states[name]=dict(layout=layout,buffers=buffers,uploads=uploads,readbacks=readbacks)
    records={}
    for name,state in states.items():
        def record(encoder,name=name,state=state):
            adapter.record(ids[name],thread_count=state['layout']['thread_count'],bindings=dict(params=state['layout']['params'],**state['buffers']),command_encoder=encoder)
        records[name]=measure(device,record,state['uploads'],state['readbacks'],iterations,warmups)
    fused=states['fuse_depth'];geometry=states['geometry']
    geo_bindings={**geometry['buffers'],'depth':fused['buffers']['depthOut'],'confidence':fused['buffers']['confidenceOut']}
    def pipeline(encoder):
        adapter.record(ids['fuse_depth'],thread_count=fused['layout']['thread_count'],bindings=dict(params=fused['layout']['params'],**fused['buffers']),command_encoder=encoder)
        adapter.record(ids['geometry'],thread_count=geometry['layout']['thread_count'],bindings=dict(params=geometry['layout']['params'],**geo_bindings),command_encoder=encoder)
    uploads=fused['uploads']+[(b,d) for b,d in geometry['uploads'] if b not in (geometry['buffers']['depth'],geometry['buffers']['confidence'])]
    records['pipeline_fuse_then_geometry']=measure(device,pipeline,uploads,fused['readbacks']+geometry['readbacks'],iterations,warmups)
    libraries=sorted(Path(spy.__file__).parent.glob('libslang-compiler*'))
    try:
        diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
        driver_summary=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable: vulkaninfo failed'
    except (OSError,subprocess.TimeoutExpired):driver_summary='unavailable: record driver independently'
    from sintergpu import kernel_path
    return dict(status='measurement_only',adoption='unqualified',height=height,width=width,sources=sources,trust=trust,distribution=distribution,
                world=world,iterations=iterations,warmups=warmups,device=device.info.adapter_name,api=device.info.api_name,
                driver_summary=driver_summary,python=platform.python_version(),os=platform.platform(),
                max_workgroups_x=limit,slangpy=importlib.metadata.version('slangpy'),floating_point_mode='precise',
                compiler_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
                source_identities=adapter.source_identities,operation_source_sha256={op:hashlib.sha256(kernel_path(op).read_bytes()).hexdigest() for op in ids.values()},measurements=records,
                scope='Shared local GPU, no controlled baseline, no speedup selection or cross-vendor qualification. Host measurement amortizes resident recording/submission/wait; transfer-inclusive samples include all explicit input uploads and output readbacks.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg,default in [('height',480),('width',640),('sources',4),('iterations',30),('warmups',5)]:p.add_argument('--'+arg,type=int,default=default)
    p.add_argument('--trust',choices=['none','all','mixed'],default='none')
    p.add_argument('--distribution',choices=['consistent','conflict','invalid'],default='consistent')
    p.add_argument('--world',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output;kwargs=vars(a);del kwargs['output']
    result=run(**kwargs);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))
