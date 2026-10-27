# SPDX-License-Identifier: Apache-2.0
"""Standalone resident/host/transfer measurements; no speedup qualification."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from vkl_geometry_depth import measure


def run(operation='densify',height=480,width=640,steps=2,pattern='holes',iterations=30,warmups=5,scale=4):
    if os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in required')
    if iterations<1 or warmups<1:raise ValueError('positive samples required')
    if operation not in ('densify','lift','pyramid','downsample') or pattern not in ('holes','sparse','edges','dense','invalid'):raise ValueError('unknown profile')
    import numpy as np
    import slangpy as spy
    from sintergpu.vkl import depth_lift_layout,depth_densify_plan,depth_pyramid_plan,depth_downsample_plan,validate_capacities
    from sintergpu.adapters.slangpy import SlangPyAdapter
    from sintergpu import kernel_path
    device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        limit=int(device.info.limits.max_compute_dispatch_thread_groups[0]);rng=np.random.default_rng(613)
        if operation=='densify':plans=depth_densify_plan(1,height,width,steps=steps,max_workgroups_x=limit)
        elif operation=='lift':plans=[depth_lift_layout(1,height,width,2*height,2*width,max_workgroups_x=limit)]
        elif operation=='downsample':
            graph=depth_downsample_plan(1,height,width,scale=scale,max_workgroups_x=limit);plans=graph['stages']
        else:plans=depth_pyramid_plan(1,height,width,max_workgroups_x=limit)
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        depth=rng.uniform(.99,1.01,height*width).astype(np.float32)
        valid=(rng.random(depth.size)>(.9 if pattern=='sparse' else .2)).astype(np.uint32)
        if pattern=='dense':valid[:]=1
        elif pattern=='invalid':valid[:]=0
        rgb=np.zeros((1,3,height,width),np.float32)
        if pattern=='edges':depth.reshape(height,width)[:,width//2:]=2;rgb[:,:,:,width//2:]=1
        arrays=dict(depth=depth,valid=valid,confidence=rng.uniform(.5,1,depth.size).astype(np.float32))
        if operation=='densify':arrays.update(rgb=rgb.reshape(-1),direct_valid=np.zeros(1,np.uint32))
        uploads=[];initial={}
        for name,a in arrays.items():
            b=device.create_buffer(element_count=a.size,struct_size=4,usage=usage,data=a);initial[name]=b;uploads.append((b,a))
        adapter=SlangPyAdapter(device,allow_experimental=True);stages=[];previous=initial
        # Allocate outside recording. Densify alternates exactly two output sets.
        pingpong=[]
        for n,plan in enumerate(plans):
            op='vkl.depth_pyramid.depthPyramidLevel' if plan['entrypoint']=='depthPyramidLevel' else 'vkl.depth_resampling.'+plan['entrypoint']
            adapter.prepare(op);outputs={}
            if operation=='densify' and n>=2:outputs=pingpong[n%2]
            else:
                for name,size in plan['elements'].items():
                    if not name.endswith('Out'):continue
                    dtype=np.float32 if name in ('depthOut','confidenceOut') else np.int32 if name=='supportOut' else np.uint32
                    outputs[name]=device.create_buffer(element_count=size,struct_size=4,usage=usage,data=np.zeros(size,dtype))
                if operation=='densify':pingpong.append(outputs)
            inputs=previous if plan['entrypoint']!='labelDirectDepth' else {'valid':previous['valid']}
            bindings={'params':plan['params'],**inputs,**outputs}
            if operation=='densify':bindings['rgb']=initial['rgb']
            validate_capacities(plan,{k:b.size//4 for k,b in bindings.items() if k!='params'})
            stages.append((op,plan,bindings,outputs))
            previous={k.removesuffix('Out'):b for k,b in outputs.items() if k in ('depthOut','validOut','confidenceOut','direct_validOut')}
        def record(encoder):
            for op,plan,bindings,_ in stages:adapter.record(op,thread_count=plan['thread_count'],bindings=bindings,command_encoder=encoder)
        readbacks=[b for *_,out in stages for b in out.values()] if operation=='pyramid' else list(stages[-1][3].values())
        if operation=='downsample':readbacks=[stages[ref['stage']][3][ref['binding']] for ref in graph['outputs'].values()]
        pipeline=measure(device,record,uploads,readbacks,iterations,warmups)
        # Full pipeline is warm before timing isolated passes over initialized inputs.
        individual=[]
        for op,plan,bindings,out in stages:
            def single(encoder,op=op,plan=plan,bindings=bindings):adapter.record(op,thread_count=plan['thread_count'],bindings=bindings,command_encoder=encoder)
            individual.append(dict(operation=op,measurement=measure(device,single,[],[],iterations,warmups)))
        try:
            diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30);driver=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
        except (OSError,subprocess.TimeoutExpired):driver='unavailable'
        return dict(status='measurement_only',adoption='unqualified',operation=operation,height=height,width=width,steps=steps,scale=scale,pattern=pattern,iterations=iterations,warmups=warmups,
                    device=device.info.adapter_name,api=device.info.api_name,driver_summary=driver,python=platform.python_version(),os=platform.platform(),max_workgroups_x=limit,
                    slangpy=importlib.metadata.version('slangpy'),floating_point_mode='precise',
                    compiler_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(spy.__file__).parent.glob('libslang-compiler*')},
                    source_identities=adapter.source_identities,operation_source_sha256={op:hashlib.sha256(kernel_path(op).read_bytes()).hexdigest() for op,_,_,_ in stages},
                    setup_source_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('src/sintergpu/vkl.py').read_bytes()).hexdigest(),
                    pipeline=pipeline,kernels=individual,scope='Shared local device; no controlled baseline/speedup or application qualification. Resident stages use one borrowed encoder and explicit caller-owned buffers, no intermediate readback. Isolated-pass transfer lists empty; only pipeline measurements include all input uploads and final output readbacks (all levels for full pyramid).')
    finally:device.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operation',choices=['densify','lift','pyramid','downsample'],default='densify');p.add_argument('--pattern',choices=['holes','sparse','edges','dense','invalid'],default='holes')
    for name,default in [('height',480),('width',640),('steps',2),('iterations',30),('warmups',5)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--scale',type=int,choices=[2,4,8],default=4)
    p.add_argument('--output',type=Path,required=True);args=vars(p.parse_args());output=args.pop('output');r=run(**args);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(r,indent=2)+'\n')
