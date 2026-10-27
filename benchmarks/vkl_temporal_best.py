# SPDX-License-Identifier: Apache-2.0
"""Standalone temporal evidence measurements; no controlled optimization claim."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
from vkl_geometry_depth import measure


def run(batches=65536,neighbors=32,iterations=30,warmups=5,pattern="dense"):
    if os.environ.get('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('explicit GPU opt-in required')
    if iterations<1 or warmups<1:raise ValueError('positive samples/warmups required')
    if pattern not in ('dense','sparse','early','late','nan'):raise ValueError('unknown evidence pattern')
    import numpy as np
    import slangpy as spy
    from sintergpu.adapters.slangpy import SlangPyAdapter
    from sintergpu.vkl import temporal_best_layout,validate_capacities
    from sintergpu import kernel_path
    device=spy.Device(type=spy.DeviceType.vulkan,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
    try:
        limit=int(device.info.limits.max_compute_dispatch_thread_groups[0])
        plan=temporal_best_layout(batches,neighbors,max_workgroups_x=limit,quality=True,confidence=True)
        if not batches:raise ValueError('measurement requires nonempty batches')
        op='vkl.temporal_best.temporalBest';adapter=SlangPyAdapter(device,allow_experimental=True);adapter.prepare(op)
        rng=np.random.default_rng(514);bindings={'params':plan['params']};uploads=[];readbacks=[];capacities={}
        usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access
        for name,size in plan['elements'].items():
            dtype=np.int32 if name=='best_indexOut' else np.uint32 if name=='equivalence' else np.float32
            if name.endswith('Out'):data=np.zeros(size,dtype)
            elif name=='equivalence':
                data=(rng.random(size)>(.95 if pattern=='sparse' else .3)).astype(dtype)
                if pattern in ('early','late','nan'):data[:]=1
            else:data=rng.random(size,dtype=dtype)
            if name=='pairwise_quality':
                if pattern=='early':data.reshape(batches,neighbors)[:,0]=2
                elif pattern=='late':data.reshape(batches,neighbors)[:,-1]=2
                elif pattern=='nan':data.reshape(batches,neighbors)[::17,-1]=np.nan
            buffer=device.create_buffer(element_count=size,struct_size=4,usage=usage,data=data)
            bindings[name]=buffer;capacities[name]=buffer.size//4
            if name.endswith('Out'):readbacks.append(buffer)
            else:uploads.append((buffer,data))
        validate_capacities(plan,capacities)
        def record(encoder):adapter.record(op,thread_count=plan['thread_count'],bindings=bindings,command_encoder=encoder)
        result=measure(device,record,uploads,readbacks,iterations,warmups)
        try:
            diagnostic=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=30)
            driver=diagnostic.stdout if diagnostic.returncode==0 else 'unavailable'
        except (OSError,subprocess.TimeoutExpired):driver='unavailable'
        return dict(status='measurement_only',adoption='unqualified',pattern=pattern,batches=batches,neighbors=neighbors,iterations=iterations,warmups=warmups,
                    device=device.info.adapter_name,api=device.info.api_name,driver_summary=driver,python=platform.python_version(),os=platform.platform(),
                    max_workgroups_x=limit,slangpy=importlib.metadata.version('slangpy'),floating_point_mode='precise',
                    compiler_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(spy.__file__).parent.glob('libslang-compiler*')},
                    source_identities=adapter.source_identities,operation_source_sha256={op:hashlib.sha256(kernel_path(op).read_bytes()).hexdigest()},
                    measurements=result,scope='Shared local GPU. One kernel is the complete GPU pipeline; full upload/readback and host costs separate. No controlled baseline, cross-vendor qualification or optimized selection.')
    finally:device.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('batches',65536),('neighbors',32),('iterations',30),('warmups',5)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--pattern',choices=['dense','sparse','early','late','nan'],default='dense')
    p.add_argument('--output',type=Path,required=True);args=vars(p.parse_args());output=args.pop('output')
    result=run(**args);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
