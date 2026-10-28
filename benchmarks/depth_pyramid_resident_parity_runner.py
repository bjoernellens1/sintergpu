# SPDX-License-Identifier: Apache-2.0
"""Review-required correctness-only VKL resident runner; no timing qualification.

Explicit global barriers differ from the incumbent adapter's recording path.
Device storage limits are captured from full vulkaninfo for the selected ICD.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import os
import re
import sys
import numpy as np
from benchmarks.depth_pyramid_resident_graph import ResidentGraph, ROOT, PADDING
from benchmarks.depth_pyramid_stream_pilot import lock_device, snapshot, verify_expected, write_once, committed_closure
from sintergpu.vkl import depth_pyramid_plan

INIT='src/sintergpu/kernels/vkl/depth_resampling.slang'
OLD='src/sintergpu/kernels/vkl/depth_pyramid.slang'
STREAM='benchmarks/shaders/depth_pyramid_vkl_stream_candidate.slang'
EXTRA=('benchmarks/depth_pyramid_resident_graph.py','benchmarks/depth_pyramid_resident_parity_runner.py', 'src/sintergpu/vkl.py', 'src/sintergpu/__init__.py', 'src/sintergpu/data/VERSION', INIT)


def closure(commit):
    result=committed_closure(commit)
    for name in EXTRA:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from committed tree: '+name)
        result[name]=hashlib.sha256(data).hexdigest()
    return result


def require_local_modules(modules,root=ROOT):
    root=Path(root).resolve()
    for name,module in modules.items():
        filename=getattr(module,'__file__',None)
        if not filename: raise ValueError('module file missing: '+name)
        path=Path(filename).resolve()
        if not path.is_relative_to(root): raise ValueError('module outside committed root: '+name)


def verify_loaded_closure(identities):
    names=('benchmarks.depth_pyramid_resident_parity_runner','benchmarks.depth_pyramid_resident_graph',
           'benchmarks.depth_pyramid_stream_pilot','benchmarks.depth_pyramid_isolated_pilot',
           'benchmarks.depth_pyramid_gpu_fixture','benchmarks.depth_pyramid_consolidation',
           'benchmarks.device_snapshot','test_depth_pyramid_gpu','sintergpu','sintergpu.vkl',
           'sintergpu.references.depth_pyramid_profiles')
    modules={name:sys.modules[name] for name in names if name in sys.modules}
    require_local_modules(modules)
    for name,module in modules.items():
        path=Path(module.__file__).resolve(); relative=str(path.relative_to(ROOT.resolve()))
        if relative not in identities or hashlib.sha256(path.read_bytes()).hexdigest()!=identities[relative]:
            raise ValueError('loaded module absent from committed closure: '+name)


def parse_storage_range(raw,expected_uuid):
    """Fail closed on ambiguous full vulkaninfo physical-device properties."""
    if not isinstance(raw,str) or not raw.strip(): raise ValueError('full vulkaninfo unavailable')
    gpu_ids=set(re.findall(r'^\s*GPU(\d+)\s*:',raw,re.MULTILINE))
    if gpu_ids != {'0'}: raise ValueError('exactly GPU0 required')
    uuids={v.lower() for v in re.findall(r'^\s*deviceUUID\s*=\s*([^\s]+)',raw,re.MULTILINE)}
    if uuids != {expected_uuid.lower()}: raise ValueError('full vulkaninfo device UUID mismatch')
    values=re.findall(r'^\s*maxStorageBufferRange\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*$',raw,re.MULTILINE)
    try: limits={int(value,0) for value in values}
    except ValueError as exc: raise ValueError('invalid storage range') from exc
    if len(limits)!=1 or next(iter(limits))<=0: raise ValueError('ambiguous or missing storage range')
    return next(iter(limits))


def full_vulkan_storage(expected_uuid):
    """Capture raw full output and its actual selected ICD environment."""
    env={k:os.environ[k] for k in ('VK_DRIVER_FILES','VK_ICD_FILENAMES') if os.environ.get(k)}
    if len(env)!=1: raise ValueError('exactly one explicit Vulkan ICD variable required')
    raw=subprocess.check_output(['vulkaninfo'],text=True,stderr=subprocess.PIPE)
    return {'raw':raw,'sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'icd_environment':env,'max_storage_buffer_range':parse_storage_range(raw,expected_uuid)}


def check_limits(limits,plans,*,max_fixture_bytes,max_storage_buffer_range):
    gx=int(limits.max_compute_dispatch_thread_groups.x); gy=int(limits.max_compute_dispatch_thread_groups.y); gz=int(limits.max_compute_dispatch_thread_groups.z)
    tx=int(limits.max_compute_thread_group_size.x); threads=int(limits.max_compute_threads_per_group)
    if min(gx,gy,gz)<1 or tx<256 or threads<256: raise ValueError('device dispatch/workgroup limit')
    if type(max_fixture_bytes) is not int or max_fixture_bytes<1: raise ValueError('fixture budget required')
    sizes=[]
    for plan in plans:
        n=plan['thread_count'][0]
        if (n+255)//256>gx or tuple(plan['thread_count'][1:])!=(1,1): raise ValueError('device dispatch limit')
        sizes.extend((int(size)+PADDING)*4 for size in plan['elements'].values())
    if type(max_storage_buffer_range) is not int or max_storage_buffer_range<=0: raise ValueError('verified storage range required')
    if max(sizes,default=0)>max_fixture_bytes: raise ValueError('per-buffer fixture budget exceeded')
    if max(sizes,default=0)>max_storage_buffer_range: raise ValueError('storage buffer device range exceeded')
    return {'max_groups_x':gx,'max_groups_y':gy,'max_groups_z':gz,'max_group_size_x':tx,'max_threads_per_group':threads,
            'max_fixture_buffer_bytes':max(sizes,default=0),'fixture_budget_bytes':max_fixture_bytes,
            'max_storage_buffer_range':max_storage_buffer_range}


def compare_graphs(device,old,new,validator,*,rounds=2):
    """Submit, wait and validate both complete graphs before resetting outputs."""
    if type(rounds) is not int or rounds<1: raise ValueError('positive rounds required')
    history=[]
    for iteration in range(rounds):
        results=[]
        for graph in (old,new):
            encoder=device.create_command_encoder(); graph.record(encoder)
            device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
            levels={}
            def check(level,outputs):
                validator(level,outputs); levels[level]={name:value.copy() for name,value in outputs.items()}
            graph.read_checked(check); results.append(levels)
        a,b=results
        if a.keys()!=b.keys(): raise AssertionError('level set differs')
        hashes={}
        for level in a:
            if a[level].keys()!=b[level].keys(): raise AssertionError('output set differs')
            hashes[str(level)]={}
            for name,x in a[level].items():
                y=b[level][name]
                if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x.view(np.uint32),y.view(np.uint32)):
                    raise AssertionError('paired output differs: '+str(level)+' '+name)
                hashes[str(level)][name]=hashlib.sha256(x.tobytes()).hexdigest()
        history.append(hashes)
        if iteration+1<rounds:
            for graph in (old,new):
                for buffers,initial in zip(graph.levels,graph.output_initial):
                    for name in buffers: buffers[name].copy_from_numpy(initial[name])
            device.wait_for_idle()
    return history


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    identities=closure(args.commit); output=Path(args.output)
    if output.exists(): raise FileExistsError('output exists')
    sys.path.insert(0,str(ROOT/'tests'))
    from test_depth_pyramid_gpu import compare
    from benchmarks.depth_pyramid_consolidation import cpu_reference, VKL_DONOR_EPSILON
    import benchmarks.device_snapshot
    import sintergpu.references.depth_pyramid_profiles
    verify_loaded_closure(identities)
    import slangpy as spy
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('compiler unavailable')
    compiler=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    with lock_device():
        verify_loaded_closure(identities)
        before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler)
        storage=full_vulkan_storage(before['physical_device_uuid'])
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id!=int(before['vendor_id'],0) or adapters[0].device_id!=int(before['device_id'],0): raise ValueError('adapter mismatch')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name!=before['adapter_name']: raise ValueError('device name mismatch')
            def load(path,entry):
                module=device.load_module(str(ROOT/path))
                return device.create_compute_kernel(device.link_program([module],[module.entry_point(entry)]))
            init=load(INIT,'initializeDepth'); old_kernel=load(OLD,'depthPyramidLevel'); stream=load(STREAM,'depthPyramidLevel')
            usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access; rows=[]
            for shape,confidence in [((2,5,7),True),((3,121,161),True),((1,1,1),True),((1,17,19),False)]:
                rng=np.random.default_rng(613); d=rng.uniform(.1,4,shape).astype(np.float32)
                v=(rng.random(shape)>.2).astype(np.uint32)*7; c=rng.random(shape,dtype=np.float32); d.flat[0]=np.nan
                inp={'depth':d,'valid':v,'confidence':c}
                plan=depth_pyramid_plan(*shape,max_workgroups_x=int(device.info.limits.max_compute_dispatch_thread_groups.x),confidence=confidence)
                observed=check_limits(device.info.limits,plan,max_fixture_bytes=32<<20,max_storage_buffer_range=storage['max_storage_buffer_range'])
                old=ResidentGraph(device,{'initializeDepth':init,'depthPyramidLevel':old_kernel},inp,usage=usage,max_workgroups_x=observed['max_groups_x'],source_ids={k:identities[k] for k in (INIT,OLD)},confidence=confidence)
                new=ResidentGraph(device,{'initializeDepth':init,'depthPyramidLevel':stream},inp,usage=usage,max_workgroups_x=observed['max_groups_x'],source_ids={k:identities[k] for k in (INIT,STREAM)},confidence=confidence)
                ref_inputs={'depth':d,'valid':v,'confidence':c if confidence else np.ones_like(c)}
                refs=cpu_reference(ref_inputs,profile='vkl',threshold=VKL_DONOR_EPSILON)['levels']
                reference={p['level']:dict(zip(('depthOut','validOut','confidenceOut'),arr)) for p,arr in zip(plan,refs)}
                history=compare_graphs(device,old,new,lambda level,result:compare(result,reference[level]))
                rows.append({'shape':shape,'confidence':confidence,'input_sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in inp.items()},'levels':[p['level'] for p in plan], 'device_limits':observed,'round_output_sha256':history})
                if closure(args.commit)!=identities: raise ValueError('source changed')
                verify_loaded_closure(identities)
            storage_after=full_vulkan_storage(before['physical_device_uuid'])
            if storage_after['icd_environment']!=storage['icd_environment'] or storage_after['max_storage_buffer_range']!=storage['max_storage_buffer_range']:
                raise ValueError('ICD or storage device limit changed')
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
            numpy_binary=Path(np._core._multiarray_umath.__file__).resolve()
            python_binary=Path(sys.executable).resolve()
            write_once(output,{'status':'correctness_only','commit':args.commit,'source_sha256':identities,'compiler_sha256':compiler,
                'runtime':{'python':{'version':sys.version,'executable':str(python_binary),'sha256':hashlib.sha256(python_binary.read_bytes()).hexdigest()},
                           'numpy':{'version':np.__version__,'binary':str(numpy_binary),'sha256':hashlib.sha256(numpy_binary.read_bytes()).hexdigest()},
                           'slangpy':{'version':importlib.metadata.version('slangpy'),'package':str(Path(spy.__file__).resolve()),'compiler':str(libs[0].resolve()),'compiler_sha256':compiler}},'snapshot_before':before,'snapshot_after':after,
                'storage_evidence_before':storage,'storage_evidence_after':storage_after,
                'completed_utc':datetime.now(timezone.utc).isoformat(),'cases':rows,'lock':'/tmp/sintergpu-gpu.lock',
                'scope':'Explicit barrier graph; no timing or performance qualification. Storage range comes from full Vulkan device output; 32 MiB is a separate fixture budget.'})
        finally: device.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true')
    for k in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','output'): p.add_argument('--'+k,required=True)
    run(p.parse_args())

if __name__=='__main__': main()
