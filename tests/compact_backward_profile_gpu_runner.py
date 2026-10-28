# SPDX-License-Identifier: Apache-2.0
"""Review-required opt-in forward/backward correctness probe, no GPU qualification."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import numpy as np
from compact_gpu_runner import (ROOT, closure as base_closure, verify_modules as verify_base,
    owned_lease, snapshot, verify_expected, full_vulkan_storage, write_once, case_for_limits)
from compact_gpu_runtime import GPUFixture
from compact_backward_profile_gpu_fixture import ProfileFixture, profile_gradient, torch_gradient
from compact_backward_gpu_runner import closure as backward_closure, verify_modules as verify_backward
from compact_gpu_support import make_case, initial_state, model_stage, verify_final
from sintergpu.compact_plan import plan_compact
from sintergpu.references.compact_numeric import select_compact
from benchmarks.depth_pyramid_resident_parity_runner import require_local_modules

EXTRA=('tests/compact_backward_profile_gpu_runner.py','tests/compact_backward_profile_gpu_fixture.py',
       'tests/test_compact_backward_profile_gpu_fixture.py','src/sintergpu/compact_backward_torch_cpu_profile.py',
       'benchmarks/shaders/vkl_compact_backward_torch_cpu_profile.slang')
PROFILE_COMMIT='e0c10ff371995ea13266e5a43b9730d92342fecb'
CASES=((0,3,2,'float16'),(1,0,4,'float32'),(255,3,4,'float32'),(256,5,8,'float64'),
       (257,4,8,'complex64'),(17,3,16,'complex128'),(7,5,2,'float16'))
DTYPES={'float16':np.float16,'float32':np.float32,'float64':np.float64,'complex64':np.complex64,'complex128':np.complex128}


def closure(commit):
    ids=backward_closure(commit)
    for name in EXTRA:
        path=ROOT/name
        if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from committed tree: '+name)
        ids[name]=hashlib.sha256(data).hexdigest()
    name='benchmarks/shaders/vkl_compact_backward_torch_cpu_profile.slang'
    if (ROOT/name).read_bytes()!=subprocess.check_output(['git','-C',str(ROOT),'show',PROFILE_COMMIT+':'+name]): raise ValueError('profile shader differs from approved candidate')
    return ids


def verify_modules(ids):
    verify_backward(ids)
    names=('compact_backward_profile_gpu_fixture','sintergpu.compact_backward_torch_cpu_profile')
    mods={name:sys.modules[name] for name in names if name in sys.modules}; mods['runner']=sys.modules[__name__]
    require_local_modules(mods)
    for mod in mods.values():
        path=Path(mod.__file__).resolve(); name=str(path.relative_to(ROOT.resolve()))
        if ids.get(name)!=hashlib.sha256(path.read_bytes()).hexdigest(): raise ValueError('loaded module differs from closure: '+name)


def make_upstream(case,dtype):
    b,_,tail=case['shape']; cap=case['capacity']; typ=DTYPES[dtype]
    result=np.arange(b*cap*tail,dtype=np.float32).astype(typ).reshape(b,cap,tail)
    component={'float16':2,'float32':4,'float64':8,'complex64':4,'complex128':8}[dtype]
    patterns={2:[0x8000,0x7c01,0xfc01,0x7e55,0x7c00,0xfc00],
              4:[0x80000000,0x7f800001,0xff800001,0x7fc12345,0x7f800000,0xff800000],
              8:[0x8000000000000000,0x7ff0000000000001,0xfff0000000000001,0x7ff8123456789abc,0x7ff0000000000000,0xfff0000000000000]}[component]
    raw=result.view(np.uint8).reshape(-1)
    for i,bits in enumerate(patterns[:len(raw)//component]): raw[i*component:(i+1)*component]=np.frombuffer(bits.to_bytes(component,'little'),np.uint8)
    return result



def odd_fp16_case(*,max_groups=65535,storage_limit=32<<20,max_threads=256,max_group_size=256):
    """One batch, three elements, one FP16 tail: six output bytes."""
    values=np.arange(8,dtype=np.uint8)*np.uint8(17)
    mask=np.array([0,1,1,1],np.uint8)
    plan=plan_compact((1,3,1),capacity=3,item_bytes=2,value_strides=(6,2,2),value_base_byte=1,value_storage_bytes=8,
        mask_strides=(3,1),mask_base_byte=1,mask_storage_bytes=4,
        max_workgroups_x=max_groups,max_buffer_bytes=min(storage_limit,32<<20)-32,
        max_threads_per_group=max_threads,max_group_size_x=max_group_size)
    return {'plan':plan,'values_bytes':values,'mask_bytes':mask,'action_bytes':np.zeros(0,np.uint8),
            'shape':(1,3,1),'width':2,'capacity':3,'action':0}


def odd_forward_reference(case):
    values=np.ndarray((1,3,1),dtype='V2',buffer=case['values_bytes'],offset=1,strides=(6,2,2))
    mask=np.ndarray((1,3),dtype=bool,buffer=case['mask_bytes'],offset=1,strides=(3,1))
    return select_compact(values,mask,capacity=3)


class OddForwardFixture(GPUFixture):
    def run_graph(self):
        encoder=self.device.create_command_encoder(); self.graph.record(encoder)
        self.device.submit_command_buffer(encoder.finish()); self.device.wait_for_idle()
        expected={name:data.copy() for name,data in self.initial.items()}
        reference=odd_forward_reference(self.case)
        for stage in self.case['plan']['stages']:
            if stage['entry']=='compactGatherWords':
                raw=reference['values'].tobytes(); output=expected['values_out'].view(np.uint8)
                output[:((len(raw)+3)//4)*4]=0; output[:len(raw)]=np.frombuffer(raw,np.uint8)
            else: model_stage(self.case,expected,stage)
        actual=self.compare_all(expected); verify_final(self.case,actual,reference)
        return actual


def preflight_oracles():
    for n,cap,width,dtype in CASES:
        case=make_case(n,cap,width,0); upstream=make_upstream(case,dtype)
        np.testing.assert_array_equal(profile_gradient(case,upstream,dtype),torch_gradient(case,upstream,dtype),err_msg=f'profile donor mismatch: {dtype}')
    case=odd_fp16_case(); upstream=make_upstream(case,'float16')
    np.testing.assert_array_equal(profile_gradient(case,upstream,'float16'),torch_gradient(case,upstream,'float16'))


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    if Path(args.output).exists(): raise FileExistsError(args.output)
    ids=closure(args.commit); verify_modules(ids)
    import torch
    preflight_oracles()
    import slangpy as spy
    libs=list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(libs)!=1 or not libs[0].is_file(): raise ValueError('compiler unavailable')
    compiler=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    with owned_lease() as lease:
        lease.verify(); verify_modules(ids)
        before=snapshot(); verify_expected(before,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,compiler)
        storage=full_vulkan_storage(before['physical_device_uuid']); max_storage=storage['max_storage_buffer_range']
        adapters=spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if len(adapters)!=1 or adapters[0].vendor_id==0x10005: raise ValueError('exactly one hardware adapter required')
        if adapters[0].vendor_id!=int(before['vendor_id'],0) or adapters[0].device_id!=int(before['device_id'],0): raise ValueError('adapter mismatch')
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapters[0].luid,compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        try:
            if device.info.adapter_name!=before['adapter_name']: raise ValueError('adapter name mismatch')
            limits=device.info.limits; rows=[]
            for n,cap,width,dtype in CASES:
                lease.verify(); case=case_for_limits((n,cap,width,0),limits,max_storage)
                forward=GPUFixture(device,spy,case); upstream=make_upstream(case,dtype)
                backward=ProfileFixture(device,spy,forward,upstream,dtype,max_groups=int(limits.max_compute_dispatch_thread_groups.x),storage_limit=max_storage,max_threads=int(limits.max_compute_threads_per_group),max_group_size=int(limits.max_compute_thread_group_size.x))
                hashes=[]
                for iteration in range(2):
                    if iteration: forward.reset(); backward.reset()
                    state=forward.run_graph(); hashes.append(backward.run(state,dtype))
                rows.append({'case':[n,cap,width,dtype],'forward_buffer_handles':forward.handle_identity,'backward_buffer_handles':backward.handles,
                    'upstream_sha256':hashlib.sha256(upstream.tobytes()).hexdigest(),'output_sha256':hashes})
                if closure(args.commit)!=ids: raise ValueError('source changed')
                verify_modules(ids)
            lease.verify()
            case=odd_fp16_case(max_groups=int(limits.max_compute_dispatch_thread_groups.x),storage_limit=max_storage,
                max_threads=int(limits.max_compute_threads_per_group),max_group_size=int(limits.max_compute_thread_group_size.x))
            forward=OddForwardFixture(device,spy,case); upstream=make_upstream(case,'float16')
            backward=ProfileFixture(device,spy,forward,upstream,'float16',max_groups=int(limits.max_compute_dispatch_thread_groups.x),
                storage_limit=max_storage,max_threads=int(limits.max_compute_threads_per_group),max_group_size=int(limits.max_compute_thread_group_size.x))
            hashes=[]
            for iteration in range(2):
                if iteration: forward.reset(); backward.reset()
                state=forward.run_graph(); hashes.append(backward.run(state,'float16'))
                raw=backward.buffers['outputWords'].to_numpy().view(np.uint8).reshape(-1)
                if bytes(raw[6:8])!=b'\x00\x00': raise AssertionError('odd FP16 padding was not zero')
                np.testing.assert_array_equal(raw[8:],backward.initial['outputWords'].view(np.uint8)[8:],err_msg='odd FP16 output guard')
            rows.append({'case':[1,3,1,3,'float16'],'forward_buffer_handles':forward.handle_identity,
                'backward_buffer_handles':backward.handles,'upstream_sha256':hashlib.sha256(upstream.tobytes()).hexdigest(),
                'output_sha256':hashes,'padding_bytes':2})
            if closure(args.commit)!=ids: raise ValueError('source changed')
            verify_modules(ids)
            after=snapshot(); verify_expected(after,args.gpu_uuid,args.driver_version,args.driver_info,args.compiler_sha256,hashlib.sha256(libs[0].read_bytes()).hexdigest())
            storage_after=full_vulkan_storage(before['physical_device_uuid'])
            if storage_after['icd_environment']!=storage['icd_environment'] or storage_after['max_storage_buffer_range']!=max_storage: raise ValueError('ICD or storage changed')
            lease.verify(); write_once(args.output,{'status':'correctness_probe_only','commit':args.commit,'source_sha256':ids,
                'snapshot_before':before,'snapshot_after':after,'storage_before':storage,'storage_after':storage_after,
                'profile_shader_origin_commit':PROFILE_COMMIT,'compiler_sha256':compiler,'slangpy_version':importlib.metadata.version('slangpy'),'torch_version':torch.__version__,
                'numpy_version':np.__version__,'python':sys.version,'cases':rows,
                'scope':'Actual forward slot buffer, fixture readbacks between passes; no production host roundtrip implied. VkBuffer uniqueness does not prove physical memory disjointness. CPU-observed Torch bit profile only; no Torch GPU parity, view accumulation, framework interop, GPU qualification or performance claim.'})
        finally: device.close()
        lease.verify()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true')
    for field in ('commit','gpu-uuid','driver-version','driver-info','compiler-sha256','output'): p.add_argument('--'+field,required=True)
    run(p.parse_args())

if __name__=='__main__': main()
