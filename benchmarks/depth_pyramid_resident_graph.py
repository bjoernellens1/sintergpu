# SPDX-License-Identifier: Apache-2.0
"""Prepared VKL resident pyramid fixture; recording owns no device operations.

Kernel objects and their actual compiler/source binding are caller-provided;
source file checks here do not prove the identity of a compiled kernel.
No submission, synchronization with host, timing or qualification is implied.
"""
import hashlib
from pathlib import Path
import numpy as np
from sintergpu.vkl import depth_pyramid_plan, validate_capacities

ROOT=Path(__file__).resolve().parents[1]
PADDING=19
INIT_SOURCE='src/sintergpu/kernels/vkl/depth_resampling.slang'
ALLOWED_LEVEL_SOURCES={
    'src/sintergpu/kernels/vkl/depth_pyramid.slang',
    'benchmarks/shaders/depth_pyramid_vkl_candidate.slang',
    'benchmarks/shaders/depth_pyramid_vkl_stream_candidate.slang',
}


def check_sources(source_ids):
    if not isinstance(source_ids,dict) or INIT_SOURCE not in source_ids or len(set(source_ids)&ALLOWED_LEVEL_SOURCES)!=1:
        raise ValueError('initialization and one pyramid source required')
    if ('benchmarks/shaders/depth_pyramid_vkl_candidate.slang' in source_ids and
        'src/sintergpu/kernels/common/depth_pyramid_core.slang' not in source_ids):
        raise ValueError('candidate include source required')
    for name,digest in source_ids.items():
        path=(ROOT/name)
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('source identity mismatch: '+name)


class ResidentGraph:
    def __init__(self,device,kernels,inputs,*,usage,max_workgroups_x,source_ids,confidence=True):
        check_sources(source_ids)
        if set(kernels)!={'initializeDepth','depthPyramidLevel'} or any(k is None for k in kernels.values()):
            raise ValueError('prepared init and pyramid kernels required')
        if type(confidence) is not bool or set(inputs)!={'depth','valid','confidence'}:
            raise ValueError('depth, validity and confidence inputs required')
        arrays={k:np.asarray(v) for k,v in inputs.items()}
        shape=arrays['depth'].shape
        if len(shape)!=3 or any(x<1 for x in shape): raise ValueError('nonempty batch-height-width required')
        for name,dtype in [('depth',np.float32),('valid',np.uint32),('confidence',np.float32)]:
            a=arrays[name]
            if a.dtype!=dtype or a.shape!=shape or not a.flags.c_contiguous: raise ValueError('input shape, dtype or contiguity mismatch')
        self.plans=depth_pyramid_plan(*shape,max_workgroups_x=max_workgroups_x,confidence=confidence)
        self.source_ids=dict(source_ids); self.original={}; self.inputs={}; self.levels=[]; self.output_initial=[]; self._dispatches=[]
        def allocate(name,values):
            initial=np.full(values.size+PADDING,73,dtype=values.dtype); initial[:values.size]=values.ravel()
            buffer=device.create_buffer(element_count=initial.size,struct_size=4,usage=usage,data=initial)
            return buffer,initial
        for name,values in arrays.items():
            if name=='confidence' and not confidence: values=np.zeros(1,np.float32)
            self.inputs[name],self.original[name]=allocate(name,values)
        previous=self.inputs
        for plan in self.plans:
            outputs={}; snapshots={}
            for name in ('depthOut','validOut','confidenceOut'):
                dtype=np.uint32 if name=='validOut' else np.float32
                data=np.full(plan['elements'][name],73,dtype=dtype)
                outputs[name],snapshots[name]=allocate(name,data)
            bindings={'params':dict(plan['params']),**previous,**outputs}
            validate_capacities(plan,{k:v.size//4 for k,v in bindings.items() if k!='params'})
            self.levels.append(outputs)
            self.output_initial.append(snapshots)
            self._dispatches.append((kernels[plan['entrypoint']],plan['thread_count'],bindings,snapshots))
            previous={name:outputs[name+'Out'] for name in ('depth','valid','confidence')}

    def record(self,encoder):
        if encoder is None: raise ValueError('borrowed encoder required')
        for kernel,threads,bindings,_ in self._dispatches:
            encoder.global_barrier()
            kernel.dispatch(thread_count=threads,vars=bindings,command_encoder=encoder)

    def read_checked(self,validate):
        """Explicit readback boundary; validate(level, outputs) is called at every level."""
        if not callable(validate): raise ValueError('per-level validator required')
        check_sources(self.source_ids)
        for name,buffer in self.inputs.items():
            original=self.original[name]
            actual=buffer.to_numpy().view(original.dtype).reshape(-1)
            if actual.shape!=original.shape or not np.array_equal(actual.view(np.uint32),original.view(np.uint32)):
                raise AssertionError('input mutated: '+name)
        for plan,outputs,(_,_,_,snapshots) in zip(self.plans,self.levels,self._dispatches):
            result={}
            for name,buffer in outputs.items():
                before=snapshots[name]; a=buffer.to_numpy().view(before.dtype).reshape(-1)
                if a.shape!=before.shape or not np.array_equal(a[-PADDING:].view(np.uint32),before[-PADDING:].view(np.uint32)):
                    raise AssertionError('output tail changed: '+name)
                result[name]=a[:-PADDING].reshape((plan['shape'][0],*plan['shape'][-2:])).copy()
            validate(plan['level'],result)
