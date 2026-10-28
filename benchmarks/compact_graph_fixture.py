# SPDX-License-Identifier: Apache-2.0
"""Borrowed-resource recording fixture for experimental compact stages.

Caller supplies prepared kernels, accurate allocation identity/ranges, and
uploads and independently verifies shape_strides_u32 contents before recording. No compilation,
allocation, upload, submission, host synchronization or readback occurs here.
Metadata cannot prove backing-memory identity, uploaded contents, or kernel
source; an adapter must independently verify those claims before GPU execution.
"""


class PreparedCompactGraph:
    def __init__(self,plan,buffers,ranges,kernels):
        if plan.get('status')!='design_only_no_gpu_implementation': raise ValueError('unexpected plan')
        resources=plan['resources']
        if set(buffers)!=set(resources) or set(ranges)!=set(resources): raise ValueError('complete resource set required')
        for name,desc in resources.items():
            region=ranges[name]
            if set(region)!={'allocation_id','offset','length'} or not isinstance(region['allocation_id'],str) or not region['allocation_id']:
                raise ValueError('allocation identity and range required')
            if type(region['offset']) is not int or region['offset']!=0:
                raise ValueError('nonzero offset unsupported for full-buffer binding')
            if type(region['length']) is not int or region['length']<desc['allocated_bytes']:
                raise ValueError('buffer range size invalid')
            if type(buffers[name].size) is not int or buffers[name].size<desc['allocated_bytes']:
                raise ValueError('buffer size invalid')
            if region['length']!=buffers[name].size:
                raise ValueError('range must cover the full buffer')
            expected_stride=8 if name=='shape_strides' else 4
            if getattr(buffers[name],'struct_size',None)!=expected_stride:
                raise ValueError('buffer struct stride invalid: '+name)
        names=list(resources)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                x,y=ranges[a],ranges[b]
                writable=resources[a]['access']!='read' or resources[b]['access']!='read'
                same=x['allocation_id']==y['allocation_id']
                if buffers[a] is buffers[b] and not same:
                    raise ValueError('same buffer has inconsistent allocation identity')
                if buffers[a] is buffers[b] and writable:
                    raise ValueError('writable shared buffer forbidden')
                overlap=x['offset']<y['offset']+y['length'] and y['offset']<x['offset']+x['length']
                if writable and same and overlap: raise ValueError('writable resource overlap')
        dispatches=[]
        for stage in plan['stages']:
            key=(stage['source'],stage['entry'])
            if key not in kernels: raise ValueError('missing prepared kernel')
            bindings={name:buffers[label] for name,label in stage['bindings'].items()}
            if stage['entry'] in ('scanBlocks','scanCarry'):
                bindings.update(stage['params'])
            else:
                bindings['params']=dict(stage['params'])
            dispatches.append((kernels[key],stage['thread_count'],bindings))
        self._dispatches=tuple(dispatches)

    def record(self,encoder):
        if encoder is None: raise ValueError('borrowed encoder required')
        for kernel,threads,bindings in self._dispatches:
            encoder.global_barrier()
            kernel.dispatch(thread_count=threads,vars=bindings,command_encoder=encoder)
