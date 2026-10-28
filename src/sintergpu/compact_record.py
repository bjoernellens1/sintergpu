# SPDX-License-Identifier: Apache-2.0
"""Experimental caller-owned stable compact graph recording.

Preparation can compile via a supplied SlangPyAdapter; recording performs only
explicit barriers and dispatch on the caller's encoder. Adapter identity checks read local files during preparation and optional
explicit verification, never during recording. It never allocates GPU resources,
uploads, submits, reads back or waits. It cannot verify caller-provided allocation
identities, actual aliasing, descriptor bytes, masks, queue/device compatibility,
or buffer lifetime; callers must verify these and keep them valid and immutable
until execution completes. No framework autograd integration is provided.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from .compact_plan import plan_compact
from .catalog import get_operation
from . import kernel_path
from .adapters.slangpy import SlangPyAdapter

FORWARD='vkl.compact_batched.'
SCAN='primitives.scan_uint.'
FORWARD_ENTRIES=frozenset(('compactInit','compactFlags','compactScatterMetadata','compactFinalizeCounts','compactGatherWords'))
SCAN_ENTRIES=frozenset(('scanBlocks','scanCarry'))


def _rebuild(plan,limits):
    if set(limits)!={'max_workgroups_x','max_buffer_bytes','max_threads_per_group','max_group_size_x'}:
        raise ValueError('actual device limits required')
    values=plan['inputs']['values']; mask=plan['inputs']['mask']; action=plan['inputs']['actions']
    return plan_compact(values['shape'],capacity=plan['capacity'],item_bytes=values['item_bytes'],
        value_strides=values['strides_bytes'],value_base_byte=values['base_byte'],value_storage_bytes=values['storage_bytes'],
        mask_strides=mask['strides_bytes'],mask_base_byte=mask['base_byte'],mask_storage_bytes=mask['storage_bytes'],
        action_item_bytes=0 if action is None else action['item_bytes'],action_strides=None if action is None else action['strides_bytes'],
        action_base_byte=0 if action is None else action['base_byte'],action_storage_bytes=0 if action is None else action['storage_bytes'],**limits)


def _operation(stage):
    entry=stage['entry']; source=stage['source']
    if entry in FORWARD_ENTRIES and source in ('benchmarks/shaders/vkl_compact_candidate.slang','src/sintergpu/kernels/vkl/compact_batched.slang'):
        return FORWARD+entry
    if entry in SCAN_ENTRIES and source=='src/sintergpu/kernels/primitives/scan_uint.slang' and stage['specialization']=='uint32_segmented_inclusive_sum':
        return SCAN+entry
    raise ValueError('unrecognized compact stage')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_hashes():
    """Hash installed compiler, SGL library and loaded Python extension files.

    This identifies files in the loaded package; it does not attest dynamic
    linker mappings or device compilation options.
    """
    import slangpy
    root=Path(slangpy.__file__).resolve().parent
    libs=list(root.glob('libslang-compiler*'))
    extension=sys.modules.get('slangpy.slangpy_ext')
    runtime=root/'libsgl.so'
    if len(libs)!=1 or not libs[0].is_file() or not runtime.is_file() or extension is None:
        raise ValueError('SlangPy runtime identity unavailable')
    ext=Path(extension.__file__).resolve()
    if ext.parent!=root or not ext.is_file(): raise ValueError('SlangPy extension identity unavailable')
    return {'compiler_sha256':_sha(libs[0]),'runtime_sha256':_sha(runtime),'extension_sha256':_sha(ext)}


def _source_and_closure(operation):
    op=get_operation(operation); source=kernel_path(operation)
    if _sha(source)!=op['sha256']: raise ValueError('catalog source hash mismatch: '+operation)
    dependency_bytes=b''.join(str(p.relative_to(source.parent)).encode()+b'\0'+p.read_bytes()+b'\0' for p in sorted(source.parent.rglob('*.slang')))
    root=source.parent.parent; common=root/'common'
    if common.is_dir(): dependency_bytes+=b''.join(str(p.relative_to(root)).encode()+b'\0'+p.read_bytes()+b'\0' for p in sorted(common.rglob('*.slang')))
    return (str(source.resolve()),op['sha256'],hashlib.sha256(source.read_bytes()+dependency_bytes).hexdigest())


class PreparedCompact:
    """Explicitly prepared, reusable forward-only graph.

    Callers must actually verify their allocation/range declarations and uploaded
    descriptor words. `allocations_verified` and `descriptor_verified` are
    explicit caller attestations, not checks of physical memory or GPU contents.
    Whole buffers only. Read-only inputs may overlap; writable buffers must be
    disjoint from all roles. Inputs, buffers and descriptor stay alive and stable
    through all submissions. `kernels`, when supplied, are keyed by catalog ID;
    their source/device/ABI identities must be verified and explicitly attested
    by the caller; this path cannot independently verify compiled identity.

    Adapter preparation requires caller-specified SHA-256 hashes for the
    installed compiler, SGL runtime and loaded Python extension files. It rejects
    preexisting per-operation cache entries, verifies catalog source bytes and
    adapter source closure at setup and checks cache identity before each record.
    ``verify_source_identity()`` explicitly rechecks file hashes when desired.
    Filesystem changes after preparation and file-change races are not detected
    automatically. File hashes do not attest dynamic linker mappings or compiler
    options.
    """
    def __init__(self,plan,buffers,ranges,*,limits,descriptor_words,allocations_verified,descriptor_verified,
                 allow_experimental=False,adapter=None,kernels=None,runtime_identity=None,prebound_identity_attested=False):
        if allow_experimental is not True: raise ValueError('explicit experimental opt-in required')
        if allocations_verified is not True or descriptor_verified is not True: raise ValueError('caller verification attestations required')
        if (adapter is None)==(kernels is None): raise ValueError('supply exactly one adapter or prebound kernel map')
        if adapter is not None and (type(adapter) is not SlangPyAdapter or adapter.allow_experimental is not True): raise ValueError('SlangPyAdapter experimental opt-in required')
        if kernels is not None and prebound_identity_attested is not True: raise ValueError('prebound compiled identity must be caller attested')
        if kernels is not None and runtime_identity is not None: raise ValueError('prebound runtime identity is caller-attested only')
        try:
            reconstructed=_rebuild(plan,limits)
            if plan!=reconstructed: raise ValueError('plan differs from validated canonical plan')
        except (KeyError,TypeError) as exc: raise ValueError('invalid plan or limits') from exc
        plan=deepcopy(reconstructed); resources=plan['resources']
        if not isinstance(descriptor_words,(tuple,list)) or list(descriptor_words)!=plan['shape_strides_u32'] or any(type(v) is not int or not 0<=v<=0xffffffff for v in descriptor_words):
            raise ValueError('descriptor words must exactly match plan')
        if set(buffers)!=set(resources) or set(ranges)!=set(resources): raise ValueError('complete resources required')
        for name,desc in resources.items():
            region=ranges[name]; buf=buffers[name]
            if not isinstance(region,dict) or set(region)!={'allocation_id','offset','length'} or not isinstance(region['allocation_id'],str) or not region['allocation_id']:
                raise ValueError('caller verified allocation identity required')
            if type(region['offset']) is not int or region['offset']!=0: raise ValueError('only whole buffer binding supported')
            if type(region['length']) is not int or type(getattr(buf,'size',None)) is not int or region['length']!=buf.size:
                raise ValueError('range must cover actual whole buffer')
            if buf.size<desc['allocated_bytes'] or buf.size>limits['max_buffer_bytes'] or buf.size%4: raise ValueError('buffer size/device limit')
            stride=8 if name=='shape_strides' else 4
            if getattr(buf,'struct_size',None)!=stride or buf.size%stride: raise ValueError('buffer element stride mismatch: '+name)
        names=list(resources)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                writable=resources[a]['access']!='read' or resources[b]['access']!='read'
                if buffers[a] is buffers[b] and ranges[a]['allocation_id']!=ranges[b]['allocation_id']: raise ValueError('inconsistent buffer identity')
                if writable and (buffers[a] is buffers[b] or ranges[a]['allocation_id']==ranges[b]['allocation_id']):
                    raise ValueError('writable buffer overlap or shared allocation')
        operations={_operation(stage) for stage in plan['stages']}
        self._adapter=adapter; self._runtime_identity=None; self._source_identity={}
        if adapter is not None:
            if not isinstance(runtime_identity,dict) or set(runtime_identity)!={'compiler_sha256','runtime_sha256','extension_sha256'} or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in runtime_identity.values()):
                raise ValueError('expected compiler/runtime/extension hashes required')
            if _runtime_hashes()!=runtime_identity: raise ValueError('SlangPy runtime hash mismatch')
            if any(op in adapter._kernels or op in adapter.source_identities for op in operations): raise ValueError('adapter cache must be empty for requested stages')
            self._runtime_identity=dict(runtime_identity)
            self._source_identity={op:_source_and_closure(op) for op in operations}
        prepared={}
        for stage in plan['stages']:
            operation=_operation(stage)
            if operation not in prepared:
                if adapter is not None: prepared[operation]=adapter.prepare(operation)
                else:
                    if operation not in kernels: raise ValueError('missing prebound kernel: '+operation)
                    prepared[operation]=kernels[operation]
                if not callable(getattr(prepared[operation],'dispatch',None)): raise ValueError('invalid prebound kernel')
        dispatches=[]
        for stage in plan['stages']:
            bindings={name:buffers[label] for name,label in stage['bindings'].items()}
            if stage['entry'] in SCAN_ENTRIES: bindings.update(stage['params'])
            else: bindings['params']=dict(stage['params'])
            dispatches.append((prepared[_operation(stage)],tuple(stage['thread_count']),bindings))
        self._dispatches=tuple(dispatches); self._prepared=prepared
        if adapter is not None: self.verify_source_identity()

    def _verify_cache(self):
        for operation,identity in self._source_identity.items():
            if self._adapter._kernels.get(operation) is not self._prepared[operation] or self._adapter.source_identities.get(operation)!=identity[2]:
                raise ValueError('adapter cache identity changed: '+operation)

    def verify_source_identity(self):
        """Rehash adapter sources and runtime files; explicit diagnostic I/O."""
        if self._adapter is None: raise ValueError('prebound compiled identity is caller-attested')
        if _runtime_hashes()!=self._runtime_identity: raise ValueError('SlangPy runtime changed')
        for operation,identity in self._source_identity.items():
            if _source_and_closure(operation)!=identity: raise ValueError('source closure changed: '+operation)
        self._verify_cache()

    def record(self,encoder):
        if encoder is None or not callable(getattr(encoder,'global_barrier',None)): raise ValueError('caller-owned barrier-capable encoder required')
        if self._adapter is not None: self._verify_cache()
        for kernel,threads,bindings in self._dispatches:
            encoder.global_barrier()
            kernel.dispatch(thread_count=threads,vars=bindings,command_encoder=encoder)
