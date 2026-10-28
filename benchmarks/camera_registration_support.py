# SPDX-License-Identifier: Apache-2.0
"""Prepare caller-owned camera graph resources for borrowed-encoder recording.

This qualification helper does not allocate buffers, submit, synchronize, or
read back. The allocator owner supplies immutable metadata describing the actual buffer,
allocation, device, byte interval and usages, and the enabled device features.
The owner retains allocations and guarantees validity through GPU completion.
Metadata must come from allocation records, not inferred from graph requests;
this helper cannot independently verify device internals or detect stale records. Compilation and
binding creation occur once at construction, outside the record loop.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / 'src/sintergpu/kernels/vkl/camera_registration.slang'


@dataclass(frozen=True)
class BufferDescriptor:
    """Owner-attested buffer allocation; byte interval refers to allocation."""
    buffer: object
    device: object
    allocation: object
    offset_bytes: int
    size_bytes: int
    usage: frozenset


_READ_BINDINGS = frozenset(('sourceIntrinsics', 'targetIntrinsics', 'sourceDistortion',
    'image', 'sourceUV', 'valid', 'depthIntrinsics', 'depthDistortion', 'depth',
    'depthRays', 'rgbIntrinsics', 'transformRgbDepth', 'rgbDistortion'))
_WRITE_BINDINGS = frozenset(('sourceUVOut', 'targetRayOut', 'displacementOut',
    'imageOut', 'depthRaysOut', 'projectedZOut', 'depthOut', 'validOut',
    'sourceValidOut', 'destinationIndexOut', 'minDepthBits', 'winnerIndex',
    'collisionCount', 'winnerSourceIndexOut', 'collisionCountOut'))


def preflight_resources(device, graph, buffers, descriptors, enabled_features):
    """Check owner-supplied records; owner remains responsible for truth/lifetime."""
    if device is None:
        raise ValueError('caller-owned device required')
    missing = set(graph['features']) - set(enabled_features)
    if missing:
        raise ValueError('device missing enabled features: ' + ', '.join(sorted(missing)))
    resources = graph['resources']
    if set(buffers) != set(resources) or set(descriptors) != set(resources):
        raise ValueError('caller resource and descriptor sets must match graph resources')
    required = {name: set() for name in resources}
    for stage in graph['stages']:
        for binding, resource in stage['bindings'].items():
            if binding in _READ_BINDINGS:
                required[resource].add('shader_resource')
            elif binding in _WRITE_BINDINGS:
                required[resource].add('unordered_access')
            else:
                raise ValueError('unknown shader binding: ' + binding)
    prior = []
    for name, spec in resources.items():
        descriptor = descriptors[name]
        if not isinstance(descriptor, BufferDescriptor) or descriptor.buffer is None or descriptor.buffer is not buffers[name]:
            raise ValueError('resource buffer identity not established: ' + name)
        if descriptor.device is not device or descriptor.allocation is None:
            raise ValueError('resource device or allocation identity mismatch: ' + name)
        if (type(descriptor.offset_bytes) is not int or descriptor.offset_bytes < 0 or
            type(descriptor.size_bytes) is not int or descriptor.size_bytes < spec['bytes']):
            raise ValueError('resource byte capacity or offset invalid: ' + name)
        if not isinstance(descriptor.usage, frozenset) or not required[name] <= descriptor.usage:
            raise ValueError('resource usage missing: ' + name)
        for other_name, other in prior:
            if descriptor.buffer is other.buffer:
                raise ValueError('resource alias: ' + name + ' and ' + other_name)
            if descriptor.allocation is other.allocation and descriptor.offset_bytes < other.offset_bytes + other.size_bytes and other.offset_bytes < descriptor.offset_bytes + descriptor.size_bytes:
                raise ValueError('resource allocation alias: ' + name + ' and ' + other_name)
        prior.append((name, descriptor))


class _FrozenDict(dict):
    """A dict accepted by SlangPy whose public mutation operations fail."""
    def _immutable(self, *args, **kwargs):
        raise TypeError('prepared bindings are immutable')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


class PreparedCameraGraph:
    def __init__(self, device, graph, buffers, *, descriptors, enabled_features=()):
        preflight_resources(device, graph, buffers, descriptors, enabled_features)
        self.device = device
        self.graph = graph
        self.buffers = dict(buffers)
        self.descriptors = _FrozenDict(descriptors)
        module = device.load_module(str(SHADER))
        kernels = {}
        dispatches = []
        for stage in graph['stages']:
            name = stage['entrypoint']
            if name not in kernels:
                kernels[name] = device.create_compute_kernel(device.link_program([module], [module.entry_point(name)]))
            bindings = {stage['params_binding']: _FrozenDict(stage['params'])}
            bindings.update({binding: self.buffers[resource] for binding, resource in stage['bindings'].items()})
            dispatches.append((kernels[name], tuple(stage['thread_count']), _FrozenDict(bindings), bool(stage['barriers'])))
        # Bindings and parameter dictionaries are frozen before any recording.
        self._dispatches = tuple(dispatches)

    def record(self, encoder):
        if encoder is None:
            raise ValueError('borrowed command encoder required')
        for kernel, threads, bindings, barrier in self._dispatches:
            if barrier:
                encoder.global_barrier()
            kernel.dispatch(thread_count=threads, vars=bindings, command_encoder=encoder)
