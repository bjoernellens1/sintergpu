from dataclasses import replace

import pytest

from sintergpu.camera_registration import depth_register_plan
from benchmarks.camera_registration_support import BufferDescriptor, PreparedCameraGraph
from types import SimpleNamespace


class FakeDevice:
    def __init__(self): self.created = []
    def load_module(self, path):
        self.created.append(('module', path))
        return SimpleNamespace(entry_point=lambda name: name)
    def link_program(self, modules, entries):
        self.created.append(('link', entries[0]))
        return entries[0]
    def create_compute_kernel(self, program):
        self.created.append(('kernel', program))
        return object()


def setup():
    graph = depth_register_plan(1, 3, 4, 5, 6, max_workgroups_x=10, debug=True)
    device = FakeDevice()
    buffers = {name: object() for name in graph['resources']}
    descriptors = {name: BufferDescriptor(buffer, device, object(), 0, spec['bytes'], frozenset({'shader_resource', 'unordered_access'}))
                   for (name, spec), buffer in zip(graph['resources'].items(), buffers.values())}
    return graph, device, buffers, descriptors


def prepare(graph, device, buffers, descriptors, features=('shaderInt64',)):
    return PreparedCameraGraph(device, graph, buffers, descriptors=descriptors, enabled_features=features)


def test_preflight_accepts_owner_allocation_evidence_and_retains_it():
    g, d, b, descriptions = setup()
    prepared = prepare(g, d, b, descriptions)
    assert prepared.descriptors['depth'] is descriptions['depth']


@pytest.mark.parametrize('kind', ['small', 'wrong_device', 'wrong_buffer', 'usage', 'missing_allocation', 'negative_offset'])
def test_preflight_rejects_invalid_owner_metadata_before_compilation(kind):
    g, d, b, descriptions = setup()
    key = 'input_depth'; original = descriptions[key]
    changes = {'small': dict(size_bytes=g['resources'][key]['bytes']-1), 'wrong_device': dict(device=object()),
               'wrong_buffer': dict(buffer=object()), 'usage': dict(usage=frozenset()),
               'missing_allocation': dict(allocation=None), 'negative_offset': dict(offset_bytes=-1)}
    descriptions[key] = replace(original, **changes[kind])
    with pytest.raises(ValueError): prepare(g, d, b, descriptions)
    assert not d.created


def test_unsafe_shared_allocation_and_missing_features_rejected():
    g, d, b, descriptions = setup()
    a, other = 'input_depth', 'depth_rays'
    descriptions[other] = replace(descriptions[other], allocation=descriptions[a].allocation)
    with pytest.raises(ValueError, match='alias'): prepare(g, d, b, descriptions)
    g, d, b, descriptions = setup()
    with pytest.raises(ValueError, match='shaderInt64'): prepare(g, d, b, descriptions, ())
    assert not d.created


def test_disjoint_suballocations_and_required_srv_uav_are_accepted():
    g, d, b, descriptions = setup()
    a, other = 'input_depth', 'depth_rays'
    descriptions[other] = replace(descriptions[other], allocation=descriptions[a].allocation, offset_bytes=descriptions[a].size_bytes)
    descriptions[a] = replace(descriptions[a], usage=frozenset({'shader_resource'}))
    descriptions['min_depth_bits'] = replace(descriptions['min_depth_bits'], usage=frozenset({'unordered_access'}))
    prepare(g, d, b, descriptions)
    descriptions['min_depth_bits'] = replace(descriptions['min_depth_bits'], usage=frozenset({'shader_resource'}))
    with pytest.raises(ValueError, match='usage'): prepare(g, d, b, descriptions)
