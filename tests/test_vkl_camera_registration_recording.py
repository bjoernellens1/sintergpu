import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from sintergpu.camera_registration import camera_rectify_plan, depth_register_plan
from benchmarks.camera_registration_support import BufferDescriptor, PreparedCameraGraph


class FakeKernel:
    def __init__(self, entry): self.entry = entry
    def dispatch(self, *, thread_count, vars, command_encoder):
        command_encoder.calls.append(('dispatch', self.entry, thread_count, vars))


class FakeDevice:
    def __init__(self): self.created = []
    def load_module(self, path):
        self.created.append(('module', path))
        return SimpleNamespace(entry_point=lambda name: name)
    def link_program(self, modules, entries):
        self.created.append(('link', entries[0])); return entries[0]
    def create_compute_kernel(self, program):
        self.created.append(('kernel', program)); return FakeKernel(program)


class FakeEncoder:
    def __init__(self): self.calls = []
    def global_barrier(self): self.calls.append(('barrier',))


def make(graph, features=()):
    device = FakeDevice()
    buffers = {name: object() for name in graph['resources']}
    descriptors = {name: BufferDescriptor(buffer, device, object(), 0, graph['resources'][name]['bytes'], frozenset({'shader_resource', 'unordered_access'})) for name, buffer in buffers.items()}
    return device, buffers, PreparedCameraGraph(device, graph, buffers, descriptors=descriptors, enabled_features=features)


def test_record_two_invocations_prepared_bindings_and_declared_order():
    graph = depth_register_plan(1,3,4,5,6,max_workgroups_x=10,debug=True)
    device, buffers, prepared = make(graph, ('shaderInt64',))
    setup = tuple(device.created)
    encoder = FakeEncoder()
    prepared.record(encoder); prepared.record(encoder)
    assert tuple(device.created) == setup
    expected = []
    for stage in graph['stages']:
        if stage['barriers']: expected.append(('barrier',))
        expected.append(('dispatch', stage['entrypoint']))
    assert [(x[0], *x[1:2]) for x in encoder.calls] == expected * 2
    dispatches = [x for x in encoder.calls if x[0] == 'dispatch']
    count = len(graph['stages'])
    for i, stage in enumerate(graph['stages']):
        first, second = dispatches[i], dispatches[i+count]
        assert first[2] is second[2] and first[3] is second[3]
        assert first[3][stage['params_binding']] == stage['params']
        with pytest.raises(TypeError): first[3]['unexpected'] = None
        with pytest.raises(TypeError): first[3][stage['params_binding']]['batches'] = 42
        assert all(first[3][binding] is buffers[resource] for binding, resource in stage['bindings'].items())
    with pytest.raises(ValueError): prepared.record(None)


def test_setup_rejects_incomplete_inputs_and_missing_required_features():
    graph = depth_register_plan(1,3,4,5,6,max_workgroups_x=10,debug=True)
    with pytest.raises(ValueError, match='shaderInt64'):
        make(graph)
    with pytest.raises(ValueError, match='resource'):
        PreparedCameraGraph(FakeDevice(), graph, {}, descriptors={}, enabled_features=('shaderInt64',))
    with pytest.raises(ValueError, match='device'):
        PreparedCameraGraph(None, graph, {}, descriptors={}, enabled_features=('shaderInt64',))


def assert_record_safe(source):
    tree = ast.parse(textwrap.dedent(source))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'record')
    # Bindings and dispatch data are created only in setup. No comprehensions or
    # container constructors/literals in the per-stage path.
    for n in ast.walk(fn):
        assert not isinstance(n, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp))
        if isinstance(n, ast.Call):
            name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
            assert name in {'ValueError', 'global_barrier', 'dispatch'}, name


def test_record_ast_guard_and_mutants():
    source = inspect.getsource(PreparedCameraGraph.record)
    assert_record_safe(source)
    for mutation in (source.replace('kernel.dispatch(', 'dict()\n            kernel.dispatch('),
                     source.replace('encoder.global_barrier()', 'encoder.wait_for_idle()'),
                     source.replace('kernel.dispatch(', 'device.create_buffer()\n            kernel.dispatch('),
                     source.replace('kernel.dispatch(', 'buffer.to_numpy()\n            kernel.dispatch(')):
        with pytest.raises(AssertionError): assert_record_safe(mutation)


def test_input_resources_are_borrowed_and_extra_resources_rejected():
    graph = camera_rectify_plan(1,3,3,4,5,6,max_workgroups_x=10)
    device, buffers, prepared = make(graph)
    assert all(prepared.buffers[name] is resource for name, resource in buffers.items())
    buffers['extra'] = object()
    with pytest.raises(ValueError, match='resource'):
        PreparedCameraGraph(device, graph, buffers, descriptors=prepared.descriptors)
