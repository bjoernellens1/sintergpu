# SPDX-License-Identifier: Apache-2.0
"""Opt-in incumbent parity; external exclusive GPU coordination is required."""
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.depth_pyramid_gpu_fixture import Fixture, ROOT, preflight, specification
from sintergpu.references.depth_pyramid_profiles import slam_level, vkl_level

F = np.float32
EPS = F(np.finfo(np.float32).eps)
OPTIN = pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS') != '1', reason='explicit GPU opt-in and external lock required')


def cases():
    for mask in range(16):
        d = np.array([1, 1, 1, 1], np.float32).reshape(1, 2, 2)
        v = np.array([7 if mask & (1 << i) else 0 for i in range(4)], np.uint32).reshape(d.shape)
        c = np.array([.1, .2, .3, .4], np.float32).reshape(d.shape)
        yield f'vkl-mask-{mask}', specification('vkl', d, valid=v, confidence=c, threshold=EPS)
        yield f'slam-mask-{mask}', specification('slam', np.where(v[0], d[0], F(0)), threshold=0)
    for value in [F(1e-8), F(1e-30), F(-0.), F(0.)]:
        d = np.full((1,2,2), value, np.float32)
        yield f'vkl-near-zero-{value!r}', specification('vkl', d, valid=np.ones_like(d, np.uint32), confidence=np.full_like(d,.5), threshold=EPS)
        yield f'slam-near-zero-{value!r}', specification('slam', d[0], threshold=1.)
    for shape in [(2,1,1), (2,1,7), (2,7,1), (3,5,7), (2,34,38)]:
        rng = np.random.default_rng(811 + sum(shape))
        d = rng.uniform(.1, 4, shape).astype(np.float32)
        v = rng.integers(0, 3, shape, dtype=np.uint32) * 7
        c = rng.random(shape, dtype=np.float32)
        d.flat[:min(d.size, 6)] = np.array([np.nan, np.inf, -np.inf, -0., -1, 1e21], np.float32)[:min(d.size, 6)]
        yield f'vkl-shape-{shape}', specification('vkl', d, valid=v, confidence=c, threshold=EPS)
    yield 'slam-tracker-api-finite', specification('slam', np.full((34,38), F(1e-8), np.float32), threshold=.1)
    for shape in [(2,2), (34,38), (70,68), (240,320)]:
        rng = np.random.default_rng(sum(shape))
        d = rng.uniform(.1, 4, shape).astype(np.float32)
        d.flat[:4] = [np.nan, np.inf, -0., 1e21]
        yield f'slam-shape-{shape}', specification('slam', d, threshold=1.)
    for threshold in [np.nextafter(F(1), F(0)), F(1), np.nextafter(F(1), F(np.inf))]:
        d = np.array([[1,2],[0,-1]], np.float32)
        yield f'slam-threshold-{threshold!r}', specification('slam', d, threshold=threshold)
    for last in [np.nextafter(F(1)+EPS*F(32), F(0)), F(1)+EPS*F(32), np.nextafter(F(1)+EPS*F(32), F(np.inf))]:
        d = np.array([1,1,1,last], np.float32).reshape(1,2,2)
        yield f'vkl-threshold-{last!r}', specification('vkl', d, valid=np.ones_like(d, np.uint32), confidence=np.array([.125,.25,.5,1],np.float32).reshape(d.shape), threshold=EPS)
    for values, jump in [([np.inf, 0, 0, 0], np.inf), ([1e21, 0, 0, 0], 0), ([1e21, 0, 0, 0], 1e21), ([1, 2, 0, 0], 1)]:
        yield f'slam-edge-{values}-{jump}', specification('slam', np.array(values, np.float32).reshape(2,2), threshold=jump)


CASES = list(cases())


def expected(spec):
    a, p = spec['inputs'], spec['params']
    if spec['profile'] == 'vkl':
        d, v, c = vkl_level(a['depth'], a['valid'], a['confidence'], p['epsilon'])
        return dict(depthOut=d, validOut=v, confidenceOut=c)
    return dict(outputDepth=slam_level(a['inputDepth'], p['depthJump']))


def compare(actual, reference):
    assert actual.keys() == reference.keys()
    for name, value in reference.items():
        assert actual[name].shape == value.shape and actual[name].dtype == value.dtype
        if value.dtype.kind in 'iu':
            np.testing.assert_array_equal(actual[name], value, err_msg=name)
        else:
            a = actual[name]
            np.testing.assert_array_equal(np.isnan(a), np.isnan(value), err_msg=name)
            np.testing.assert_array_equal(np.isposinf(a), np.isposinf(value), err_msg=name)
            np.testing.assert_array_equal(np.isneginf(a), np.isneginf(value), err_msg=name)
            zero = value == 0
            np.testing.assert_array_equal(a == 0, zero, err_msg=name + ' zero classification')
            np.testing.assert_array_equal(a[zero].view(np.uint32), value[zero].view(np.uint32), err_msg=name + ' zero bits')
            finite = np.isfinite(value) & ~zero
            assert np.isfinite(a[finite]).all(), name
            # Ordered signed float bit representation; eight ULPs covers scalar
            # accumulation rounding without permitting zero/branch substitution.
            def ordered(x):
                bits = x.view(np.uint32).astype(np.int64)
                return np.where(bits & 0x80000000, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)
            assert np.all(np.abs(ordered(a[finite]) - ordered(value[finite])) <= 8), name


def test_preflight_and_all_fixture_mappings():
    assert len(preflight()) == 6
    assert len({name for name, _ in CASES}) == len(CASES)
    for _, spec in CASES:
        result = expected(spec)
        assert set(result) == set(spec['outputs'])
        assert all(a.shape == spec['shape'] and a.size == spec['count'] for a in result.values())
    assert sum(name.startswith('vkl-mask-') for name, _ in CASES) == 16
    assert sum(name.startswith('slam-mask-') for name, _ in CASES) == 16


def test_invalid_host_resources():
    d = np.ones((1, 2, 2), np.float32); v = np.ones_like(d, np.uint32)
    with pytest.raises(ValueError): specification('vkl', d, valid=v.astype(np.float32), confidence=d)
    with pytest.raises(ValueError): specification('vkl', d, valid=v, confidence=d, max_groups=(0, 1))
    with pytest.raises(ValueError): specification('slam', np.ones((3, 2), np.float32))
    with pytest.raises(ValueError): specification('slam', d[0], valid=v[0])
    with pytest.raises(ValueError): specification('slam', np.ones((34,34),np.float32), max_groups=(1,1))


def test_record_uses_borrowed_encoder_only():
    events = []
    class Encoder:
        def global_barrier(self): events.append('barrier')
    class Kernel:
        def dispatch(self, **kwargs): events.append(kwargs)
    f = object.__new__(Fixture)
    f.spec = dict(threads=(2,3,1)); f.bindings = {'x': object()}; f.kernel = Kernel()
    encoder = Encoder(); f.record(encoder)
    assert events == ['barrier', dict(thread_count=(2,3,1), vars=f.bindings, command_encoder=encoder)]
    with pytest.raises(ValueError): f.record(None)


@pytest.fixture(scope='module')
def device():
    identities = preflight()  # Must fail before any Device construction.
    import slangpy as spy
    compiler = list(Path(spy.__file__).parent.glob('libslang-compiler*'))
    if len(compiler) != 1 or not compiler[0].is_file():
        raise RuntimeError('cannot identify bundled SlangPy compiler')
    identities['slangpy_compiler'] = hashlib.sha256(compiler[0].read_bytes()).hexdigest()
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    if len(adapters) != 1 or adapters[0].vendor_id == 0x10005:
        raise RuntimeError('exactly one hardware adapter must be selected externally')
    dev = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=adapters[0].luid, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    identities.update(adapter=dev.info.adapter_name, api=dev.info.api_name, vendor_id=adapters[0].vendor_id, device_id=adapters[0].device_id)
    print('incumbent GPU identity:', identities)
    try: yield dev
    finally: dev.close()


def execute(device, fixture):
    encoder = device.create_command_encoder()
    fixture.record(encoder)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    return fixture.read_checked()


@OPTIN
@pytest.mark.parametrize('case_id,spec', CASES, ids=[name for name, _ in CASES])
def test_incumbent_full_output(device, case_id, spec):
    fixture = Fixture(device, spec)
    compare(execute(device, fixture), expected(spec))


@OPTIN
@pytest.mark.parametrize('profile', ['vkl', 'slam'])
def test_incumbent_reused_buffers(device, profile):
    if profile == 'vkl':
        d = np.ones((2,5,7), np.float32); v = np.ones_like(d, np.uint32)*7; c = np.full_like(d, .5)
        specs = [specification('vkl', d*x, valid=v*(x != 0), confidence=c, threshold=EPS) for x in (1,0,2)]
    else:
        d = np.ones((34,38), np.float32)
        specs = [specification('slam', d*x, threshold=0) for x in (1,0,2)]
    fixture = Fixture(device, specs[0])
    for i, spec in enumerate(specs):
        if i: fixture.upload(spec['inputs'])
        compare(execute(device, fixture), expected(spec))


@OPTIN
def test_vkl_frozen_donor(device):
    torch = pytest.importorskip('torch')
    path = ROOT/'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py'
    module_spec = importlib.util.spec_from_file_location('frozen_vkl_depth_pyramid', path)
    donor = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(donor)
    rng = np.random.default_rng(922)
    d = rng.uniform(.2, 4, (2,5,7)).astype(np.float32); d.flat[0] = np.nan
    v = rng.integers(0,3,d.shape,dtype=np.uint32)*7; c = rng.random(d.shape,dtype=np.float32)
    levels = donor.depth_pyramid(depth=torch.from_numpy(d[:,None]), valid=torch.from_numpy((v != 0)[:,None]), confidence=torch.from_numpy(c[:,None]))
    usable = (v != 0) & np.isfinite(d) & (d > 0)
    current = (np.where(usable,d,F(0)), usable.astype(np.uint32), np.where(usable,c,F(0)))
    for key in ('/2', '/4', '/8'):
        if key not in levels: break
        spec = specification('vkl', current[0], valid=current[1], confidence=current[2], threshold=EPS)
        actual = execute(device, Fixture(device, spec))
        donor_level = levels[key]
        compare(actual, dict(depthOut=donor_level['depth'].numpy()[:,0], validOut=donor_level['valid'].numpy()[:,0].astype(np.uint32), confidenceOut=donor_level['confidence'].numpy()[:,0]))
        current = (actual['depthOut'], actual['validOut'], actual['confidenceOut'])


def test_fake_readback_checks_inputs_and_tails():
    class Buffer:
        def __init__(self, array): self.array = array.copy()
        def to_numpy(self): return self.array.copy()
        def copy_from_numpy(self, array): self.array[:] = array
    spec = specification('vkl', np.ones((1,2,2),np.float32), valid=np.ones((1,2,2),np.uint32), confidence=np.ones((1,2,2),np.float32), threshold=EPS)
    f = object.__new__(Fixture); f.spec = spec
    f.initial = {name: np.full(a.size+19,73,a.dtype) for name,a in spec['inputs'].items()}
    for name,a in spec['inputs'].items(): f.initial[name][:a.size] = a.ravel()
    for name,dtype in spec['outputs'].items(): f.initial[name] = np.full(spec['count']+19,73,dtype)
    f.buffers = {name: Buffer(value) for name,value in f.initial.items()}
    assert set(f.read_checked()) == set(spec['outputs'])
    f.buffers['depthOut'].array[-1] = 0
    with pytest.raises(AssertionError, match='tail'): f.read_checked()
    f.buffers['depthOut'].array[-1] = 73
    f.buffers['depth'].array[0] = 2
    with pytest.raises(AssertionError, match='input mutated'): f.read_checked()
    with pytest.raises(ValueError): f.upload({'depth': spec['inputs']['depth']})
    changed = dict(spec['inputs']); changed['valid'] = changed['valid'].astype(np.float32)
    with pytest.raises(ValueError): f.upload(changed)


def test_near_zero_and_signed_zero_compare_rejects_wrong_values():
    with pytest.raises(AssertionError): compare({'x': np.array([0],np.float32)}, {'x': np.array([1e-8],np.float32)})
    with pytest.raises(AssertionError): compare({'x': np.array([-0.],np.float32)}, {'x': np.array([0.],np.float32)})
    x = np.array([1e-8],np.float32)
    compare({'x': np.nextafter(x, np.float32(np.inf))}, {'x': x})


def test_slam_raw_shader_fixture_scope():
    slam = [(name,s) for name,s in CASES if s['profile'] == 'slam']
    assert any(not spec['tracker_api_compatible'] and not np.isfinite(spec['inputs']['inputDepth']).all() for _,spec in slam)
    valid = specification('slam', np.ones((4,4),np.float32), threshold=.1)
    assert valid['tracker_api_compatible']


def test_actual_prepared_fixture_records_prebound_resources(monkeypatch):
    import sys
    import types
    events = []
    class Buffer:
        def __init__(self, data): self.data = data.copy()
    class Kernel:
        def dispatch(self, **kw): events.append(('dispatch', kw))
    class Device:
        info = types.SimpleNamespace(limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535,y=65535), max_compute_threads_per_group=256, max_compute_thread_group_size=types.SimpleNamespace(x=256,y=256)))
        def create_buffer(self, **kw): events.append(('allocation', kw)); return Buffer(kw['data'])
        def load_module(self, path): events.append(('load',path)); return types.SimpleNamespace(entry_point=lambda x: x)
        def load_module_from_source(self, name, source): events.append(('source', source)); return types.SimpleNamespace(entry_point=lambda x:x)
        def link_program(self, modules, entries): return entries[0]
        def create_compute_kernel(self, program): return Kernel()
    class Encoder:
        def global_barrier(self): events.append(('barrier',))
    monkeypatch.setitem(sys.modules, 'slangpy', types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1, unordered_access=2)))
    for profile in ('vkl','slam'):
        d=np.ones((1,4,4),np.float32)
        spec = specification('vkl',d,valid=np.ones_like(d,np.uint32),confidence=d,threshold=EPS) if profile == 'vkl' else specification('slam',d[0],threshold=.1)
        fixture=Fixture(Device(),spec)
        bindings=fixture.bindings; start=len(events); encoder=Encoder()
        fixture.record(encoder); fixture.record(encoder)
        recorded=events[start:]
        assert [event[0] for event in recorded] == ['barrier','dispatch','barrier','dispatch']
        for event in recorded[1::2]:
            assert event[1]['vars'] is bindings and event[1]['command_encoder'] is encoder
        if profile == 'slam':
            from benchmarks.depth_pyramid_gpu_fixture import composed_slam_source
            assert any(e[0]=='source' and e[1]==composed_slam_source() for e in events)


def test_record_ast_guard_and_adversarial_mutations():
    import ast
    import inspect
    import textwrap
    def valid(source):
        tree=ast.parse(textwrap.dedent(source)); f=tree.body[0]
        # Only the null check, barrier and single prebound dispatch are allowed.
        if len(f.body) != 3 or not isinstance(f.body[0],ast.If): return False
        calls=[n for n in ast.walk(f) if isinstance(n,ast.Call)]
        names=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in calls]
        if sorted(names) != ['ValueError','dispatch','global_barrier']: return False
        if any(isinstance(n,(ast.Dict,ast.DictComp,ast.ListComp,ast.SetComp,ast.GeneratorExp,ast.Assign,ast.AugAssign,ast.For,ast.While)) for n in ast.walk(f)): return False
        return True
    source=inspect.getsource(Fixture.record)
    assert valid(source)
    for mutation in ("\n        self.device.create_buffer()", "\n        self.device.submit()", "\n        self.device.wait()", "\n        self.buffers['x'].to_numpy()", "\n        bindings = dict(self.bindings)"):
        assert not valid(source + mutation)


def test_fake_recording_benchmark_samples():
    from benchmarks.depth_pyramid_recording import measure
    report = measure()
    assert 'no GPU' in report['scope']
    assert {row['profile'] for row in report['measurements']} == {'vkl','slam'}
    for row in report['measurements']:
        assert row['warmups']==5 and len(row['samples_ns'])==50 and all(type(n) is int and n>0 for n in row['samples_ns'])
