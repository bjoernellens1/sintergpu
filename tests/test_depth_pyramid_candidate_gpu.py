# SPDX-License-Identifier: Apache-2.0
"""Opt-in candidate versus incumbent and independent CPU references.

Requires external exclusive GPU lock, independent review and BOTH explicit
SINTERGPU_GPU_TESTS=1 and SINTERGPU_DEPTH_PYRAMID_CANDIDATE_GPU=1.
"""
import os
import types
import numpy as np
import pytest

from test_depth_pyramid_gpu import CASES, EPS, compare, device as incumbent_device, execute, expected
from benchmarks.depth_pyramid_gpu_fixture import Fixture, specification
from depth_pyramid_candidate_fixture import candidate_source, candidate_fixture, candidate_preflight

CANDIDATE_OPTIN = pytest.mark.skipif(
    os.getenv('SINTERGPU_GPU_TESTS') != '1' or os.getenv('SINTERGPU_DEPTH_PYRAMID_CANDIDATE_GPU') != '1',
    reason='candidate GPU opt-in, independent review and external exclusive lock required')


@pytest.fixture(scope='module')
def candidate_device():
    candidate_preflight()  # Fail before device construction.
    generator = incumbent_device.__wrapped__()
    dev = next(generator)
    try:
        yield dev
    finally:
        generator.close()


def test_candidate_source_preflight_and_composition():
    identities = candidate_preflight()
    assert len(identities) == 9
    for profile in ('vkl', 'slam'):
        source = candidate_source(profile)
        assert '#include "depth_pyramid_core.slang"' not in source
        assert 'DepthPyramidVklResult depthPyramidVklBlock' in source
        assert ('caspar_backproject' in source) == (profile == 'slam')
    with pytest.raises(ValueError): candidate_source('other')


def test_candidate_fixture_loads_composed_source_and_borrowed_bindings(monkeypatch):
    import sys
    events = []
    class Kernel:
        def dispatch(self, **kwargs): events.append(('dispatch', kwargs))
    class Device:
        info = types.SimpleNamespace(limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=types.SimpleNamespace(x=65535,y=65535), max_compute_threads_per_group=256, max_compute_thread_group_size=types.SimpleNamespace(x=256,y=256)))
        def create_buffer(self, **kwargs): return object()
        def load_module(self, path): events.append(('incumbent',path)); return types.SimpleNamespace(entry_point=lambda x:x)
        def load_module_from_source(self, name, source): events.append(('source',name,source)); return types.SimpleNamespace(entry_point=lambda x:x)
        def link_program(self, modules, entries): return entries[0]
        def create_compute_kernel(self, program): return Kernel()
    class Encoder:
        def global_barrier(self): events.append(('barrier',))
    monkeypatch.setitem(sys.modules, 'slangpy', types.SimpleNamespace(BufferUsage=types.SimpleNamespace(shader_resource=1,unordered_access=2)))
    for profile in ('vkl','slam'):
        d=np.ones((1,4,4),np.float32)
        spec=specification('vkl',d,valid=np.ones_like(d,np.uint32),confidence=d,threshold=EPS) if profile == 'vkl' else specification('slam',d[0],threshold=.1)
        f=candidate_fixture(Device(),spec)
        assert events[-1][0] == 'source' and events[-1][2] == candidate_source(profile)
        bindings=f.bindings; encoder=Encoder(); f.record(encoder)
        assert events[-2][0] == 'barrier' and events[-1][0] == 'dispatch'
        assert events[-1][1]['vars'] is bindings and events[-1][1]['command_encoder'] is encoder


def assert_exact(actual, baseline):
    assert actual.keys() == baseline.keys()
    for name, value in baseline.items():
        assert actual[name].dtype == value.dtype and actual[name].shape == value.shape
        np.testing.assert_array_equal(actual[name].view(np.uint32), value.view(np.uint32), err_msg=name)


@CANDIDATE_OPTIN
@pytest.mark.parametrize('case_id,spec', CASES, ids=[name for name, _ in CASES])
def test_candidate_against_incumbent_and_independent_reference(candidate_device, case_id, spec):
    device = candidate_device
    candidate_preflight()
    incumbent = execute(device, Fixture(device, spec))
    candidate = execute(device, candidate_fixture(device, spec))
    oracle = expected(spec)
    compare(incumbent, oracle)
    compare(candidate, oracle)
    assert_exact(candidate, incumbent)
    candidate_preflight()


@CANDIDATE_OPTIN
@pytest.mark.parametrize('profile', ['vkl', 'slam'])
def test_candidate_reuses_buffers_without_stale_results(candidate_device, profile):
    device = candidate_device
    if profile == 'vkl':
        d=np.ones((2,5,7),np.float32); v=np.ones_like(d,np.uint32)*7; c=np.full_like(d,.5)
        specs=[specification('vkl',d*x,valid=v*(x != 0),confidence=c,threshold=EPS) for x in (1,0,2)]
    else:
        d=np.ones((34,38),np.float32)
        specs=[specification('slam',d*x,threshold=0) for x in (1,0,2)]
    incumbent=Fixture(device,specs[0]); candidate=candidate_fixture(device,specs[0])
    for i,spec in enumerate(specs):
        if i:
            incumbent.upload(spec['inputs']); candidate.upload(spec['inputs'])
        old=execute(device,incumbent); new=execute(device,candidate)
        compare(old,expected(spec)); compare(new,expected(spec)); assert_exact(new,old)


def test_candidate_preflight_rejects_changed_or_symlinked_sources(tmp_path):
    from pathlib import Path
    from benchmarks.depth_pyramid_gpu_fixture import ROOT, SOURCES
    from depth_pyramid_candidate_fixture import CANDIDATES
    for name in {**SOURCES, **CANDIDATES}:
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    assert len(candidate_preflight(tmp_path)) == 9
    candidate=tmp_path/next(iter(CANDIDATES))
    original=candidate.read_bytes()
    candidate.write_bytes(original+b'\n')
    with pytest.raises(ValueError, match='candidate source digest mismatch'):
        candidate_preflight(tmp_path)
    candidate.unlink()
    candidate.symlink_to(ROOT/next(iter(CANDIDATES)))
    with pytest.raises(ValueError, match='symlinked candidate source'):
        candidate_preflight(tmp_path)


def test_bitwise_comparison_rejects_one_ulp_and_signed_zero():
    x=np.array([1.],np.float32)
    with pytest.raises(AssertionError):
        assert_exact({'d':np.nextafter(x,np.float32(np.inf))},{'d':x})
    with pytest.raises(AssertionError):
        assert_exact({'d':np.array([-0.],np.float32)},{'d':np.array([0.],np.float32)})


def test_fixture_case_coverage():
    names=[name for name,_ in CASES]
    for profile in ('vkl','slam'):
        assert sum(name.startswith(profile+'-mask-') for name in names) == 16
        assert any(name.startswith(profile+'-threshold-') for name in names)
    assert any(name.startswith('vkl-shape-') and spec['shape'][0]>1 for name,spec in CASES)
    assert any(name.startswith('slam-edge-') for name in names)
