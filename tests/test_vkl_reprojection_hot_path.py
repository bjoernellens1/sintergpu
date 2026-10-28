"""CPU-only guards for borrowed recording, sampling and staged scratch reset."""

def test_record_has_only_borrowed_barrier_and_dispatch_calls():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'benchmarks/reprojection_support.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PreparedGraph')
    record = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'record')
    calls = [node for node in ast.walk(record) if isinstance(node, ast.Call)]
    assert {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)} == {'global_barrier', 'dispatch'}
    assert {node.func.id for node in calls if isinstance(node.func, ast.Name)} <= {'ValueError'}
    assert not any(isinstance(node, (ast.ListComp, ast.DictComp, ast.Import, ast.ImportFrom)) for node in ast.walk(record))


def test_shader_cast_reset_reduction_and_atomic_guards():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'src/sintergpu/kernels/vkl/reprojection.slang').read_text()
    def body(name):
        start = text.index('{', text.index(name + '('))
        depth = 1; end = start
        while depth:
            end += 1
            depth += (text[end] == '{') - (text[end] == '}')
        return text[start:end + 1]
    assert 'return' not in body('projectPoints')  # Every padded lane reaches barriers.
    assert body('reduceTrees').count('GroupMemoryBarrierWithGroupSync()') == 2
    assert 'p+=256' in body('reduceProjection')
    assert 'for(' not in body('rasterMin') + body('rasterSupport')
    assert 'minDepthBits[i]=0x7f800000;maxConfidenceBits[i]=0;supportBits[i]=0' in body('clearRaster')
    assert 'InterlockedMin' in body('rasterMin')
    assert 'InterlockedMax' in body('rasterSupport') and 'InterlockedAdd' in body('rasterSupport')
    assert body('nearestDepth').index('!isfinite(u)') < body('nearestDepth').index('nearestEven(u)')
    assert body('bilinear').index('!isfinite(u)') < body('bilinear').index('int(floor(u))')
    assert 'visibleOut[i]==0' in body('rasterMin') and 'visibleOut[i]==0' in body('rasterSupport')
    assert 'source!=0' in body('packTemporal') and 'packedTrusted[i]=0' in body('packTemporal')



def test_two_records_only_use_supplied_encoder_and_prepared_bindings():
    from benchmarks.reprojection_support import PreparedGraph
    calls = []
    class Encoder:
        def global_barrier(self): calls.append(('barrier', self))
        def __getattr__(self, name): raise AssertionError('unexpected encoder work: ' + name)
    class Kernel:
        def dispatch(self, *, thread_count, vars, command_encoder):
            calls.append(('dispatch', command_encoder, thread_count, vars))
        def __getattr__(self, name): raise AssertionError('unexpected kernel work: ' + name)
    graph = PreparedGraph.__new__(PreparedGraph)
    bindings = {'params': {'pairs': 1}, 'input': object()}
    threads = [256, 1, 1]
    graph.dispatches = [(Kernel(), threads, bindings, True)]
    first, second = Encoder(), Encoder()
    graph.record(first); graph.record(second)
    assert [call[0] for call in calls] == ['barrier', 'dispatch', 'barrier', 'dispatch']
    assert calls[1][1] is first and calls[3][1] is second
    assert calls[1][2] is threads and calls[3][3] is bindings


def test_record_guard_rejects_allocation_transfer_submit_and_wait_mutations():
    import ast
    from pathlib import Path
    import pytest
    source = (Path(__file__).resolve().parents[1] / 'benchmarks/reprojection_support.py').read_text()
    def allowed(text):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PreparedGraph')
        record = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'record')
        return {n.func.attr for n in ast.walk(record) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)} <= {'global_barrier', 'dispatch'}
    assert allowed(source)
    for forbidden in ('create_buffer', 'load_module', 'create_compute_kernel', 'copy_from_numpy', 'to_numpy', 'submit_command_buffer', 'wait_for_idle'):
        assert not allowed(source.replace('encoder.global_barrier()', 'encoder.' + forbidden + '()', 1)), forbidden


def test_isolated_stateful_stage_replays_prerequisites_for_every_sample(monkeypatch):
    import sys
    import types
    from benchmarks.reprojection_support import measure_stage
    monkeypatch.setitem(sys.modules, 'slangpy', types.SimpleNamespace(QueryType=types.SimpleNamespace(timestamp='timestamp')))
    calls = []
    class Encoder:
        def global_barrier(self): calls.append('barrier')
        def write_timestamp(self, pool, index): calls.append(('timestamp', index))
        def finish(self): return self
    class Kernel:
        def __init__(self, name): self.name = name
        def dispatch(self, **kwargs): calls.append(self.name)
    class Pool:
        def get_results(self, start, count): return [1, 2, 3, 4]
    class Device:
        info = types.SimpleNamespace(timestamp_frequency=1000)
        def create_command_encoder(self): return Encoder()
        def submit_command_buffer(self, encoder): pass
        def wait_for_idle(self): pass
        def create_query_pool(self, kind, count): return Pool()
    prepared = types.SimpleNamespace(device=Device(), dispatches=[(Kernel('clear'), [256, 1, 1], {}, True), (Kernel('minimum'), [256, 1, 1], {}, True), (Kernel('support'), [256, 1, 1], {}, True)])
    result = measure_stage(prepared, 2, iterations=2, warmups=1)
    assert result['resident']['samples_ms'] == [1, 1]
    start0, end0 = calls.index(('timestamp', 0)), calls.index(('timestamp', 1))
    start1, end1 = calls.index(('timestamp', 2)), calls.index(('timestamp', 3))
    assert calls[:start0].count('clear') == 2  # warmup plus first sample.
    assert calls[end0:start1].count('clear') == 1
    assert calls[end0:start1].count('minimum') == 1
    assert calls[start0:end0].count('support') == calls[start1:end1].count('support') == 1
    assert 'clear' not in calls[start0:end0] and 'minimum' not in calls[start1:end1]


def test_cpu_cases_cover_all_new_entries_distributions_and_source_counts():
    from benchmarks.vkl_reprojection import cases
    cases = list(cases())
    assert {case['family'] for case in cases} == {'projection', 'raster', 'temporal', 'reference'}
    assert {case['variant'] for case in cases} == {'ordinary', 'collision', 'sparse', 'all_invalid', 'odd_tail'}
    assert {case['neighbors'] for case in cases if case['family'] == 'reference'} >= {1, 3}
    entries = {stage['entrypoint'] for case in cases for stage in case['graph']['stages']}
    assert entries >= {'projectPoints', 'reduceProjection', 'clearRaster', 'rasterMin', 'rasterSupport', 'finalizeRaster', 'sampleTemporal', 'packTemporal', 'finalizeReference'}
    for case in cases:
        assert set(case['reference']()) == set(case['graph']['outputs'])
        assert case['workload_identity']['sha256']


def test_exact_workload_identity_binds_array_bytes_dtype_shape_graph_and_arguments():
    from benchmarks.vkl_reprojection import cases, workload_identity
    case = next(cases(family='projection', variants=('ordinary',)))
    original = workload_identity(case, arguments={'iterations': 3})
    assert original == workload_identity(next(cases(family='projection', variants=('ordinary',))), arguments={'iterations': 3})
    case['inputs']['source_depth'].reshape(-1)[0] += 1
    assert workload_identity(case, arguments={'iterations': 3})['sha256'] != original['sha256']
    case = next(cases(family='projection', variants=('ordinary',)))
    case['graph']['stages'][1]['params']['threshold'] *= 2
    assert workload_identity(case, arguments={'iterations': 3})['sha256'] != original['sha256']
    case = next(cases(family='projection', variants=('ordinary',)))
    assert workload_identity(case, arguments={'iterations': 4})['sha256'] != original['sha256']


def test_isolated_support_postcondition_is_checked_before_any_reset():
    import numpy as np
    import pytest
    from benchmarks.reprojection_support import validate_isolated_stage
    events = []
    class Buffer:
        def __init__(self, values): self.values = np.array(values, np.uint32)
        def to_numpy(self): events.append('read_support'); return self.values
    specification = dict(entrypoint='rasterSupport', bindings={'supportBits': 'support'})
    resource = dict(elements=2, dtype='uint32', role='scratch')
    class Prepared:
        graph = dict(resources={'support': resource})
        buffers = {'support': Buffer([2, 1, 73])}
        def record(self, encoder): raise AssertionError('must not replay/reset before checking sampled support')
    expected = {'support': np.array([2, 1], np.uint32)}
    validate_isolated_stage(Prepared(), specification, expected)
    Prepared.buffers['support'] = Buffer([4, 2, 73])  # missing-reset duplicate accumulation
    with pytest.raises(AssertionError): validate_isolated_stage(Prepared(), specification, expected)
    assert events == ['read_support', 'read_support']


def test_benchmark_owner_closes_one_device_on_success_and_exception(monkeypatch):
    import sys
    import types
    import pytest
    import benchmarks.vkl_reprojection as recipe
    monkeypatch.setenv('SINTERGPU_GPU_TESTS', '1')
    events = []
    class Device:
        info = types.SimpleNamespace(limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=[65535]))
        def __init__(self, **kwargs): events.append('create')
        def close(self): events.append('close')
    fake = types.SimpleNamespace(Device=Device, DeviceType=types.SimpleNamespace(vulkan='vulkan'),
                                 SlangFloatingPointMode=types.SimpleNamespace(precise='precise'))
    monkeypatch.setitem(sys.modules, 'slangpy', fake)
    monkeypatch.setattr(recipe, 'PreparedGraph', lambda *args: object())
    monkeypatch.setattr(recipe, 'validate_outputs', lambda *args: None)
    monkeypatch.setattr(recipe, 'measure_graph', lambda *args, **kwargs: {})
    result = recipe.run(family='projection', variants=('ordinary',), max_cases=1, iterations=1, warmups=1)
    assert len(result['measurements']) == 1 and events == ['create', 'close']
    events.clear()
    def fail(*args): raise AssertionError('injected parity failure')
    monkeypatch.setattr(recipe, 'validate_outputs', fail)
    with pytest.raises(AssertionError, match='injected'):
        recipe.run(family='projection', variants=('ordinary',), max_cases=1, iterations=1, warmups=1)
    assert events == ['create', 'close']


def test_benchmark_source_identity_includes_host_oracles_and_transitive_shader_helpers():
    from benchmarks.reprojection_support import ROOT, source_identities
    sources = source_identities([ROOT / 'src/sintergpu/kernels/vkl/reprojection.slang'])
    assert set(sources) >= {
        'src/sintergpu/references/reprojection_numeric.py',
        'src/sintergpu/references/temporal_reprojection_numeric.py',
        'src/sintergpu/references/geometry_depth_numeric.py',
        'src/sintergpu/reprojection.py', 'benchmarks/reprojection_support.py',
        'benchmarks/vkl_reprojection.py', 'src/sintergpu/kernels/common/float32_divide.slang',
        'src/sintergpu/kernels/common/float32_multiply.slang'}
    assert all(len(value) == 64 for value in sources.values())


def test_case_filters_bound_default_qualification_cost_and_identity_binds_dtype_shape():
    from benchmarks.vkl_reprojection import cases, workload_identity
    cases_list = list(cases(family='reference', variants=('all_invalid',), neighbors=2, max_cases=1))
    assert len(cases_list) == 1 and cases_list[0]['neighbors'] == 2
    case = cases_list[0]
    original = workload_identity(case)['sha256']
    array = case['inputs']['source_rgb']
    case['inputs']['source_rgb'] = array.reshape(-1)
    assert workload_identity(case)['sha256'] != original
    case['inputs']['source_rgb'] = array.astype('float64')
    assert workload_identity(case)['sha256'] != original


def _assert_sampler_cast_guards(text):
    """Require every finite/range guard before any sampler integer conversion."""
    def body(name):
        start = text.index('{', text.index(name + '(')); end = start; depth = 1
        while depth:
            end += 1
            depth += (text[end] == '{') - (text[end] == '}')
        return text[start:end + 1]
    nearest = body('nearestDepth').split('int x=nearestEven(u)', 1)[0]
    bilinear = body('bilinear').split('int x=int(floor(u))', 1)[0]
    for token in ('!isfinite(u)', '!isfinite(v)', 'u<-.5', 'v<-.5', 'u>float(params.targetWidth)-.5', 'v>float(params.targetHeight)-.5'):
        assert token in nearest, 'nearest guard missing or after integer conversion: ' + token
    for token in ('!isfinite(u)', '!isfinite(v)', 'u< -1', 'v< -1', 'u>float(w)', 'v>float(h)'):
        assert token in bilinear, 'bilinear guard missing or after integer conversion: ' + token


def test_sampler_cast_guards_reject_deleted_or_moved_finite_and_range_checks():
    from pathlib import Path
    import pytest
    text = (Path(__file__).resolve().parents[1] / 'src/sintergpu/kernels/vkl/reprojection.slang').read_text()
    _assert_sampler_cast_guards(text)
    for token in ('!isfinite(u)', '!isfinite(v)', 'u<-.5', 'v<-.5', 'u>float(params.targetWidth)-.5', 'v>float(params.targetHeight)-.5', 'u< -1', 'v< -1', 'u>float(w)', 'v>float(h)'):
        with pytest.raises(AssertionError, match='guard missing or after integer conversion'):
            _assert_sampler_cast_guards(text.replace(token, 'false'))
    for guard_start, cast in [('if(!isfinite(u)||!isfinite(v)||u<-.5', '    int x=nearestEven(u),y=nearestEven(v);'),
                              ('if(u< -1||v< -1', '    int x=int(floor(u)),y=int(floor(v));'),
                              ('if(!isfinite(u)||!isfinite(v))', '    int x=int(floor(u)),y=int(floor(v));')]:
        line = next(line for line in text.splitlines() if guard_start in line)
        moved = text.replace(line + '\n', '', 1).replace(cast, cast + '\n' + line, 1)
        with pytest.raises(AssertionError, match='guard missing or after integer conversion'):
            _assert_sampler_cast_guards(moved)
