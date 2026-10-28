"""Exporter tests use synthetic CPU tensors; no Torch dependency or GPU mocks."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import export_reprojection_fixtures as exporter


def test_cli_requires_output_path():
    with pytest.raises(SystemExit):
        exporter.main([])


def test_wrong_torch_build_fails_before_thread_configuration(tmp_path):
    torch = SimpleNamespace(__version__='2.14.1+cpu')
    with pytest.raises(ValueError, match='2.14.1\\+rocm7.2'):
        exporter.export_archive(tmp_path / 'archive', torch_module=torch)
    assert not (tmp_path / 'archive').exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / 'archive'
    destination.mkdir()
    (destination / 'keep').write_text('preserve')
    with pytest.raises(FileExistsError):
        exporter.export_archive(destination, torch_module=SimpleNamespace())
    assert (destination / 'keep').read_text() == 'preserve'


class _CpuTensor:
    def __init__(self, array):
        self.array = array
        self.device = SimpleNamespace(type='cpu')

    def detach(self):
        return self

    def numpy(self):
        return self.array


def _synthetic_runtime():
    return dict(torch_version=exporter.TORCH_VERSION, execution_device='cpu',
                torch_config='synthetic CPU plumbing test, not canonical evidence',
                python='synthetic test runtime', numpy_version=np.__version__, cpu={'architecture': 'synthetic'},
                threads={'intraop': 1, 'interop': 1},
                binaries_sha256={'synthetic': '0' * 64})


@pytest.fixture
def synthetic_cpu(monkeypatch):
    """Synthetic tensor adapter exercises export plumbing, not donor numerics."""
    import importlib
    torch = SimpleNamespace(from_numpy=_CpuTensor)
    monkeypatch.setattr(exporter, '_runtime_profile', lambda value: _synthetic_runtime())

    def module(name, root):
        numeric = importlib.import_module('sintergpu.references.' + (
            'reprojection_numeric' if name == 'reproject' else 'temporal_reprojection_numeric'))

        def call(function):
            def oracle(**kwargs):
                arrays = {key: value.array if isinstance(value, _CpuTensor) else value for key, value in kwargs.items()}
                return {key: _CpuTensor(value) for key, value in getattr(numeric, function)(**arrays).items()}
            return oracle

        names = ('reproject_rgbd', 'rasterize_reprojection') if name == 'reproject' else ('temporal_reproject_rgb', 'temporal_rgbd_reference')
        return SimpleNamespace(**{name: call(name) for name in names})

    monkeypatch.setattr(exporter, '_frozen_module', module)
    return torch


def test_complete_synthetic_cpu_archive_round_trip(tmp_path, synthetic_cpu):
    from benchmarks.reprojection_fixture_archive import load_fixture
    from benchmarks.reprojection_fixtures import payloads

    destination = tmp_path / 'archive'
    manifest = exporter.export_archive(destination, torch_module=synthetic_cpu)
    assert manifest['counts'] == dict(cases=63, states=113)
    assert len(manifest['records']) == 113
    assert len({record['identity']['sha256'] for record in manifest['records']}) == 113
    for payload in payloads():
        outputs = load_fixture(payload, destination)
        exporter._validate_outputs(outputs, exporter.numpy_outputs(payload))
    assert len(list(destination.glob('*.npz'))) == 113
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_comparison_leaves_no_archive_or_staging(tmp_path, synthetic_cpu, monkeypatch):
    monkeypatch.setattr(exporter, '_donor_outputs', lambda *args: {'wrong': np.zeros(1, np.float32)})
    with pytest.raises(RuntimeError, match='iteration=0') as error:
        exporter.export_archive(tmp_path / 'archive', torch_module=synthetic_cpu)
    assert isinstance(error.value.__cause__, ValueError)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('reference,actual', [
    (np.array([1], np.uint32), np.array([2], np.uint32)),
    (np.array([True]), np.array([False])),
    (np.array([1.], np.float32), np.array([1.001], np.float32)),
])
def test_export_comparison_rejects_numerical_drift(reference, actual):
    with pytest.raises(AssertionError):
        exporter._validate_outputs({'out': actual}, {'out': reference})


def test_runtime_binds_actual_binaries_and_distribution(tmp_path, monkeypatch):
    files = {}
    for name in ('python', 'torch/_C.so', 'torch/lib/libtorch_cpu.so',
                 'torch/lib/libtorch_python.so', 'torch/lib/libc10.so',
                 'torch-2.14.1.dist-info/METADATA', 'torch-2.14.1.dist-info/RECORD'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = path
    state = {}
    torch = SimpleNamespace(__version__=exporter.TORCH_VERSION,
                            __file__=str(tmp_path / 'torch/__init__.py'),
                            _C=SimpleNamespace(__file__=str(files['torch/_C.so'])),
                            __config__=SimpleNamespace(show=lambda: 'synthetic build config'),
                            set_num_threads=lambda value: state.update(intraop=value),
                            set_num_interop_threads=lambda value: state.update(interop=value),
                            get_num_threads=lambda: state['intraop'],
                            get_num_interop_threads=lambda: state['interop'])
    metadata = [name for name in files if '.dist-info/' in name]
    distribution = SimpleNamespace(files=metadata, locate_file=lambda path: tmp_path / path)
    monkeypatch.setattr(exporter.importlib.metadata, 'distribution', lambda name: distribution)
    monkeypatch.setattr(exporter.sys, 'executable', str(files['python']))
    runtime = exporter._runtime_profile(torch)
    assert runtime['threads'] == {'intraop': 1, 'interop': 1}
    assert len(runtime['binaries_sha256']) == 7
    assert set(runtime['binaries_sha256'].values()) == {exporter._file_sha(path) for path in files.values()}
    assert 'flags' in runtime['cpu']['proc_cpuinfo'] or 'Features' in runtime['cpu']['proc_cpuinfo']


def test_atomic_publish_refuses_concurrent_destination(tmp_path):
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'new').write_text('new')
    destination = tmp_path / 'archive'
    destination.mkdir()
    (destination / 'keep').write_text('preserve')
    with pytest.raises(FileExistsError):
        exporter._publish(staging, destination)
    assert (destination / 'keep').read_text() == 'preserve'
    assert not (destination / 'new').exists()
    assert (staging / 'new').is_file()


def test_donor_receives_copied_cpu_tensor_inputs(monkeypatch):
    original = np.array([1.], np.float32)
    payload = dict(oracle_module='reproject', oracle_function='reproject_rgbd',
                   oracle_kwargs={'source_depth': original, 'target_hw': (1, 1)})

    def oracle(**kwargs):
        assert kwargs['target_hw'] == (1, 1)
        kwargs['source_depth'].array.fill(4)
        return {'out': kwargs['source_depth']}

    monkeypatch.setattr(exporter, '_frozen_module', lambda *args: SimpleNamespace(reproject_rgbd=oracle))
    outputs = exporter._donor_outputs(SimpleNamespace(from_numpy=_CpuTensor), payload, exporter.ROOT)
    np.testing.assert_array_equal(original, [1])
    np.testing.assert_array_equal(outputs['out'], [4])


def test_gpu_preflight_checks_all_states_without_device_or_torch(monkeypatch):
    import builtins
    import runpy
    from pathlib import Path

    original_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'slangpy'):
            raise AssertionError('preflight imported GPU/Torch runtime: ' + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', cpu_only_import)
    gpu = runpy.run_path(str(Path(__file__).with_name('test_vkl_reprojection_gpu.py')))
    validated = []

    def load(payload, directory):
        validated.append((payload['case_id'], payload['iteration']))
        return {'synthetic': np.zeros(1, np.float32)}

    preflight = gpu['canonical_outputs'].__wrapped__
    monkeypatch.setitem(preflight.__globals__, 'load_fixture', load)
    outputs = preflight()
    assert len(outputs) == 63
    assert len(validated) == len(set(validated)) == 113
    assert 'canonical_outputs' in gpu['device'].__wrapped__.__code__.co_varnames

    def missing(*args):
        raise FileNotFoundError('missing canonical fixture')

    monkeypatch.setitem(preflight.__globals__, 'load_fixture', missing)
    with pytest.raises(FileNotFoundError, match='canonical fixture'):
        preflight()
