"""CPU-only provenance snapshot guards; no runtime or GPU initialization."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PATH = Path(__file__).resolve().parents[1] / 'diagnostics/rhi_build_snapshot.py'
spec = importlib.util.spec_from_file_location('rhi_build_snapshot', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def lab_fixture(tmp_path):
    lab = tmp_path / 'lab'
    for name in module.ROOTS:
        (lab / name).mkdir(parents=True, exist_ok=True)
    for name in module.INPUT_FILES + module.OUTPUT_FILES:
        path = lab / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    extension = lab / 'slangpy-source/slangpy/slangpy_ext.test.so'
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(b'extension')
    tool = tmp_path / 'tool'; tool.write_bytes(b'tool'); tool.chmod(0o755)
    (lab / 'build/CMakeCache.txt').write_text(''.join(f'{name}:FILEPATH={tool}\n' for name in module.TOOLS))
    return lab


def test_snapshot_tracks_source_tools_binary_and_external_file_links(tmp_path):
    lab = lab_fixture(tmp_path)
    external = tmp_path / 'sdk-lib'; external.write_bytes(b'sdk')
    (lab / 'slang-sdk/external.so').symlink_to(external)
    before = module.snapshot(lab, "post")
    assert before == module.snapshot(lab, "post")
    for path in (lab / 'rhi-source/source.cpp', lab / 'build/libsgl.so', external, tmp_path / 'tool'):
        existed = path.exists(); old = path.read_bytes() if existed else None
        path.write_bytes(b'changed')
        assert module.snapshot(lab, "post") != before
        if existed: path.write_bytes(old)
        else: path.unlink()
    assert module.snapshot(lab, "post") == before


def test_missing_and_external_directory_link_fail_closed(tmp_path):
    lab = lab_fixture(tmp_path)
    (lab / 'rhi-source/escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='directory link outside'):
        module.snapshot(lab, "post")
    (lab / 'rhi-source/escape').unlink()
    (lab / 'build/libsgl.so').unlink()
    with pytest.raises(ValueError, match='missing required'):
        module.snapshot(lab, "post")


def test_capture_check_and_no_overwrite(tmp_path, monkeypatch):
    lab = lab_fixture(tmp_path); manifest = tmp_path / 'receipt.json'
    monkeypatch.setattr(sys, 'argv', ['snapshot', 'capture', '--lab', str(lab), '--phase', 'post', '--manifest', str(manifest)])
    module.main(); original = json.loads(manifest.read_text())
    with pytest.raises(FileExistsError): module.main()
    assert json.loads(manifest.read_text()) == original
    monkeypatch.setattr(sys, 'argv', ['snapshot', 'check', '--lab', str(lab), '--phase', 'post', '--manifest', str(manifest)])
    module.main()
    (lab / 'probe.cpp').write_text('changed')
    with pytest.raises(SystemExit, match='snapshot mismatch'): module.main()


def test_clean_prebuild_capture_then_post_compare_detects_changes(tmp_path, monkeypatch):
    lab = lab_fixture(tmp_path)
    for name in module.OUTPUT_FILES:
        (lab / name).unlink()
    (lab / 'slangpy-source/slangpy/slangpy_ext.test.so').unlink()
    pre = module.snapshot(lab, 'pre')
    assert pre['outputs'] == {}
    with pytest.raises(ValueError, match='missing required build output'):
        module.snapshot(lab, 'post')
    for name in module.OUTPUT_FILES:
        (lab / name).write_bytes(name.encode())
    (lab / 'slangpy-source/slangpy/slangpy_ext.test.so').write_bytes(b'extension')
    post = module.snapshot(lab, 'post')
    assert pre['inputs'] == post['inputs']
    pre_file = tmp_path / 'pre.json'; post_file = tmp_path / 'post.json'
    pre_file.write_text(json.dumps(pre)); post_file.write_text(json.dumps(post))
    monkeypatch.setattr(sys, 'argv', ['snapshot', 'compare-inputs', '--manifest', str(pre_file), '--post-manifest', str(post_file)])
    module.main()
    (lab / 'probe.cpp').write_bytes(b'changed')
    post_file.write_text(json.dumps(module.snapshot(lab, 'post')))
    with pytest.raises(SystemExit, match='inputs mismatch'):
        module.main()


def test_identical_bytes_symlink_substitution_rejected(tmp_path):
    lab = lab_fixture(tmp_path)
    before = module.snapshot(lab, 'post')
    path = lab / 'build/libsgl.so'
    external = tmp_path / 'identical.so'; external.write_bytes(path.read_bytes())
    path.unlink(); path.symlink_to(external)
    assert module.snapshot(lab, 'post')['outputs']['build/libsgl.so'] != before['outputs']['build/libsgl.so']
    input_path = lab / 'probe.cpp'
    replacement = tmp_path / 'same.cpp'; replacement.write_bytes(input_path.read_bytes())
    input_path.unlink(); input_path.symlink_to(replacement)
    assert module.snapshot(lab, 'pre')['inputs']['file:probe.cpp'] != before['inputs']['file:probe.cpp']


@pytest.mark.parametrize('mutation', ['empty', 'missing-root', 'missing-tool', 'missing-output',
                                     'bad-hash', 'bad-scope', 'missing-tree-file'])
def test_compare_rejects_malformed_or_incomplete_manifests(tmp_path, monkeypatch, mutation):
    lab = lab_fixture(tmp_path)
    (lab / 'rhi-source/source.cpp').write_text('source')
    pre = module.snapshot(lab, 'pre'); post = module.snapshot(lab, 'post')
    if mutation == 'empty':
        pre = {'schema_version': 2, 'phase': 'pre', 'lab': str(lab), 'scope': module.SCOPE, 'inputs': {'fake': 1}, 'outputs': {}}
        post = dict(pre, phase='post', outputs={'fake': 1})
    elif mutation == 'missing-root':
        pre['inputs'].pop('root:rhi-source'); post['inputs'].pop('root:rhi-source')
    elif mutation == 'missing-tool':
        pre['inputs'].pop('tool:CMAKE_C_COMPILER'); post['inputs'].pop('tool:CMAKE_C_COMPILER')
    elif mutation == 'missing-output':
        post['outputs'].pop('build/libsgl.so')
    elif mutation == 'bad-hash':
        pre['inputs']['file:probe.cpp']['sha256'] = 'bad'; post['inputs']['file:probe.cpp']['sha256'] = 'bad'
    elif mutation == 'bad-scope':
        pre['scope'] = 'trusted'
    else:
        pre['inputs'].pop('rhi-source/source.cpp'); post['inputs'].pop('rhi-source/source.cpp')
    a = tmp_path / 'pre.json'; b = tmp_path / 'post.json'
    a.write_text(json.dumps(pre)); b.write_text(json.dumps(post))
    monkeypatch.setattr(sys, 'argv', ['snapshot', 'compare-inputs', '--manifest', str(a), '--post-manifest', str(b)])
    with pytest.raises(SystemExit, match='invalid manifest'):
        module.main()
