from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location('import_sources', Path(__file__).resolve().parents[1] / 'tools/import_sources.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_preflight_refuses_edits_without_partial_writes(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    a.write_bytes(b'original')
    b.write_bytes(b'edited')
    import hashlib
    prior = {str(p.relative_to(tmp_path)): {'imported_sha256': hashlib.sha256(b'original').hexdigest(), 'changes': []} for p in (a,b)}
    with pytest.raises(RuntimeError, match='modified'):
        mod.write_imports(tmp_path, {a: b'new', b: b'new'}, prior)
    assert a.read_bytes() == b'original'
    assert b.read_bytes() == b'edited'


def test_preflight_preserves_adapted_and_untracked_destinations(tmp_path):
    import hashlib
    p = tmp_path/'a'
    p.write_bytes(b'adapted')
    prior = {'a': {'imported_sha256': hashlib.sha256(b'adapted').hexdigest(), 'changes': ['ABI adaptation']}}
    with pytest.raises(RuntimeError, match='adapted'):
        mod.write_imports(tmp_path, {p: b'original'}, prior)
    with pytest.raises(RuntimeError, match='untracked'):
        mod.write_imports(tmp_path, {p: b'original'}, {})
    mod.write_imports(tmp_path, {p: b'adapted'}, {})


def test_preflight_reimports_unchanged_and_creates_new(tmp_path):
    import hashlib
    p, q = tmp_path/'a', tmp_path/'nested/b'
    p.write_bytes(b'original')
    prior = {'a': {'imported_sha256': hashlib.sha256(b'original').hexdigest(), 'changes': []}}
    mod.write_imports(tmp_path, {p: b'new', q: b'created'}, prior)
    assert p.read_bytes() == b'new'
    assert q.read_bytes() == b'created'
