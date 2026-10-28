# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract checks for stream pilot; no GPU or dispatch."""
import hashlib
import pytest
from benchmarks import depth_pyramid_stream_pilot as pilot


def test_stream_closure_binds_probe_and_runner():
    assert {'benchmarks/depth_pyramid_stream_pilot.py','benchmarks/shaders/depth_pyramid_vkl_stream_candidate.slang','tests/depth_pyramid_stream_fixture.py'} <= set(pilot.CLOSURE)
    assert len(pilot.CLOSURE)==len(set(pilot.CLOSURE))
    assert pilot.workloads()==[{'profile':'vkl','batch':3,'height':121,'width':161}]


def test_stream_closure_rejects_mutation(monkeypatch,tmp_path):
    commit='a'*40; source=tmp_path/'probe'; source.write_bytes(b'valid')
    monkeypatch.setattr(pilot,'ROOT',tmp_path); monkeypatch.setattr(pilot,'CLOSURE',('probe',))
    monkeypatch.setattr(pilot.subprocess,'check_output',lambda *a,**kw:commit+'\n' if kw.get('text') else b'valid')
    assert pilot.committed_closure(commit)['probe']==hashlib.sha256(b'valid').hexdigest()
    source.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='differs from committed'): pilot.committed_closure(commit)


def test_no_execution_without_explicit_flag():
    import types
    with pytest.raises(ValueError,match='explicit --execute'):
        pilot.run(types.SimpleNamespace(execute=False))
