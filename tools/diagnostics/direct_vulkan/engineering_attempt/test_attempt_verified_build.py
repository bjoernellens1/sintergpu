# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('build_fixture',HERE/'verified_build.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_two_actual_local_builds_and_source_tamper():
    if not m.COMPILER.is_file() or m.sha(m.COMPILER.read_bytes())!=m.COMPILER_SHA: pytest.skip('target host pinned compiler unavailable')
    source=(HERE/'point_plane.cpp').read_bytes()
    result=m.build(source)
    assert m.sha(result)==m.BINARY_SHA
    with pytest.raises(ValueError,match='source mismatch'):m.build(source+b'\n')

def test_reject_untrusted_compiler(tmp_path):
    fake=tmp_path/'compiler';fake.write_bytes(b'fake')
    with pytest.raises(ValueError,match='compiler mismatch'):m.build((HERE/'point_plane.cpp').read_bytes(),fake)
