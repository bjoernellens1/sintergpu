# SPDX-License-Identifier: Apache-2.0
"""CPU-only native timestamp runner contract checks."""
import types
import pytest
from benchmarks import depth_pyramid_native_timestamp_runner as runner


def test_explicit_optin_and_closure():
    assert {'benchmarks/native/vulkan_timestamp_probe.c','benchmarks/depth_pyramid_native_timestamp_runner.py'} <= set(runner.EXTRA)
    with pytest.raises(ValueError,match='explicit --execute'):
        runner.run(types.SimpleNamespace(execute=False))


def test_device_output_requires_exact_expected_identity():
    expected={'device_uuid':'uuid','vendor_id':4318,'device_id':42,'driver_version':123,'timestamp_period_ns':1.0,'timestamp_valid_bits':64,'query_pairs':[[1,2]]*50}
    runner.validate_result(expected,'uuid',4318,42,123,50)
    for key,value in [('device_uuid','wrong'),('driver_version',124),('timestamp_period_ns',0),('timestamp_valid_bits',0),('query_pairs',[])]:
        mutated=dict(expected); mutated[key]=value
        with pytest.raises(ValueError): runner.validate_result(mutated,'uuid',4318,42,123,50)


def test_real_compiler_dependency_closure_without_gpu():
    import shutil
    from pathlib import Path
    cc=shutil.which('cc')
    if not cc: pytest.skip('C compiler unavailable')
    hashes=runner.dependency_hashes(Path(cc).resolve())
    assert str((runner.ROOT/runner.SOURCE).resolve()) in hashes
    assert any(path.endswith('/vulkan/vulkan.h') for path in hashes)
    assert all(len(value)==64 for value in hashes.values())


def test_loaded_helper_path_shadow_and_mutation_fail_closed(tmp_path,monkeypatch):
    import hashlib, types
    import benchmarks.depth_pyramid_native_timestamp_runner as runner
    path=tmp_path/'helper.py'; path.write_text('source')
    module=types.SimpleNamespace(__file__=str(path))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verify_loaded_closure',lambda ids:None)
    ids={'helper.py':hashlib.sha256(path.read_bytes()).hexdigest()}
    runner.verify_modules(ids,{'helper':module})
    path.write_text('modified')
    with pytest.raises(ValueError,match='committed closure'):
        runner.verify_modules(ids,{'helper':module})
    module.__file__='/tmp/shadow.py'
    with pytest.raises(ValueError,match='outside committed root'):
        runner.verify_modules(ids,{'helper':module})


def test_native_result_is_raw_top_of_pipe_and_not_slangpy_comparison():
    from pathlib import Path
    text=Path(runner.__file__).read_text()
    assert 'TOP_OF_PIPE' in text and 'BOTTOM_OF_PIPE' in text
    assert 'raw ticks only' in text
    assert 'no shader' in text
