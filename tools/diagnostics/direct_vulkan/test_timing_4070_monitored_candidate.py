# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
import timing_4070_monitored_candidate as candidate
import timing_4070_monitored_parent as blocked

def test_explicit_consent_and_committed_closure_before_io(monkeypatch):
    monkeypatch.setattr(candidate,'committed_sources',lambda *a:pytest.fail('unexpected source access'))
    with pytest.raises(ValueError,match='explicit execute required'):candidate.run(SimpleNamespace(execute=False))
    with pytest.raises(RuntimeError,match='blocked pending independent review'):blocked.run(None)
    assert 'timing_4070_monitored_candidate.py' in candidate.LOCAL
    assert candidate.NATIVE_SHA=='9066c395018bc11c83b8eb3614875051d83fc229c68077b5a7495eb0391f1f49'
