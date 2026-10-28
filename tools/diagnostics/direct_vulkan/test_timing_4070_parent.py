# SPDX-License-Identifier: Apache-2.0
import pytest
import timing_4070_parent as candidate

def test_review_gate_blocks_before_source_or_device(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('unexpected source/device access')
    monkeypatch.setattr(candidate,'committed_sources',forbidden)
    with pytest.raises(RuntimeError,match='blocked pending independent review'): candidate.run(None)
