# SPDX-License-Identifier: Apache-2.0
import os
import threading
import pytest
from timing_handshake import CooperativeHandshake

def test_ready_and_cleanup_pipe_order():
    h=CooperativeHandshake();child=os.dup(h.write_fd);h.parent_after_spawn()
    try:
        os.write(child,b'RC');os.close(child);h.await_ready(threading.Event());h.confirm_cleanup()
    finally:h.close()

def test_missing_cleanup_and_monitor_failure_before_ready():
    h=CooperativeHandshake();child=os.dup(h.write_fd);h.parent_after_spawn();failed=threading.Event();failed.set()
    try:
        with pytest.raises(RuntimeError,match='monitor failed'):h.await_ready(failed)
        os.write(child,b'R');os.close(child);h.await_ready(threading.Event())
        with pytest.raises(RuntimeError,match='cleanup'):h.confirm_cleanup()
    finally:h.close()
