# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
import sys
from pathlib import Path
import os
import pytest
from timing_quarantine_lease import QuarantineLease

GUARDIAN=Path(__file__).with_name('timing_quarantine_guardian.py').resolve()

def competitor(path):
    script='import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDWR);\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(11)\nsys.exit(0)'
    return subprocess.run([sys.executable,'-c',script,str(path)],check=False).returncode

def test_guardian_keeps_real_flock_after_parent_closes(tmp_path):
    lock=tmp_path/'gpu.lock';digest=hashlib.sha256(GUARDIAN.read_bytes()).hexdigest();lease=QuarantineLease(lock,GUARDIAN,digest)
    assert competitor(lock)==11
    receipt=lease.quarantine(None,'simulated unconfirmed exit');assert receipt['guardian_pid']>0
    process=lease.process
    try:
        lease.close();assert competitor(lock)==11
    finally:process.terminate();process.wait(timeout=5)
    assert competitor(lock)==0

def test_bad_guardian_identity_keeps_parent_lease(tmp_path):
    lock=tmp_path/'gpu.lock';lease=QuarantineLease(lock,GUARDIAN,'0'*64)
    try:
        with pytest.raises(ValueError,match='guardian source mismatch'):lease.quarantine(None,'x')
        assert competitor(lock)==11
        with pytest.raises(RuntimeError,match='quarantine unconfirmed'):lease.close()
    finally:
        # Test teardown only: no GPU exists in this synthetic case.
        os.close(lease.fd);lease.fd=-1

def test_prestarted_guardian_allows_controlled_normal_release(tmp_path):
    lock=tmp_path/'gpu.lock';lease=QuarantineLease(lock,GUARDIAN,hashlib.sha256(GUARDIAN.read_bytes()).hexdigest())
    lease.start_guardian();assert competitor(lock)==11
    lease.release_normal();assert competitor(lock)==0

def test_quarantine_reuses_prestarted_guardian_and_blocks_normal_release(tmp_path):
    lock=tmp_path/'gpu.lock';lease=QuarantineLease(lock,GUARDIAN,hashlib.sha256(GUARDIAN.read_bytes()).hexdigest())
    first=lease.start_guardian();second=lease.quarantine(None,'uncertain');assert first['guardian_pid']==second['guardian_pid']
    process=lease.process
    try:
        with pytest.raises(RuntimeError,match='independent resolution'):lease.release_normal()
        lease.close();assert competitor(lock)==11
    finally:process.terminate();process.wait(timeout=5)

def test_dead_guardian_is_replaced_before_parent_can_close(tmp_path):
    lock=tmp_path/'gpu.lock';lease=QuarantineLease(lock,GUARDIAN,hashlib.sha256(GUARDIAN.read_bytes()).hexdigest())
    lease.start_guardian();old=lease.process;old.terminate();old.wait(timeout=5)
    try:
        lease.quarantine(None,'guardian died');assert lease.process.pid!=old.pid
        process=lease.process;lease.close();assert competitor(lock)==11
    finally:
        if lease.process is not None and lease.process.poll() is None:lease.process.terminate();lease.process.wait(timeout=5)
