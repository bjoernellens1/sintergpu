# SPDX-License-Identifier: Apache-2.0
"""Fake timestamp diagnostics; never initializes GPU hardware."""
import contextlib
import types
import pytest
from benchmarks.depth_pyramid_timestamp_probe import analyze, collect

class Encoder:
    def __init__(self,events): self.events=events
    def write_timestamp(self,pool,index): self.events.append(('timestamp',index))
    def finish(self): return object()
class Pool:
    def __init__(self,n): self.n=n
    def get_results(self,start,count): return [10000+i*1024 for i in range(count)]
class Device:
    info=types.SimpleNamespace(timestamp_frequency=1000000)
    def __init__(self,events): self.events=events
    def create_query_pool(self,t,n): self.events.append(('pool',n)); return Pool(n)
    def create_command_encoder(self): return Encoder(self.events)
    def submit_command_buffer(self,c): self.events.append('submit')
    def wait_for_idle(self): self.events.append('wait')
class Graph:
    source_ids={'shader':'a'*64}
    def __init__(self,events): self.events=events
    def record(self,encoder): self.events.extend(['barrier','dispatch'])

@pytest.fixture
def lease(tmp_path,monkeypatch):
    import os,fcntl
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: yield module.BorrowedLease(fd)
    finally: os.close(fd)

def identity():
    return {'device_uuid':'uuid','driver':'driver','compiler_sha256':'b'*64,'source_sha256':{'shader':'a'*64},'lock_receipt_sha256':'c'*64}

def test_absolute_ticks_empty_queries_and_bounded_recording(lease):
    events=[]
    report=collect(Device(events),object(),Graph(events),identity,lambda:events.append('validate'),samples=50,lease=lease)
    assert report['empty_pairs'][0]==[10000,11024]
    assert report['pipeline_pairs'][0]==[12048,13072]
    assert report['timestamp_frequency']==1000000
    assert len(report['empty_pairs'])==len(report['pipeline_pairs'])==50
    assert events.count('validate')==2
    i=events.index(('timestamp',0))
    assert events[i:i+6]==[('timestamp',0),('timestamp',1),('timestamp',2),'barrier','dispatch',('timestamp',3)]
    result=analyze(report)
    assert result['observed_tick_step_gcd']==1024
    assert result['modulo_1024_unique']==[784]


def test_fail_closed_invalid_timestamps_or_identity(lease):
    with pytest.raises(ValueError,match='50'):
        collect(Device([]),object(),Graph([]),identity,lambda:None,samples=49,lease=lease)
    states=iter([identity(),{**identity(),'driver':'changed'}])
    with pytest.raises(ValueError,match='identity changed'):
        collect(Device([]),object(),Graph([]),lambda:next(states),lambda:None,samples=50,lease=lease)
    with pytest.raises(ValueError,match='timestamp'):
        analyze({'timestamp_frequency':1,'empty_pairs':[[4,3]],'pipeline_pairs':[[4,5]]})


def test_corrupted_query_and_source_fail_before_reporting(lease):
    events=[]; wrong={**identity(),'source_sha256':{'shader':'0'*64}}
    with pytest.raises(ValueError,match='source identity'):
        collect(Device(events),object(),Graph(events),lambda:wrong,lambda:None,samples=50,lease=lease)
    assert events==[]
    class BadPool(Pool):
        def get_results(self,start,count): return [1.5]*count
    d=Device(events); d.create_query_pool=lambda kind,count:BadPool(count)
    with pytest.raises(ValueError,match='integers'):
        collect(d,object(),Graph(events),identity,lambda:None,samples=50,lease=lease)


def test_all_zero_tick_deltas_remain_unknown():
    report={'timestamp_frequency':100,'empty_pairs':[[10,10]],'pipeline_pairs':[[10,10]]}
    result=analyze(report)
    assert result['observed_tick_step_gcd'] is None
    assert result['pipeline_duration_ticks']==[0]
    assert 'do not establish timer resolution' in result['scope']


def test_missing_or_unowned_lease_rejected(tmp_path,monkeypatch):
    import os
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    with pytest.raises(ValueError,match='lease'):
        collect(Device([]),object(),Graph([]),identity,lambda:None,samples=50,lease=None)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        lease=module.BorrowedLease(fd)
        with pytest.raises(ValueError,match='lock'):
            collect(Device([]),object(),Graph([]),identity,lambda:None,samples=50,lease=lease)
    finally: os.close(fd)


def test_caller_holds_lock_through_device_teardown(tmp_path,monkeypatch):
    import os, fcntl, subprocess, sys
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    competitor='import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDWR);\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(75)\nsys.exit(0)'
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        lease=module.BorrowedLease(fd)
        # Caller has created its device while the lease is held.
        collect(Device([]),object(),Graph([]),identity,lambda:None,samples=50,lease=lease)
        assert subprocess.run([sys.executable,'-c',competitor,str(path)]).returncode==75
        # Teardown happens before the caller releases the same lock.
        device_closed=True
        assert device_closed and subprocess.run([sys.executable,'-c',competitor,str(path)]).returncode==75
    finally: os.close(fd)
    assert subprocess.run([sys.executable,'-c',competitor,str(path)]).returncode==0


def test_foreign_descriptor_does_not_count_as_owned_lease(tmp_path,monkeypatch):
    import os,fcntl
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    held=os.open(path,os.O_CREAT|os.O_RDWR,0o600); other=os.open(path,os.O_RDWR)
    fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError,match='exclusive'):
            module.BorrowedLease(other).verify()
    finally:
        os.close(other); os.close(held)


def test_shared_lease_never_upgraded(tmp_path,monkeypatch):
    import os,fcntl
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError,match='exclusive'):
            module.BorrowedLease(fd).verify()
        # A second shared lock is possible only if verification did not upgrade it.
        other=os.open(path,os.O_RDWR)
        try: fcntl.flock(other,fcntl.LOCK_SH|fcntl.LOCK_NB)
        finally: os.close(other)
    finally: os.close(fd)


def test_unlocked_fd_rejected_even_after_foreign_lock_released(tmp_path,monkeypatch):
    import os,fcntl
    import benchmarks.depth_pyramid_timestamp_probe as module
    path=tmp_path/'gpu.lock'; monkeypatch.setattr(module,'LOCK',path)
    owner=os.open(path,os.O_CREAT|os.O_RDWR,0o600); other=os.open(path,os.O_RDWR)
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError): module.BorrowedLease(other).verify()
        fcntl.flock(owner,fcntl.LOCK_UN)
        with pytest.raises(ValueError): module.BorrowedLease(other).verify()
        # Verification must not acquire an exclusive lock on the unlocked fd.
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally: os.close(other); os.close(owner)


def test_unavailable_fdinfo_fails_closed(lease,monkeypatch):
    from pathlib import Path
    original=Path.read_text
    def denied(path,*args,**kwargs):
        if str(path).startswith('/proc/self/fdinfo/'):
            raise OSError('fdinfo unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',denied)
    with pytest.raises(ValueError,match='unavailable'):
        lease.verify()
