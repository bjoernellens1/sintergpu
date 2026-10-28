# SPDX-License-Identifier: Apache-2.0
"""Explicit parent-owned flock and best-effort quarantine guardian handoff."""
import fcntl
import hashlib
import os
from pathlib import Path
import select
import subprocess
import sys
from timing_quarantine_guardian import verify

class QuarantineLease:
    def __init__(self,path,guardian_source,guardian_sha):
        self.path=Path(path);self.guardian=Path(guardian_source);self.guardian_sha=guardian_sha;self.fd=-1;self.process=None;self.quarantine_required=False
        if not self.path.is_absolute() or self.path.is_symlink() or not self.guardian.is_absolute() or self.guardian.is_symlink():raise ValueError('absolute non-symlink paths required')
        self.fd=os.open(self.path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(self.fd,self.path)
        except Exception:os.close(self.fd);self.fd=-1;raise
    def check(self):
        if self.fd<0:raise ValueError('lease unavailable')
        verify(self.fd,self.path)
    def start_guardian(self):
        self.check()
        if self.process is not None:raise ValueError('guardian already started')
        if hashlib.sha256(self.guardian.read_bytes()).hexdigest()!=self.guardian_sha:raise ValueError('guardian source mismatch')
        r,w=os.pipe2(os.O_CLOEXEC)
        try:
            process=subprocess.Popen([sys.executable,'-I',str(self.guardian),'--fd',str(self.fd),'--path',str(self.path),'--ready-fd',str(w)],pass_fds=(self.fd,w),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            os.close(w);w=-1
            ready,_,_=select.select([r],[],[],5)
            if not ready or os.read(r,1)!=b'Q' or process.poll() is not None:raise RuntimeError('guardian unavailable; retain parent lease')
            self.process=process
            return {'guardian_pid':process.pid,'lock_inode':os.fstat(self.fd).st_ino,'scope':'borrowed flock only; quiescence unverified'}
        finally:
            os.close(r)
            if w>=0:os.close(w)
    def quarantine(self,child,reason):
        del child,reason
        self.quarantine_required=True
        self.check()
        if self.process is None:return self.start_guardian()
        if self.process.poll() is not None:
            self.process=None
            return self.start_guardian()
        return {'guardian_pid':self.process.pid,'lock_inode':os.fstat(self.fd).st_ino,'scope':'borrowed flock only; quiescence unverified'}
    def release_normal(self):
        self.check()
        if self.quarantine_required:raise RuntimeError('quarantine requires independent resolution')
        if self.process is not None:
            if self.process.poll() is not None:raise RuntimeError('guardian exited; retain parent lease')
            self.process.terminate();self.process.wait(timeout=5);self.process=None
        os.close(self.fd);self.fd=-1
    def close(self):
        self.check()
        if self.quarantine_required and self.process is None:raise RuntimeError('quarantine unconfirmed; retain parent lease')
        if self.process is not None and self.process.poll() is not None:raise RuntimeError('guardian exited; retain parent lease')
        os.close(self.fd);self.fd=-1
