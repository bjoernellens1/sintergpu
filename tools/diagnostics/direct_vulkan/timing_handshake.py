# SPDX-License-Identifier: Apache-2.0
"""Cooperative child ready/cleanup pipe. Receipt is not device attestation."""
import os
import select
import time

class CooperativeHandshake:
    def __init__(self):
        self.read_fd,self.write_fd=os.pipe2(os.O_CLOEXEC);self._closed=False
    def child_options(self,env,pass_fds):
        env=dict(env);env['SINTERGPU_COOPERATIVE_FD']=str(self.write_fd)
        return env,tuple(pass_fds)+(self.write_fd,)
    def parent_after_spawn(self):
        os.close(self.write_fd);self.write_fd=-1
    def _byte(self,timeout,failed=None):
        deadline=time.monotonic()+timeout
        while True:
            if failed is not None and failed.is_set():raise RuntimeError('monitor failed before child ready')
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('cooperative handshake timeout')
            ready,_,_=select.select([self.read_fd],[],[],min(remaining,.05))
            if ready:return os.read(self.read_fd,1)
    def await_ready(self,failed,timeout=5):
        if self._byte(timeout,failed)!=b'R':raise RuntimeError('cooperative ready missing')
    def confirm_cleanup(self,timeout=2):
        if self._byte(timeout)!=b'C' or self._byte(timeout)!=b'':raise RuntimeError('cooperative cleanup receipt missing or extra')
    def close(self):
        if not self._closed:
            os.close(self.read_fd)
            if self.write_fd>=0:os.close(self.write_fd)
            self._closed=True
