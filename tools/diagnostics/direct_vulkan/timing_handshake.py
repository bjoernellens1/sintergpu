# SPDX-License-Identifier: Apache-2.0
"""Cooperative child ready/cleanup pipe. Receipt is not device attestation."""
import os
import select
import time

class CooperativeHandshake:
    def __init__(self):
        self.read_fd,self.write_fd=os.pipe2(os.O_CLOEXEC);self.gate_read,self.gate_write=os.pipe2(os.O_CLOEXEC);self._closed=False
    def child_options(self,env,pass_fds):
        env=dict(env);env['SINTERGPU_COOPERATIVE_FD']=str(self.write_fd);env['SINTERGPU_COOPERATIVE_GATE_FD']=str(self.gate_read)
        return env,tuple(pass_fds)+(self.write_fd,self.gate_read)
    def parent_after_spawn(self):
        os.close(self.write_fd);self.write_fd=-1;os.close(self.gate_read);self.gate_read=-1
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
    def release(self):
        if os.write(self.gate_write,b'G')!=1:raise RuntimeError('cooperative gate failed')
        os.close(self.gate_write);self.gate_write=-1
    def cancel_gate(self):
        if self.gate_write>=0:os.close(self.gate_write);self.gate_write=-1
    def confirm_cleanup(self,timeout=2):
        if self._byte(timeout)!=b'C' or self._byte(timeout)!=b'':raise RuntimeError('cooperative cleanup receipt missing or extra')
    def close(self):
        if not self._closed:
            os.close(self.read_fd)
            if self.write_fd>=0:os.close(self.write_fd)
            if self.gate_read>=0:os.close(self.gate_read)
            if self.gate_write>=0:os.close(self.gate_write)
            self._closed=True
