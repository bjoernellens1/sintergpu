# SPDX-License-Identifier: Apache-2.0
"""CPU-testable cooperative supervision, not a qualifying runner.

The caller must verify the exact cooperative child and keep its canonical GPU
lease on *every* quarantine. Pipe messages are process claims, not device proof.
"""
import json
import os
import signal
import subprocess
import tempfile
import time
from timing_handshake import CooperativeHandshake

class QuarantineRequired(RuntimeError):
    def __init__(self,child):super().__init__('unconfirmed child teardown; retain GPU lease and quarantine');self.child=child


def supervise(command,monitor,journal,lease_check,quarantine,*,label,pass_fds=(),timeout_s=120,poll_s=.05,popen=subprocess.Popen):
    if not command or not callable(lease_check) or not callable(quarantine) or not label or '/' in label or timeout_s<=0 or poll_s<=0:raise ValueError('supervision arguments')
    child=None;started=None;ended=None;error=None;monitor_started=False;ready=False;clean=False;returncode=None;receipt=None;quarantined=False
    handshake=CooperativeHandshake()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            lease_check();monitor.start();monitor_started=True;started=time.monotonic_ns()
            if monitor.failed.is_set():raise RuntimeError('monitor failed before child')
            env,fds=handshake.child_options(os.environ,pass_fds)
            child=popen(command,stdout=out,stderr=err,pass_fds=fds,env=env)
            handshake.parent_after_spawn();handshake.await_ready(monitor.failed);ready=True
            deadline=time.monotonic()+timeout_s
            while child.poll() is None:
                if monitor.failed.is_set():raise RuntimeError('sampled control violation')
                if time.monotonic()>=deadline:raise TimeoutError('child timeout')
                monitor.failed.wait(poll_s)
            returncode=child.returncode;ended=time.monotonic_ns()
            handshake.confirm_cleanup();clean=True
            if returncode!=0:raise RuntimeError(f'child exit {returncode}')
            if monitor.failed.is_set():raise RuntimeError('sampled control violation')
        except Exception as exc:
            error=exc
            if child is not None:
                # Never signal before the child's handler confirms readiness.
                if ready and child.poll() is None:
                    try:child.send_signal(signal.SIGUSR1);child.wait(timeout=10)
                    except Exception as stop_error:error=ExceptionGroup('operation and cooperative stop failure',[error,stop_error])
                if child.poll() is not None:
                    returncode=child.returncode
                    if not clean:
                        try:handshake.confirm_cleanup();clean=True
                        except Exception as cleanup_error:error=ExceptionGroup('operation and missing cleanup receipt',[error,cleanup_error])
                # Abnormal/ambiguous outcomes require external, independently
                # verified device quiescence before the caller releases the lease.
                try:quarantine(child,error);quarantined=True
                except Exception as quarantine_error:error=ExceptionGroup('operation and quarantine failure',[error,quarantine_error])
                else:error=ExceptionGroup('operation requires retained lease',[error,QuarantineRequired(child)])
            ended=time.monotonic_ns()
        finally:
            if monitor_started and started is not None:
                try:receipt=monitor.stop(started,ended)
                except Exception as failure:error=ExceptionGroup('child and monitor failure',[error,failure]) if error else failure
            try:lease_check()
            except Exception as failure:error=ExceptionGroup('operation and lease failure',[error,failure]) if error else failure
            try:
                out.seek(0);err.seek(0)
                journal.put(label+'.stdout',out.read());journal.put(label+'.stderr',err.read())
                data={'receipt':receipt,'samples':list(monitor.samples),'monitor_error':str(monitor.error) if monitor.error else None,
                      'returncode':returncode,'ready':ready,'cleanup_claim':clean,'started_ns':started,'ended_ns':ended,
                      'error':str(error) if error else None,'scope':'sampled unqualified evidence; peer absence and device cleanup unproven'}
                journal.put(label+'.monitor.json',(json.dumps(data,sort_keys=True)+'\n').encode())
            except Exception as failure:error=ExceptionGroup('operation and evidence failure',[error,failure]) if error else failure
            if error and child is not None and not quarantined:
                try:quarantine(child,error);quarantined=True
                except Exception as failure:error=ExceptionGroup('operation and quarantine failure',[error,failure])
                else:error=ExceptionGroup('operation requires retained lease',[error,QuarantineRequired(child)])
            handshake.close()
    if error:raise error
    return {'returncode':returncode,'monitor':receipt,'scope':'UNQUALIFIED; caller must validate complete outputs and controls'}
