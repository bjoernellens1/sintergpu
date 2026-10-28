# SPDX-License-Identifier: Apache-2.0
"""Raw GPU timestamp diagnostic; no optimization or qualification claim.

Caller acquires the canonical lock before device creation and holds it through
device teardown; this collector verifies a borrowed lease without acquiring it.
Device and source receipts remain caller assertions. GPU use requires separate review.
"""
import copy
import math
import numbers
import os
from pathlib import Path
import re
from functools import reduce
LOCK=Path('/tmp/sintergpu-gpu.lock')


class BorrowedLease:
    """Borrow an already exclusively locked, caller-owned file descriptor."""
    def __init__(self,fd):
        self.fd=fd; self.pid=os.getpid()

    def verify(self):
        if self.pid!=os.getpid(): raise ValueError('lease belongs to another process')
        try:
            actual=os.fstat(self.fd); expected=LOCK.stat()
            if LOCK.is_symlink() or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
                raise ValueError('lease lock file mismatch')
            info=Path('/proc/self/fdinfo').joinpath(str(self.fd)).read_text()
        except (OSError,ValueError) as exc:
            raise ValueError('invalid or unavailable lease lock descriptor') from exc
        locks=re.findall(r'^lock:\s*\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*$',info,re.MULTILINE)
        # fdinfo may expose the backing mount's device number rather than the
        # overlay device reported by fstat; inode and canonical fstat still match.
        owned=[(int(pid),int(inode)) for pid,major,minor,inode in locks]
        expected_lock=(self.pid,actual.st_ino)
        if expected_lock not in owned:
            raise ValueError('exclusive lease lock not held by this descriptor')



def _identity(value,graph):
    required={'device_uuid','driver','compiler_sha256','source_sha256','lock_receipt_sha256'}
    if not isinstance(value,dict) or not required<=set(value): raise ValueError('identity receipt incomplete')
    if any(not isinstance(value[k],str) or not value[k] for k in ('device_uuid','driver')): raise ValueError('device identity required')
    for key in ('compiler_sha256','lock_receipt_sha256'):
        sha=value[key]
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('identity SHA required')
    if not isinstance(value['source_sha256'],dict) or not value['source_sha256'] or any(value['source_sha256'].get(k)!=v for k,v in graph.source_ids.items()):
        raise ValueError('graph source identity mismatch')


def collect(device,timestamp_type,graph,identity,validate,*,samples=50,lease=None):
    if type(samples) is not int or not 50<=samples<=1024: raise ValueError('50 to 1024 samples required')
    if not callable(validate): raise ValueError('correctness validation required')
    if not isinstance(lease,BorrowedLease): raise ValueError('borrowed lease required')
    lease.verify()
    observed=copy.deepcopy(identity()); _identity(observed,graph)
    frequency=int(device.info.timestamp_frequency)
    if frequency<=0: raise ValueError('timestamp frequency unavailable')
    validate()
    pool=device.create_query_pool(timestamp_type,4*samples)
    encoder=device.create_command_encoder()
    for i in range(samples):
        encoder.write_timestamp(pool,4*i); encoder.write_timestamp(pool,4*i+1)
        encoder.write_timestamp(pool,4*i+2); graph.record(encoder); encoder.write_timestamp(pool,4*i+3)
    device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
    ticks=list(pool.get_results(0,4*samples))
    if len(ticks)!=4*samples: raise ValueError('timestamp count mismatch')
    if any(isinstance(t,bool) or not isinstance(t,numbers.Integral) for t in ticks): raise ValueError('timestamp ticks must be integers')
    ticks=[int(t) for t in ticks]
    if any(t<0 for t in ticks) or any(b<a for a,b in zip(ticks,ticks[1:])): raise ValueError('timestamp ordering invalid')
    validate()
    if identity()!=observed: raise ValueError('identity changed')
    lease.verify()
    report={'timestamp_frequency':frequency,'empty_pairs':[[ticks[4*i],ticks[4*i+1]] for i in range(samples)],
            'pipeline_pairs':[[ticks[4*i+2],ticks[4*i+3]] for i in range(samples)],
            'identity_unverified':observed,'lock':str(LOCK),
            'sequence':'One encoder; each adjacent empty query pair precedes one explicitly barriered resident graph query pair. No reset, upload or readback is recorded inside query intervals.',
            'scope':'Diagnostic tick observations only; caller device/source/control assertions remain unverified. Validation occurs before and after the batch only, not per sample. Repeated graph inputs and buffers are reused. No inference of timer cause, speed, or qualification.'}
    return report


def analyze(report):
    frequency=report.get('timestamp_frequency')
    if type(frequency) is not int or frequency<=0: raise ValueError('invalid timestamp frequency')
    empty=report.get('empty_pairs'); pipeline=report.get('pipeline_pairs')
    if not isinstance(empty,list) or not isinstance(pipeline,list) or not empty or len(empty)!=len(pipeline): raise ValueError('invalid timestamp pairs')
    ordered=[]
    for a,b in zip(empty,pipeline):
        for pair in (a,b):
            if not isinstance(pair,list) or len(pair)!=2 or any(type(t) is not int or t<0 for t in pair) or pair[1]<pair[0]: raise ValueError('invalid timestamp pair')
            ordered.extend(pair)
    if any(b<a for a,b in zip(ordered,ordered[1:])): raise ValueError('invalid timestamp ordering')
    steps=[b-a for a,b in zip(ordered,ordered[1:]) if b>a]
    return {'samples':len(empty),'timestamp_frequency':frequency,
            'observed_tick_step_gcd':reduce(math.gcd,steps) if steps else None,
            'modulo_1024_unique':sorted({t%1024 for t in ordered}),
            'empty_duration_ticks':[b-a for a,b in empty],
            'pipeline_duration_ticks':[b-a for a,b in pipeline],
            'scope':'Observed grid and remainders only; they do not establish timer resolution or root cause.'}
