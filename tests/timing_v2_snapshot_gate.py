# SPDX-License-Identifier: Apache-2.0
"""Provisional CPU-only immutable-mount experiment; never collection-ready.

A private namespace tmpfs is populated then remounted read-only, followed by
capability drop and a fresh isolated Python child. This is not proof of Python,
compiler, loader, GPU or immutable launch bootstrap identity; no timing result
is qualified or promoted by this helper. Requires Linux user/mount namespaces.
The finder is not a security boundary: arbitrary exec, import aliases, native
code and the original checkout remain accessible.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from sintergpu.timing_closure_v2 import POLICIES

ROOT=Path(__file__).resolve().parents[1]
FAMILY='vkl.depth_convert.depthConvert'
HELPERS=('tests/timing_v2_snapshot_gate.py','tests/timing_v2_snapshot_namespace.py',
         'tests/timing_v2_snapshot_child.py','tests/timing_v2_snapshot_late.py',
         'tests/timing_v2_snapshot_allow.py')


class SnapshotGateError(RuntimeError):
    def __init__(self,phase,reason):
        self.phase=phase; self.reason=reason
        super().__init__(f'snapshot gate failed closed: {phase}/{reason}')


def run(scenario='ok'):
    if scenario not in ('ok','late','preimport','pyc','unexpected','shadow','cap_prm','cap_inh','cap_amb','cap_missing'): raise ValueError('unknown test scenario')
    names=tuple(sorted(set(POLICIES[FAMILY]['files'].values())|set(HELPERS)))
    files={}
    for name in names:
        path=ROOT
        for part in Path(name).parts:
            path=path/part
            if path.is_symlink(): raise SnapshotGateError('preflight','symlink_source')
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()): raise SnapshotGateError('preflight','untrusted_source')
        files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    command=['/usr/bin/unshare','-Urnm',sys.executable,'-I','-S','-B',str(ROOT/'tests/timing_v2_snapshot_namespace.py'),str(ROOT)]
    result=subprocess.run(command,input=json.dumps({'scenario':scenario,'files':files}),capture_output=True,text=True,
                          env={'PATH':'/usr/bin:/bin','PYTHONNOUSERSITE':'1'})
    if result.returncode:
        try: error=json.loads(result.stderr.strip())
        except (ValueError,TypeError): error={'phase':'launch','reason':'unstructured_failure'}
        if set(error)!={'phase','reason'}: error={'phase':'launch','reason':'unstructured_failure'}
        raise SnapshotGateError(error['phase'],error['reason'])
    payload=json.loads(result.stdout)
    if payload['before']!=files or payload['after']!=files: raise RuntimeError('snapshot identity mismatch')
    return payload
