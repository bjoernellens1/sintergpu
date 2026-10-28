# SPDX-License-Identifier: Apache-2.0
"""Provisional NumPy-only real CPU workload smoke; never collection-ready.

This experiment copies NumPy source/native files to a private read-only tmpfs.
The Python executable, standard library, dynamic linker and system libraries
are external and not attested. Import hooks are not a hostile-code sandbox.
No Torch donor, compiler, GPU or measurement identity is established.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from sintergpu.timing_closure_v2 import POLICIES

ROOT=Path(__file__).resolve().parents[1]
FAMILY='vkl.depth_convert.depthConvert'
HELPERS=('tests/timing_v2_workload_gate.py','tests/timing_v2_workload_namespace.py','tests/timing_v2_workload_child.py')

class SmokeError(RuntimeError):
    def __init__(self,phase,reason): self.phase=phase;self.reason=reason;super().__init__(f'{phase}/{reason}')


def _regular(root,name):
    path=root
    for part in Path(name).parts:
        path=path/part
        if path.is_symlink(): raise SmokeError('preflight','symlink')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()): raise SmokeError('preflight','source_missing')
    return path


def run(scenario='ok'):
    if scenario not in ('ok','preimport','shadow','late','unexpected','pyc','numpy_source_tamper','numpy_native_tamper','source_drift','source_side_effect','mutate_nonfirst'): raise ValueError('unknown scenario')
    spec=importlib.util.find_spec('numpy')
    if spec is None or not spec.origin: raise SmokeError('preflight','numpy_unavailable')
    numpy_root=Path(spec.origin).resolve().parent; site=numpy_root.parent
    libs=site/'numpy.libs'
    if numpy_root.name!='numpy' or not libs.is_dir() or numpy_root.is_symlink() or libs.is_symlink(): raise SmokeError('preflight','numpy_layout_unavailable')
    source_names=set(POLICIES[FAMILY]['files'].values())|set(HELPERS)
    sources={name:hashlib.sha256(_regular(ROOT,name).read_bytes()).hexdigest() for name in sorted(source_names)}
    external={}
    for base in (numpy_root,libs):
        for path in sorted(base.rglob('*')):
            if path.is_symlink(): raise SmokeError('preflight','numpy_symlink')
            if not path.is_file() or path.suffix=='.pyc' or '__pycache__' in path.parts: continue
            relative=path.relative_to(site).as_posix()
            external[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not external: raise SmokeError('preflight','numpy_empty')
    payload={'scenario':scenario,'sources':sources,'external':external,'numpy_site':str(site),
             'allowed':POLICIES[FAMILY]['required_modules']}
    cmd=['/usr/bin/unshare','-Urnm',sys.executable,'-I','-S','-B',str(ROOT/'tests/timing_v2_workload_namespace.py'),str(ROOT)]
    result=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,env={'PATH':'/usr/bin:/bin','PYTHONNOUSERSITE':'1'})
    if result.returncode:
        try: error=json.loads(result.stderr.strip())
        except (ValueError,TypeError): error={'phase':'launch','reason':'unstructured_failure'}
        raise SmokeError(error.get('phase','launch'),error.get('reason','unstructured_failure'))
    out=json.loads(result.stdout)
    if out['source_before']!=sources or out['source_after']!=sources or out['numpy_before']!=external or out['numpy_after']!=external: raise SmokeError('verify','hash_mismatch')
    out['bootstrap_unverified']={'source':'tests/timing_v2_workload_namespace.py','sha256_observed':sources['tests/timing_v2_workload_namespace.py']}
    out['runtime_executable_unverified']={'path':str(Path(sys.executable).resolve()),'sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()}
    out['status']='provisional_cpu_smoke_not_measurement'
    return out
