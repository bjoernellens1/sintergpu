# SPDX-License-Identifier: Apache-2.0
"""Canonical trial preregistration; caller must authenticate source and runtime."""
import hashlib
import json
import os
import re
from pathlib import Path
from timing_monitor import validate_policy
from timing_preregistered_analysis import (ALPHA,BLOCK_LENGTH,BOOTSTRAP_REPLICATES,DRIFT_LIMIT,METRICS,P95_REGRESSION_LIMIT,PIPELINE_IMPROVEMENT_MIN,SEED)

HEX=re.compile('[0-9a-f]{64}\\Z')
ROLES={'cooperative_executable','monitor_binary','old_spirv','candidate_spirv','fixture','analysis_source','monitor_source','supervisor_source'}

def make_manifest(commit,identities,policy):
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('full commit required')
    if set(identities)!=ROLES or any(not isinstance(v,str) or not HEX.fullmatch(v) for v in identities.values()):raise ValueError('exact identity roles required')
    validate_policy(policy)
    controls=dict(policy);controls['allowed_compute_pids']=sorted(policy['allowed_compute_pids'])
    return {'schema':'sintergpu-point-plane-trial-v1','commit':commit,'identities_sha256':dict(sorted(identities.items())),
            'controls':controls,'child_pid_rule':'allow only actual ready child PID bound before gate release in addition to explicit allowed PIDs','run_count':3,'samples_per_run':50,'warmups_per_run':5,'sequence':['old_before','candidate','old_after'],
            'analysis':{'metrics':list(METRICS),'p95':'nearest-rank','p95_limit':P95_REGRESSION_LIMIT,'drift_limit':DRIFT_LIMIT,
                        'minimum_pipeline_improvement':PIPELINE_IMPROVEMENT_MIN,'bootstrap_replicates':BOOTSTRAP_REPLICATES,'bootstrap_seed':SEED,
                        'bootstrap_block_length':BLOCK_LENGTH,'one_sided_alpha_per_test':ALPHA},
            'scope':'preregistered caller claims only; runtime identity, environmental isolation, statistical assumptions and qualification unverified'}

def canonical(manifest):return (json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def write_preregistration(path,manifest):
    path=Path(path);data=canonical(manifest);parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        fd=os.open(path.name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400,dir_fd=parent)
        try:
            with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
            os.fsync(parent)
        except Exception:
            try:os.unlink(path.name,dir_fd=parent);os.fsync(parent)
            except OSError:pass
            raise
    finally:os.close(parent)
    return hashlib.sha256(data).hexdigest()

def verify_preregistration(path,expected_sha):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        with os.fdopen(fd,'rb') as source:data=source.read(1024*1024+1)
    except Exception:raise
    if len(data)>1024*1024 or hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('preregistration mismatch')
    value=json.loads(data)
    if canonical(value)!=data:raise ValueError('noncanonical preregistration')
    return value
