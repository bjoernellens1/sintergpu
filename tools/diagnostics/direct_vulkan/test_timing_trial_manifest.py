# SPDX-License-Identifier: Apache-2.0
import pytest
from timing_trial_manifest import ROLES,make_manifest,write_preregistration,verify_preregistration

def policy():return {'driver':'595.99.02','min_sm_mhz':1900,'max_sm_mhz':2100,'min_memory_mhz':9000,'max_temperature_c':70,'max_power_w':200,'allowed_compute_pids':set()}

def test_canonical_write_and_tamper_detection(tmp_path):
    ids={role:'a'*64 for role in ROLES};m=make_manifest('b'*40,ids,policy());p=tmp_path/'policy.json';sha=write_preregistration(p,m)
    assert verify_preregistration(p,sha)==m
    with pytest.raises(FileExistsError):write_preregistration(p,m)
    p.chmod(0o600);p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='mismatch'):verify_preregistration(p,sha)

def test_policy_roles_and_identity_fail_closed():
    ids={role:'a'*64 for role in ROLES};ids.pop('fixture')
    with pytest.raises(ValueError,match='identity roles'):make_manifest('b'*40,ids,policy())
    with pytest.raises(ValueError,match='full commit'):make_manifest('bad',ids,policy())
    p=policy();p['max_sm_mhz']=10
    with pytest.raises(ValueError,match='control policy'):make_manifest('b'*40,{r:'a'*64 for r in ROLES},p)
